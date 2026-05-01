import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

RUN_MODEL_PATH = Path(__file__).resolve().parents[2] / "run_model"
if str(RUN_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL_PATH))

from src import notebook_workflow as nw  # noqa: E402


def _fake_env_config(tmp_path, country_iso3="ESP"):
    raw_data_path = tmp_path / "raw"
    output_path = tmp_path / "output"
    config_dir = tmp_path / "config"
    raw_data_path.mkdir()
    output_path.mkdir()
    config_dir.mkdir()
    return SimpleNamespace(
        config_dir=config_dir,
        country_iso3=country_iso3,
        seed=45,
        t_max=100,
        raw_data_path=raw_data_path,
        output_path=output_path,
    )


def _summary_config():
    return SimpleNamespace(
        firms=SimpleNamespace(
            functions=SimpleNamespace(
                productivity_growth=SimpleNamespace(name="SimpleTFPGrowth"),
                wage_setter=SimpleNamespace(
                    name="WorkEffortFirmWageSetter",
                    parameters={"labour_market_tightness_markup_scale": 0.05},
                ),
            )
        ),
        labour_market=SimpleNamespace(
            functions=SimpleNamespace(
                clearing=SimpleNamespace(
                    name="DefaultLabourMarketClearer",
                    parameters={"allow_switching_industries": True},
                )
            )
        ),
        central_bank=SimpleNamespace(functions=SimpleNamespace(policy_rate=SimpleNamespace(name="SmoothTaylorRule"))),
        central_government=SimpleNamespace(
            functions=SimpleNamespace(
                social_benefits=SimpleNamespace(name="ConstantSocialBenefitsSetter", parameters={})
            )
        ),
        government_entities=SimpleNamespace(
            functions=SimpleNamespace(
                consumption=SimpleNamespace(
                    name="ExpectedGrowthGovernmentConsumptionSetter",
                    parameters={"sectoral_weights": "initial_fixed"},
                )
            )
        ),
        assume_zero_noise=False,
    )


def test_config_loaders_use_country_iso3_paths(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "country_config_ESP.yaml").write_text("ESP:\n  marker: 1\n")
    (config_dir / "data_config_ESP.yaml").write_text("country_configs:\n  ESP: {}\n")
    cfg = SimpleNamespace(config_dir=config_dir, country_iso3="ESP")

    monkeypatch.setattr(nw, "CountryConfiguration", lambda **kwargs: kwargs)
    monkeypatch.setattr(nw, "split_country_configs", lambda value: {"split": value})
    monkeypatch.setattr(nw, "DataConfiguration", lambda **kwargs: kwargs)

    assert nw._load_country_config(cfg) == {"marker": 1}
    assert nw._load_data_config(cfg) == {"country_configs": {"split": {"ESP": {}}}}


def test_prepare_data_uses_deterministic_cache_path(tmp_path, monkeypatch):
    env_cfg = _fake_env_config(tmp_path)
    national_accounts = pd.DataFrame({"GDP (Value)": [1.0]})
    data = SimpleNamespace(
        n_industries=2,
        synthetic_countries={
            "ESP": SimpleNamespace(exogenous_data=SimpleNamespace(national_accounts=national_accounts))
        },
    )

    class FakeDataWrapper:
        @staticmethod
        def init_from_pickle(path):
            assert path == str(env_cfg.output_path / "data.pkl")
            return data

        @staticmethod
        def from_config(**kwargs):
            raise AssertionError("prepare_data should load the cache when it exists")

    (env_cfg.output_path / "data.pkl").write_bytes(b"cached")
    monkeypatch.setattr(nw.Config, "from_env", classmethod(lambda cls: env_cfg))
    monkeypatch.setattr(nw, "_load_data_config", lambda cfg: SimpleNamespace(name="data_config"))
    monkeypatch.setattr(nw, "DataWrapper", FakeDataWrapper)

    result = nw.prepare_data(nw.NotebookRunConfig(country_iso3="ESP"))

    assert result.data_pkl_path == env_cfg.output_path / "data.pkl"
    assert result.output_dir == env_cfg.output_path
    assert result.raw_data_path == env_cfg.raw_data_path
    assert result.national_accounts.equals(national_accounts)


def test_build_country_config_aligns_and_applies_overrides(tmp_path, monkeypatch):
    env_cfg = _fake_env_config(tmp_path)
    country_cfg = _summary_config()
    data = SimpleNamespace(
        n_industries=4,
        synthetic_countries={
            "ESP": SimpleNamespace(
                population=SimpleNamespace(
                    household_data=[object(), object()],
                    individual_data=[object(), object(), object()],
                ),
                firms=SimpleNamespace(firm_data=[object(), object(), object(), object()]),
                banks=SimpleNamespace(number_of_banks=1),
                government_entities=SimpleNamespace(number_of_entities=3),
                central_bank=object(),
                central_government=object(),
            )
        },
        synthetic_rest_of_the_world=object(),
    )

    monkeypatch.setattr(nw.Config, "from_env", classmethod(lambda cls: env_cfg))
    monkeypatch.setattr(nw, "_load_country_config", lambda cfg: country_cfg)

    def fake_align(cfg, n_industries):
        assert n_industries == 4
        cfg.aligned = True
        return cfg

    monkeypatch.setattr(nw, "align_country_configuration_to_data", fake_align)

    configs = nw.build_country_config(
        data=data,
        config=nw.NotebookRunConfig(country_iso3="ESP"),
        overrides={
            "labour_market.functions.clearing.name": "ReservationWageBindingDefaultLabourMarketClearer",
            "firms.functions.wage_setter.parameters['labour_market_tightness_markup_scale']": 0.5,
        },
    )

    assert configs["ESP"].aligned is True
    assert configs["ESP"].labour_market.functions.clearing.name == "ReservationWageBindingDefaultLabourMarketClearer"
    assert configs["ESP"].firms.functions.wage_setter.parameters["labour_market_tightness_markup_scale"] == 0.5


def test_summarize_agent_counts_reads_synthetic_country_counts():
    data = SimpleNamespace(
        n_industries=18,
        synthetic_countries={
            "ESP": SimpleNamespace(
                population=SimpleNamespace(household_data=range(2), individual_data=range(5)),
                firms=SimpleNamespace(firm_data=range(3)),
                banks=SimpleNamespace(number_of_banks=1),
                government_entities=SimpleNamespace(number_of_entities=7),
                central_bank=object(),
                central_government=object(),
            )
        },
        synthetic_rest_of_the_world=object(),
    )

    assert nw.summarize_agent_counts(data, "ESP") == {
        "industries": 18,
        "firms": 3,
        "households": 2,
        "individuals": 5,
        "banks": 1,
        "government_entities": 7,
        "central_bank": 1,
        "central_government": 1,
        "rest_of_world": 1,
    }


def test_benchmark_overrides_default_to_no_overrides():
    config = nw.NotebookRunConfig()

    assert config.benchmark_overrides is None


def test_run_benchmark_uses_cached_dataframe_without_rerun(tmp_path, monkeypatch):
    env_cfg = _fake_env_config(tmp_path)
    cached = pd.DataFrame({"gdp": [1.0, 2.0]})
    cached.attrs["benchmark_spec"] = {"country_iso3": "ESP", "cached": True}
    cached_path = env_cfg.output_path / "ESP_df_benchmark.pkl"
    cached.to_pickle(cached_path)

    class RaisingDataWrapper:
        @staticmethod
        def init_from_pickle(path):
            raise AssertionError("benchmark should not load data when dataframe cache is valid")

        @staticmethod
        def from_config(**kwargs):
            raise AssertionError("benchmark should not rebuild data when dataframe cache is valid")

    monkeypatch.setattr(nw.Config, "from_env", classmethod(lambda cls: env_cfg))
    monkeypatch.setattr(nw, "DataWrapper", RaisingDataWrapper)

    result = nw.run_benchmark(nw.NotebookRunConfig(country_iso3="ESP", force_rerun_benchmark=False))

    assert result.loaded_from_cache is True
    assert result.model is None
    assert result.df_benchmark.equals(cached)
    assert result.benchmark_spec == {"country_iso3": "ESP", "cached": True}


def test_run_benchmark_records_empty_overrides_when_none(tmp_path, monkeypatch):
    env_cfg = _fake_env_config(tmp_path)
    data = SimpleNamespace(n_industries=2)
    df_base = pd.DataFrame({"gdp": [1.0]})

    class FakeDataWrapper:
        @staticmethod
        def init_from_pickle(path):
            return data

        @staticmethod
        def from_config(**kwargs):
            return SimpleNamespace(save=lambda path: None)

    monkeypatch.setattr(nw.Config, "from_env", classmethod(lambda cls: env_cfg))
    monkeypatch.setattr(nw, "DataWrapper", FakeDataWrapper)
    monkeypatch.setattr(nw, "_load_data_config", lambda cfg: SimpleNamespace(name="data_config"))
    monkeypatch.setattr(nw, "build_country_config", lambda data, config, overrides: {"ESP": "country_cfg"})
    monkeypatch.setattr(
        nw,
        "run_single_simulation",
        lambda data, country_configurations, config: SimpleNamespace(model="model", df_base=df_base),
    )

    result = nw.run_benchmark(
        nw.NotebookRunConfig(
            country_iso3="ESP",
            force_rerun_benchmark=True,
            benchmark_overrides=None,
        )
    )

    assert result.benchmark_spec["overrides"] == {}
