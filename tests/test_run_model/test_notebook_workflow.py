import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

RUN_MODEL_PATH = Path(__file__).resolve().parents[2] / "run_model"
if str(RUN_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL_PATH))

from src import notebook_workflow as nw  # noqa: E402
from src.helpers import align_country_configuration_to_data  # noqa: E402


def _fake_env_config(tmp_path, country_iso3="ESP"):
    raw_data_path = tmp_path / "raw"
    output_path = tmp_path / "output"
    config_dir = tmp_path / "config"
    raw_data_path.mkdir()
    output_path.mkdir()
    config_dir.mkdir()
    (config_dir / f"data_config_{country_iso3}.yaml").write_text("country_configs:\n  ESP: {}\n")
    (config_dir / f"country_config_{country_iso3}.yaml").write_text("ESP: {}\n")
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
            parameters=SimpleNamespace(capital_inputs_delay=[0] * 18, depreciation_rates=[0.0] * 18),
            functions=SimpleNamespace(
                productivity_growth=SimpleNamespace(
                    name="SimpleTFPGrowth",
                    parameters={"investment_effectiveness": 0.003},
                ),
                productivity_investment_planner=SimpleNamespace(
                    name="TargetIntensityTFPInvestmentPlanner",
                    parameters={"n_firms": 18},
                ),
                wage_setter=SimpleNamespace(
                    name="WorkEffortFirmWageSetter",
                    parameters={"labour_market_tightness_markup_scale": 0.05},
                ),
            ),
            substitution_bundles=list(range(18)),
        ),
        households=SimpleNamespace(substitution_bundles=list(range(18))),
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


def test_requires_cfc_rate_cache_rebuild_detects_stale_basis():
    data_config = SimpleNamespace(
        country_configs={
            "ESP": SimpleNamespace(
                firms_configuration=SimpleNamespace(capital_depreciation_accounting_mode="eurostat_cfc")
            )
        }
    )

    stale_data = SimpleNamespace(
        synthetic_countries={"ESP": SimpleNamespace(firms=SimpleNamespace(capital_depreciation_rate_basis="output"))}
    )
    current_data = SimpleNamespace(
        synthetic_countries={
            "ESP": SimpleNamespace(firms=SimpleNamespace(capital_depreciation_rate_basis="capital_stock"))
        }
    )

    assert nw._requires_cfc_rate_cache_rebuild(stale_data, data_config)
    assert not nw._requires_cfc_rate_cache_rebuild(current_data, data_config)


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
                firms=SimpleNamespace(firm_data=range(7)),
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

    def fake_align(cfg, n_industries, n_firms=None):
        assert n_industries == 4
        assert n_firms == 7
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


def test_align_country_configuration_uses_firm_count_for_productivity_planner():
    country_cfg = _summary_config()

    aligned = align_country_configuration_to_data(
        country_cfg,
        n_industries=18,
        n_firms=821,
    )

    assert aligned.firms.functions.productivity_investment_planner.parameters["n_firms"] == 821
    assert aligned.firms.functions.productivity_investment_planner.parameters["investment_effectiveness"] == 0.003
    assert aligned.firms.parameters.capital_inputs_delay == [0] * 18
    assert aligned.firms.parameters.depreciation_rates == [0.0] * 18


def test_planner_effectiveness_follows_realised_tfp_phi_after_overrides():
    country_cfg = _summary_config()

    nw.apply_country_config_overrides(
        country_cfg,
        {
            "firms.functions.productivity_investment_planner.parameters['investment_effectiveness']": 0.01,
            "firms.functions.productivity_growth.parameters['investment_effectiveness']": 0.004,
        },
    )

    planner_params = country_cfg.firms.functions.productivity_investment_planner.parameters
    assert planner_params["investment_effectiveness"] == 0.004


def test_align_country_configuration_falls_back_to_industry_count_for_productivity_planner():
    country_cfg = _summary_config()

    aligned = align_country_configuration_to_data(country_cfg, n_industries=18)

    assert aligned.firms.functions.productivity_investment_planner.parameters["n_firms"] == 18


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


def test_run_single_simulation_rejects_path_like_model_file_name(tmp_path, monkeypatch):
    env_cfg = _fake_env_config(tmp_path)

    monkeypatch.setattr(nw.Config, "from_env", classmethod(lambda cls: env_cfg))

    try:
        nw.run_single_simulation(
            data=SimpleNamespace(),
            country_configurations={"ESP": "country_cfg"},
            config=nw.NotebookRunConfig(country_iso3="ESP", model_file_name="nested/model.h5"),
        )
    except ValueError as exc:
        assert "model_file_name must be a file name" in str(exc)
    else:
        raise AssertionError("path-like model_file_name should be rejected")


def test_benchmark_overrides_default_to_no_overrides():
    config = nw.NotebookRunConfig()

    assert config.benchmark_overrides is None


def test_run_benchmark_uses_cached_dataframe_without_rerun(tmp_path, monkeypatch):
    env_cfg = _fake_env_config(tmp_path)
    cached = pd.DataFrame({"gdp": [1.0, 2.0]})
    monkeypatch.setattr(nw.Config, "from_env", classmethod(lambda cls: env_cfg))
    data_cache_spec = nw._benchmark_data_cache_spec(
        env_cfg,
        env_cfg.raw_data_path,
        nw.NotebookRunConfig(country_iso3="ESP", force_rerun_benchmark=False),
    )
    cached.attrs["benchmark_spec"] = {
        "country_iso3": "ESP",
        "seed": 32,
        "t_max": 50,
        "raw_data_path": str(env_cfg.raw_data_path),
        "single_hfcs_survey": False,
        "data_cache_spec": data_cache_spec,
        "n_industries": 2,
        "overrides": {},
    }
    cached_path = env_cfg.output_path / "ESP_df_benchmark.pkl"
    cached.to_pickle(cached_path)

    class RaisingDataWrapper:
        @staticmethod
        def init_from_pickle(path):
            raise AssertionError("benchmark should not load data when dataframe cache is valid")

        @staticmethod
        def from_config(**kwargs):
            raise AssertionError("benchmark should not rebuild data when dataframe cache is valid")

    monkeypatch.setattr(nw, "DataWrapper", RaisingDataWrapper)

    result = nw.run_benchmark(nw.NotebookRunConfig(country_iso3="ESP", force_rerun_benchmark=False))

    assert result.loaded_from_cache is True
    assert result.model is None
    assert result.df_benchmark.equals(cached)
    assert result.benchmark_spec == cached.attrs["benchmark_spec"]


def test_run_benchmark_rebuilds_when_cached_dataframe_spec_differs(tmp_path, monkeypatch):
    env_cfg = _fake_env_config(tmp_path)
    stale = pd.DataFrame({"gdp": [99.0]})
    stale.attrs["benchmark_spec"] = {
        "country_iso3": "ESP",
        "seed": 999,
        "t_max": 50,
        "raw_data_path": str(env_cfg.raw_data_path),
        "single_hfcs_survey": False,
        "n_industries": 2,
        "overrides": {},
    }
    stale.to_pickle(env_cfg.output_path / "ESP_df_benchmark.pkl")
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

    result = nw.run_benchmark(nw.NotebookRunConfig(country_iso3="ESP", seed=32))

    assert result.loaded_from_cache is False
    assert result.df_benchmark.equals(df_base)
    assert result.benchmark_spec["seed"] == 32


def test_run_benchmark_rebuilds_when_cached_dataframe_config_fingerprint_differs(tmp_path, monkeypatch):
    env_cfg = _fake_env_config(tmp_path)
    run_config = nw.NotebookRunConfig(country_iso3="ESP")
    stale_data_cache_spec = nw._benchmark_data_cache_spec(env_cfg, env_cfg.raw_data_path, run_config)
    (env_cfg.config_dir / "country_config_ESP.yaml").write_text("ESP:\n  changed: true\n")
    stale = pd.DataFrame({"gdp": [99.0]})
    stale.attrs["benchmark_spec"] = {
        "country_iso3": "ESP",
        "seed": 32,
        "t_max": 50,
        "raw_data_path": str(env_cfg.raw_data_path),
        "single_hfcs_survey": False,
        "data_cache_spec": stale_data_cache_spec,
        "n_industries": 2,
        "overrides": {},
    }
    stale.to_pickle(env_cfg.output_path / "ESP_df_benchmark.pkl")
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

    result = nw.run_benchmark(run_config)

    assert result.loaded_from_cache is False
    assert result.df_benchmark.equals(df_base)


def test_run_benchmark_rebuilds_when_data_cache_metadata_differs(tmp_path, monkeypatch):
    env_cfg = _fake_env_config(tmp_path)
    (env_cfg.output_path / "data_benchmark.pkl").write_bytes(b"stale")
    (env_cfg.output_path / "data_benchmark.pkl.meta.json").write_text('{"country_iso3": "ITA"}')
    data = SimpleNamespace(n_industries=2)
    calls = {"from_config": 0, "init_from_pickle": 0}

    class FakeDataWrapper:
        @staticmethod
        def init_from_pickle(path):
            calls["init_from_pickle"] += 1
            return data

        @staticmethod
        def from_config(**kwargs):
            calls["from_config"] += 1
            return SimpleNamespace(save=lambda path: None)

    monkeypatch.setattr(nw.Config, "from_env", classmethod(lambda cls: env_cfg))
    monkeypatch.setattr(nw, "DataWrapper", FakeDataWrapper)
    monkeypatch.setattr(nw, "_load_data_config", lambda cfg: SimpleNamespace(name="data_config"))
    monkeypatch.setattr(nw, "build_country_config", lambda data, config, overrides: {"ESP": "country_cfg"})
    monkeypatch.setattr(
        nw,
        "run_single_simulation",
        lambda data, country_configurations, config: SimpleNamespace(model="model", df_base=pd.DataFrame({"gdp": [1]})),
    )

    nw.run_benchmark(nw.NotebookRunConfig(country_iso3="ESP"))

    assert calls == {"from_config": 1, "init_from_pickle": 1}


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
