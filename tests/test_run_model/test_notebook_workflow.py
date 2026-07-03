import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
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

    def fake_load_country_configuration(path, country_iso3):
        return {"path": path, "country_iso3": country_iso3}

    monkeypatch.setattr(nw, "load_country_configuration", fake_load_country_configuration)
    monkeypatch.setattr(nw, "split_country_configs", lambda value: {"split": value})
    monkeypatch.setattr(nw, "DataConfiguration", lambda **kwargs: kwargs)

    assert nw._load_country_config(cfg) == {
        "path": config_dir / "country_config_ESP.yaml",
        "country_iso3": "ESP",
    }
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
    monkeypatch.setattr(nw, "_requires_cfc_rate_cache_rebuild", lambda data, data_config: False)
    monkeypatch.setattr(nw, "_requires_population_schema_cache_rebuild", lambda data, country_iso3: False)

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


def test_build_permanent_income_log_ratio_decomposition_df_reads_saved_household_diagnostics(tmp_path):
    h5_path = tmp_path / "simulation_ESP.h5"
    with h5py.File(h5_path, "w") as handle:
        household_group = handle.create_group("ESP").create_group("households")
        household_group.create_dataset(
            "target_consumption_permanent_income_log_ratio",
            data=np.array([[0.3, 0.5], [0.2, 0.4]]),
        )
        household_group.create_dataset(
            "target_consumption_permanent_income_log_ratio_individual",
            data=np.array([[0.2, 0.4], [0.1, 0.3]]),
        )
        household_group.create_dataset(
            "target_consumption_permanent_income_log_ratio_common",
            data=np.array([[0.1, 0.1], [0.1, 0.1]]),
        )

        result = nw.build_permanent_income_log_ratio_decomposition_df(h5_path, reducer="mean")

        expected = pd.DataFrame(
            {
                "ln_y_p_over_y": [0.4, 0.3],
                "zeta_times_posterior_mean": [0.3, 0.2],
                "common_log_ratio": [0.1, 0.1],
            },
            index=pd.RangeIndex(2, name="period"),
        )
    pd.testing.assert_frame_equal(result, expected)
    assert result.attrs["country_code"] == "ESP"
    assert result.attrs["model_h5_path"] == str(h5_path)
    assert result.attrs["reducer"] == "mean"


def test_build_permanent_income_log_ratio_decomposition_df_can_include_log_real_pc_income(tmp_path):
    h5_path = tmp_path / "simulation_ESP.h5"
    with h5py.File(h5_path, "w") as handle:
        country_group = handle.create_group("ESP")
        household_group = country_group.create_group("households")
        household_group.create_dataset(
            "target_consumption_permanent_income_log_ratio",
            data=np.array([[0.3, 0.5], [0.2, 0.4]]),
        )
        household_group.create_dataset(
            "target_consumption_permanent_income_log_ratio_individual",
            data=np.array([[0.2, 0.4], [0.1, 0.3]]),
        )
        household_group.create_dataset(
            "target_consumption_permanent_income_log_ratio_common",
            data=np.array([[0.1, 0.1], [0.1, 0.1]]),
        )
        household_group.create_dataset(
            "income",
            data=np.array([[100.0, 100.0], [110.0, 110.0]]),
        )
        country_group.create_group("economy").create_dataset(
            "cpi_fixed_basket",
            data=np.array([[1.0], [1.0]]),
        )
        country_group.create_group("individuals").create_dataset(
            "n_individuals",
            data=np.array([[2.0, 2.0]]),
        )

    result = nw.build_permanent_income_log_ratio_decomposition_df(
        h5_path,
        reducer="mean",
        include_log_real_pc_income=True,
    )

    assert result["log_real_pc_income_t"].tolist() == [np.log(100.0), np.log(110.0)]


def test_plot_permanent_income_log_ratio_decomposition_returns_three_traces(tmp_path):
    h5_path = tmp_path / "simulation_ESP.h5"
    with h5py.File(h5_path, "w") as handle:
        household_group = handle.create_group("ESP").create_group("households")
        household_group.create_dataset(
            "target_consumption_permanent_income_log_ratio",
            data=np.array([[0.3, 0.5], [0.2, 0.4]]),
        )
        household_group.create_dataset(
            "target_consumption_permanent_income_log_ratio_individual",
            data=np.array([[0.2, 0.4], [0.1, 0.3]]),
        )
        household_group.create_dataset(
            "target_consumption_permanent_income_log_ratio_common",
            data=np.array([[0.1, 0.1], [0.1, 0.1]]),
        )

    fig = nw.plot_permanent_income_log_ratio_decomposition(h5_path, show=False)

    assert [trace.name for trace in fig.data] == [
        "ln(y^p / y)",
        "zeta * posterior_mean",
        "common_log_ratio",
    ]


def test_plot_permanent_income_log_ratio_decomposition_can_select_subset_of_columns(tmp_path):
    h5_path = tmp_path / "simulation_ESP.h5"
    with h5py.File(h5_path, "w") as handle:
        household_group = handle.create_group("ESP").create_group("households")
        household_group.create_dataset(
            "target_consumption_permanent_income_log_ratio",
            data=np.array([[0.3, 0.5], [0.2, 0.4]]),
        )
        household_group.create_dataset(
            "target_consumption_permanent_income_log_ratio_individual",
            data=np.array([[0.2, 0.4], [0.1, 0.3]]),
        )
        household_group.create_dataset(
            "target_consumption_permanent_income_log_ratio_common",
            data=np.array([[0.1, 0.1], [0.1, 0.1]]),
        )

    fig = nw.plot_permanent_income_log_ratio_decomposition(
        h5_path,
        columns=["ln_y_p_over_y", "common_log_ratio"],
        show=False,
    )

    assert [trace.name for trace in fig.data] == [
        "ln(y^p / y)",
        "common_log_ratio",
    ]


def test_plot_permanent_income_log_ratio_decomposition_can_plot_log_real_pc_income(tmp_path):
    h5_path = tmp_path / "simulation_ESP.h5"
    with h5py.File(h5_path, "w") as handle:
        country_group = handle.create_group("ESP")
        household_group = country_group.create_group("households")
        household_group.create_dataset(
            "target_consumption_permanent_income_log_ratio",
            data=np.array([[0.3, 0.5], [0.2, 0.4]]),
        )
        household_group.create_dataset(
            "target_consumption_permanent_income_log_ratio_individual",
            data=np.array([[0.2, 0.4], [0.1, 0.3]]),
        )
        household_group.create_dataset(
            "target_consumption_permanent_income_log_ratio_common",
            data=np.array([[0.1, 0.1], [0.1, 0.1]]),
        )
        household_group.create_dataset(
            "income",
            data=np.array([[100.0, 100.0], [110.0, 110.0]]),
        )
        country_group.create_group("economy").create_dataset(
            "cpi_fixed_basket",
            data=np.array([[1.0], [1.0]]),
        )
        country_group.create_group("individuals").create_dataset(
            "n_individuals",
            data=np.array([[2.0, 2.0]]),
        )

    fig = nw.plot_permanent_income_log_ratio_decomposition(
        h5_path,
        columns=["ln_y_p_over_y", "log_real_pc_income_t"],
        show=False,
    )

    assert [trace.name for trace in fig.data] == [
        "ln(y^p / y)",
        "log_real_pc_income_t",
    ]
    np.testing.assert_allclose(fig.data[1].y, np.array([np.log(100.0), np.log(110.0)]))


def test_plot_permanent_income_log_ratio_decomposition_rejects_unknown_columns(tmp_path):
    h5_path = tmp_path / "simulation_ESP.h5"
    with h5py.File(h5_path, "w") as handle:
        household_group = handle.create_group("ESP").create_group("households")
        household_group.create_dataset(
            "target_consumption_permanent_income_log_ratio",
            data=np.array([[0.3, 0.5], [0.2, 0.4]]),
        )
        household_group.create_dataset(
            "target_consumption_permanent_income_log_ratio_individual",
            data=np.array([[0.2, 0.4], [0.1, 0.3]]),
        )
        household_group.create_dataset(
            "target_consumption_permanent_income_log_ratio_common",
            data=np.array([[0.1, 0.1], [0.1, 0.1]]),
        )

    try:
        nw.plot_permanent_income_log_ratio_decomposition(
            h5_path,
            columns=["not_a_real_column"],
            show=False,
        )
    except ValueError as exc:
        assert "Unknown columns requested" in str(exc)
    else:
        raise AssertionError("unknown decomposition columns should be rejected")


def test_build_permanent_income_forecast_contribution_table_returns_regressor_contributions(monkeypatch):
    class FakeSimulation:
        def __init__(self, countries):
            self.countries = countries

    monkeypatch.setattr(nw, "Simulation", FakeSimulation)

    country = SimpleNamespace(
        start_period=pd.Period("2014Q1", freq="Q"),
        _permanent_income_forecast_inputs=SimpleNamespace(
            coefficient_table=pd.DataFrame({"coefficient": [2.0, -1.0]}, index=["constant", "time_trend"]),
            hac_covariance=pd.DataFrame(
                np.eye(2), index=["constant", "time_trend"], columns=["constant", "time_trend"]
            ),
            residual_variance=1.0,
        ),
        _permanent_income_design_matrix=pd.DataFrame(
            {"constant": [1.0], "time_trend": [0.0]},
            index=pd.PeriodIndex([pd.Period("2014Q1", freq="Q")]),
        ),
        economy=SimpleNamespace(
            ts=SimpleNamespace(
                dicts={
                    "cpi_fixed_basket": [[100.0], [100.0], [100.0], [100.0], [100.0]],
                    "unemployment_rate": [[0.05], [0.05], [0.05], [0.05], [0.05]],
                }
            )
        ),
        central_bank=SimpleNamespace(
            ts=SimpleNamespace(dicts={"policy_rate": [[0.01], [0.01], [0.01], [0.01], [0.01]]})
        ),
        individuals=SimpleNamespace(
            ts=SimpleNamespace(dicts={"n_individuals": [[10.0], [10.0], [10.0], [10.0], [10.0]]})
        ),
        households=SimpleNamespace(
            ts=SimpleNamespace(
                dicts={
                    "income": [
                        np.array([1000.0]),
                        np.array([1000.0]),
                        np.array([1000.0]),
                        np.array([1000.0]),
                        np.array([1000.0]),
                    ]
                }
            )
        ),
    )

    monkeypatch.setattr(
        nw,
        "build_permanent_income_forecast_regressors",
        lambda **kwargs: pd.Series({"constant": 1.0, "time_trend": float(kwargs["sources"].current_period.quarter)}),
    )
    monkeypatch.setattr(
        nw,
        "forecast_common_permanent_income",
        lambda x_t, forecast_inputs: SimpleNamespace(
            point_forecast=float((x_t * forecast_inputs.coefficient_table["coefficient"]).sum())
        ),
    )

    result = nw.build_permanent_income_forecast_contribution_table(
        FakeSimulation({"ESP": country}),
        country_code="ESP",
        periods=[1, 2],
    )

    assert list(result.columns) == [
        "period",
        "date",
        "regressor",
        "simulation_source",
        "is_fixed",
        "x_t",
        "coefficient",
        "contribution",
        "point_forecast",
    ]
    assert result["period"].tolist() == [1, 1, 2, 2]
    assert result["regressor"].tolist() == ["constant", "time_trend", "constant", "time_trend"]
    assert (
        result["simulation_source"].tolist() == ["frozen_design_matrix_initial_period", "simulation_period_index"] * 2
    )
    assert result["is_fixed"].tolist() == [True, False, True, False]
    assert result["contribution"].tolist() == [2.0, -2.0, 2.0, -3.0]


def test_build_permanent_income_forecast_contribution_table_can_exclude_fixed_regressors(monkeypatch):
    class FakeSimulation:
        def __init__(self, countries):
            self.countries = countries

    monkeypatch.setattr(nw, "Simulation", FakeSimulation)

    country = SimpleNamespace(
        start_period=pd.Period("2014Q1", freq="Q"),
        _permanent_income_forecast_inputs=SimpleNamespace(
            coefficient_table=pd.DataFrame({"coefficient": [2.0, -1.0]}, index=["constant", "time_trend"]),
            hac_covariance=pd.DataFrame(
                np.eye(2), index=["constant", "time_trend"], columns=["constant", "time_trend"]
            ),
            residual_variance=1.0,
        ),
        _permanent_income_design_matrix=pd.DataFrame(
            {"constant": [1.0], "time_trend": [0.0]},
            index=pd.PeriodIndex([pd.Period("2014Q1", freq="Q")]),
        ),
        economy=SimpleNamespace(
            ts=SimpleNamespace(
                dicts={
                    "cpi_fixed_basket": [[100.0], [100.0], [100.0], [100.0], [100.0]],
                    "unemployment_rate": [[0.05], [0.05], [0.05], [0.05], [0.05]],
                }
            )
        ),
        central_bank=SimpleNamespace(
            ts=SimpleNamespace(dicts={"policy_rate": [[0.01], [0.01], [0.01], [0.01], [0.01]]})
        ),
        individuals=SimpleNamespace(
            ts=SimpleNamespace(dicts={"n_individuals": [[10.0], [10.0], [10.0], [10.0], [10.0]]})
        ),
        households=SimpleNamespace(
            ts=SimpleNamespace(
                dicts={
                    "income": [
                        np.array([1000.0]),
                        np.array([1000.0]),
                        np.array([1000.0]),
                        np.array([1000.0]),
                        np.array([1000.0]),
                    ]
                }
            )
        ),
    )

    monkeypatch.setattr(
        nw,
        "build_permanent_income_forecast_regressors",
        lambda **kwargs: pd.Series({"constant": 1.0, "time_trend": float(kwargs["sources"].current_period.quarter)}),
    )
    monkeypatch.setattr(
        nw,
        "forecast_common_permanent_income",
        lambda x_t, forecast_inputs: SimpleNamespace(
            point_forecast=float((x_t * forecast_inputs.coefficient_table["coefficient"]).sum())
        ),
    )

    result = nw.build_permanent_income_forecast_contribution_table(
        FakeSimulation({"ESP": country}),
        country_code="ESP",
        periods=[1],
        include_fixed=False,
    )

    assert result["regressor"].tolist() == ["time_trend"]
    assert result["simulation_source"].tolist() == ["simulation_period_index"]
    assert result["is_fixed"].tolist() == [False]
    assert result.attrs["include_fixed"] is False


def test_build_permanent_income_forecast_contribution_table_recovers_missing_source_map(monkeypatch):
    class FakeSimulation:
        def __init__(self, countries):
            self.countries = countries

    monkeypatch.setattr(nw, "Simulation", FakeSimulation)

    country = SimpleNamespace(
        start_period=pd.Period("2014Q1", freq="Q"),
        _permanent_income_forecast_inputs=SimpleNamespace(
            coefficient_table=pd.DataFrame({"coefficient": [-1.0]}, index=["time_trend"]),
            hac_covariance=pd.DataFrame(np.eye(1), index=["time_trend"], columns=["time_trend"]),
            residual_variance=1.0,
        ),
        _permanent_income_design_matrix=pd.DataFrame(
            {"time_trend": [0.0]},
            index=pd.PeriodIndex([pd.Period("2014Q1", freq="Q")]),
        ),
        economy=SimpleNamespace(
            ts=SimpleNamespace(
                dicts={
                    "cpi_fixed_basket": [[100.0]],
                    "unemployment_rate": [[0.05]],
                }
            )
        ),
        central_bank=SimpleNamespace(ts=SimpleNamespace(dicts={"policy_rate": [[0.01]]})),
        individuals=SimpleNamespace(ts=SimpleNamespace(dicts={"n_individuals": [[10.0]]})),
        households=SimpleNamespace(ts=SimpleNamespace(dicts={"income": [np.array([1000.0])]})),
    )

    monkeypatch.setattr(
        nw,
        "build_permanent_income_forecast_regressors",
        lambda **kwargs: pd.Series({"time_trend": 1.0}),
    )
    monkeypatch.setattr(
        nw,
        "forecast_common_permanent_income",
        lambda x_t, forecast_inputs: SimpleNamespace(
            point_forecast=float((x_t * forecast_inputs.coefficient_table["coefficient"]).sum())
        ),
    )
    monkeypatch.delattr(nw, "FORECAST_READER_TO_SIMULATION_SOURCE_NAME", raising=False)

    result = nw.build_permanent_income_forecast_contribution_table(
        FakeSimulation({"ESP": country}),
        country_code="ESP",
        periods=[0],
        include_fixed=False,
    )

    assert result["regressor"].tolist() == ["time_trend"]
    assert result["simulation_source"].tolist() == ["simulation_period_index"]
    assert result["is_fixed"].tolist() == [False]
