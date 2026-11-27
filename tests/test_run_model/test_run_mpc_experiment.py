import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

RUN_MODEL_PATH = Path(__file__).resolve().parents[2] / "run_model"
if str(RUN_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL_PATH))

import run_mpc_experiment as mpc_runner  # noqa: E402


def test_run_mpc_experiment_accepts_n_jobs_in_python_api(tmp_path, monkeypatch):
    """The notebook-facing API should expose the same seed parallelism as the CLI."""
    data = SimpleNamespace(configuration=SimpleNamespace(year=2014))
    country_cfg = SimpleNamespace(assume_zero_growth=False)
    calls = []

    monkeypatch.setattr(
        mpc_runner,
        "_prepare_mpc_inputs",
        lambda **kwargs: (
            SimpleNamespace(),
            data,
            {"ESP": country_cfg},
            "ESP",
            tmp_path,
        ),
    )

    def fake_run_seed_pair(**kwargs):
        calls.append((kwargs["seed"], kwargs["baseline_dir"], kwargs["shock_dir"]))
        return pd.DataFrame({"seed": [kwargs["seed"]], "household_id": [0]})

    monkeypatch.setattr(mpc_runner, "_run_seed_pair", fake_run_seed_pair)
    monkeypatch.setattr(
        mpc_runner,
        "build_household_mpc_panel",
        lambda **kwargs: pd.DataFrame(
            {
                "seed": [kwargs["metadata"]["seed"].iat[0]],
                "mpc_impact": [0.5],
                "cmpc_2p": [0.45],
                "cmpc_3p": [0.48],
                "real_cmpc_4q": [0.4],
                "cmpc_4q": [0.5],
            }
        ),
    )
    monkeypatch.setattr(mpc_runner, "add_mpc_bins", lambda panel: panel.assign(income_bin="all"))
    monkeypatch.setattr(mpc_runner, "summarize_mpc_bins", lambda panel, **kwargs: pd.DataFrame({"rows": [len(panel)]}))

    paths = mpc_runner.run_mpc_experiment(
        seeds=[1, 2],
        t_max=10,
        shock_period=2,
        horizon_periods=4,
        shock_fraction=0.01,
        output_dir=str(tmp_path),
        country_iso3="ESP",
        n_jobs=1,
        apply_mpc_filters=False,
    )

    assert paths["analysis_dir"] == tmp_path / "analysis"
    assert [call[0] for call in calls] == [1, 2]
    assert (tmp_path / "analysis" / "household_mpc_panel.csv").exists()
    assert (tmp_path / "analysis" / "household_mpc_panel_raw.csv").exists()
    assert (tmp_path / "analysis" / "household_mpc_panel_filtered.csv").exists()
    assert (tmp_path / "analysis" / "household_mpc_filter_report.csv").exists()
    default_panel = pd.read_csv(tmp_path / "analysis" / "household_mpc_panel.csv")
    assert {"mpc_impact", "cmpc_2p", "cmpc_3p", "cmpc_4q"}.issubset(default_panel.columns)


def test_run_mpc_experiment_routes_filtered_panel_to_default_outputs(tmp_path, monkeypatch):
    data = SimpleNamespace(configuration=SimpleNamespace(year=2014))
    country_cfg = SimpleNamespace(assume_zero_growth=False)

    monkeypatch.setattr(
        mpc_runner,
        "_prepare_mpc_inputs",
        lambda **kwargs: (
            SimpleNamespace(),
            data,
            {"ESP": country_cfg},
            "ESP",
            tmp_path,
        ),
    )
    monkeypatch.setattr(
        mpc_runner,
        "_run_seed_pair",
        lambda **kwargs: pd.DataFrame({"seed": [kwargs["seed"], kwargs["seed"]], "household_id": [0, 1]}),
    )

    def fake_build_household_mpc_panel(**kwargs):
        return pd.DataFrame(
            {
                "seed": [kwargs["metadata"]["seed"].iat[0], kwargs["metadata"]["seed"].iat[0]],
                "household_id": [0, 1],
                "income": [200.0, 50.0],
                "baseline_consumption_impact": [100.0, 40.0],
                "shock_consumption_impact": [105.0, 42.0],
                "debt_asset_ratio": [0.0, 0.0],
                "cmpc_2p": [0.2, 0.7],
                "cmpc_3p": [0.2, 0.75],
                "real_cmpc_4q": [0.2, 0.8],
                "cmpc_4q": [0.2, 0.8],
            }
        )

    monkeypatch.setattr(mpc_runner, "build_household_mpc_panel", fake_build_household_mpc_panel)
    monkeypatch.setattr(mpc_runner, "add_mpc_bins", lambda panel: panel.assign(income_bin="all"))
    monkeypatch.setattr(
        mpc_runner,
        "summarize_mpc_bins",
        lambda panel, **kwargs: pd.DataFrame(
            {"rows": [len(panel)], "ids": [",".join(panel["household_id"].astype(str))]}
        ),
    )

    mpc_runner.run_mpc_experiment(
        seeds=[1],
        t_max=10,
        shock_period=2,
        horizon_periods=4,
        shock_fraction=0.01,
        output_dir=tmp_path,
        country_iso3="ESP",
    )

    analysis_dir = tmp_path / "analysis"
    raw_panel = pd.read_csv(analysis_dir / "household_mpc_panel_raw.csv")
    default_panel = pd.read_csv(analysis_dir / "household_mpc_panel.csv")
    default_summary = pd.read_csv(analysis_dir / "household_mpc_summary.csv")

    assert raw_panel["household_id"].tolist() == [0, 1]
    assert default_panel["household_id"].tolist() == [0]
    assert default_summary.loc[0, "rows"] == 1
    assert {"cmpc_2p", "cmpc_3p", "cmpc_4q"}.issubset(default_panel.columns)


def test_run_mpc_experiment_respects_explicit_nominal_plot_measure(tmp_path, monkeypatch):
    data = SimpleNamespace(configuration=SimpleNamespace(year=2014))
    country_cfg = SimpleNamespace(assume_zero_growth=False)

    monkeypatch.setattr(
        mpc_runner,
        "_prepare_mpc_inputs",
        lambda **kwargs: (
            SimpleNamespace(),
            data,
            {"ESP": country_cfg},
            "ESP",
            tmp_path,
        ),
    )
    monkeypatch.setattr(
        mpc_runner,
        "_run_seed_pair",
        lambda **kwargs: pd.DataFrame({"seed": [kwargs["seed"]], "household_id": [0]}),
    )
    monkeypatch.setattr(
        mpc_runner,
        "build_household_mpc_panel",
        lambda **kwargs: pd.DataFrame(
            {
                "seed": [kwargs["metadata"]["seed"].iat[0]],
                "household_id": [0],
                "income": [200.0],
                "baseline_consumption_impact": [100.0],
                "shock_consumption_impact": [105.0],
                "debt_asset_ratio": [0.0],
                "cmpc_2p": [0.25],
                "cmpc_3p": [0.28],
                "real_cmpc_4q": [0.2],
                "cmpc_4q": [0.3],
            }
        ),
    )
    monkeypatch.setattr(mpc_runner, "add_mpc_bins", lambda panel: panel.assign(income_bin="all"))
    monkeypatch.setattr(mpc_runner, "summarize_mpc_bins", lambda panel, **kwargs: pd.DataFrame({"rows": [len(panel)]}))

    mpc_runner.run_mpc_experiment(
        seeds=[1],
        t_max=10,
        shock_period=2,
        horizon_periods=4,
        shock_fraction=0.01,
        output_dir=tmp_path,
        country_iso3="ESP",
        mpc_plot_measure="nominal",
    )

    default_summary = pd.read_csv(tmp_path / "analysis" / "household_mpc_summary.csv")
    assert default_summary.loc[0, "rows"] == 1


def test_run_mpc_experiment_rejects_zero_n_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        mpc_runner,
        "_prepare_mpc_inputs",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("inputs should not be prepared")),
    )

    try:
        mpc_runner.run_mpc_experiment(
            seeds=[1],
            t_max=10,
            shock_period=2,
            horizon_periods=4,
            shock_fraction=0.01,
            output_dir=tmp_path,
            n_jobs=0,
        )
    except ValueError as exc:
        assert "n_jobs must be non-zero" in str(exc)
    else:
        raise AssertionError("n_jobs=0 should be rejected")


def test_keep_households_in_stable_effective_labor_state_requires_state_columns():
    panel = pd.DataFrame(
        {
            "household_id": [0],
            "activity_bracket_t0": [1],
            "activity_bracket_t1": [1],
        }
    )

    with pytest.raises(KeyError, match="worked_hours_"):
        mpc_runner._keep_households_in_stable_effective_labor_state(panel)


def test_run_mpc_experiment_strict_filter_excludes_hours_and_income_switchers(tmp_path, monkeypatch):
    data = SimpleNamespace(configuration=SimpleNamespace(year=2014))
    country_cfg = SimpleNamespace(assume_zero_growth=False)

    monkeypatch.setattr(
        mpc_runner,
        "_prepare_mpc_inputs",
        lambda **kwargs: (
            SimpleNamespace(),
            data,
            {"ESP": country_cfg},
            "ESP",
            tmp_path,
        ),
    )
    monkeypatch.setattr(
        mpc_runner,
        "_run_seed_pair",
        lambda **kwargs: pd.DataFrame({"seed": [kwargs["seed"]] * 3, "household_id": [0, 1, 2]}),
    )
    monkeypatch.setattr(
        mpc_runner,
        "build_household_mpc_panel",
        lambda **kwargs: pd.DataFrame(
            {
                "seed": [kwargs["metadata"]["seed"].iat[0]] * 3,
                "household_id": [0, 1, 2],
                "income": [200.0, 200.0, 200.0],
                "baseline_consumption_impact": [100.0, 100.0, 100.0],
                "shock_consumption_impact": [105.0, 105.0, 105.0],
                "debt_asset_ratio": [0.0, 0.0, 0.0],
                "real_cmpc_4q": [0.2, 0.2, 0.2],
                "cmpc_4q": [0.2, 0.2, 0.2],
                "activity_bracket_t0": [1, 1, 1],
                "activity_bracket_t1": [1, 1, 1],
                "activity_bracket_shock_t0": [1, 1, 1],
                "activity_bracket_shock_t1": [1, 1, 1],
                "worked_hours_t0": [1.0, 1.0, 1.0],
                "worked_hours_t1": [1.0, 2.0, 1.0],
                "worked_hours_shock_t0": [1.0, 1.0, 1.0],
                "worked_hours_shock_t1": [1.0, 2.0, 1.0],
                "labor_income_t0": [100.0, 100.0, 100.0],
                "labor_income_t1": [100.0, 100.0, 100.0],
                "labor_income_shock_t0": [100.0, 100.0, 100.0],
                "labor_income_shock_t1": [100.0, 100.0, 120.0],
            }
        ),
    )
    monkeypatch.setattr(mpc_runner, "add_mpc_bins", lambda panel: panel.assign(income_bin="all"))
    monkeypatch.setattr(
        mpc_runner,
        "filter_mpc_panel",
        lambda panel, config, required_mpc_columns=None: (
            panel.copy(),
            pd.DataFrame({"filter": ["identity"], "before": [len(panel)], "dropped": [0], "remaining": [len(panel)]}),
        ),
    )
    monkeypatch.setattr(
        mpc_runner,
        "summarize_mpc_bins",
        lambda panel, **kwargs: pd.DataFrame(
            {"rows": [len(panel)], "ids": [",".join(panel["household_id"].astype(str))]}
        ),
    )

    paths = mpc_runner.run_mpc_experiment(
        seeds=[1],
        t_max=10,
        shock_period=2,
        horizon_periods=4,
        shock_fraction=0.01,
        output_dir=tmp_path,
        country_iso3="ESP",
        apply_mpc_filters=False,
        stable_effective_labor_state_only=True,
    )

    panel = pd.read_csv(paths["analysis_dir"] / "household_mpc_panel.csv")
    summary = pd.read_csv(paths["analysis_dir"] / "household_mpc_summary.csv")

    assert panel["household_id"].tolist() == [0]
    assert str(summary.loc[0, "ids"]) == "0"


def test_run_mpc_experiment_labor_income_tolerance_keeps_all_households_and_flags_identified(tmp_path, monkeypatch):
    data = SimpleNamespace(configuration=SimpleNamespace(year=2014))
    country_cfg = SimpleNamespace(assume_zero_growth=False)

    monkeypatch.setattr(
        mpc_runner,
        "_prepare_mpc_inputs",
        lambda **kwargs: (
            SimpleNamespace(),
            data,
            {"ESP": country_cfg},
            "ESP",
            tmp_path,
        ),
    )
    monkeypatch.setattr(
        mpc_runner,
        "_run_seed_pair",
        lambda **kwargs: pd.DataFrame({"seed": [kwargs["seed"]] * 3, "household_id": [0, 1, 2]}),
    )
    monkeypatch.setattr(
        mpc_runner,
        "build_household_mpc_panel",
        lambda **kwargs: pd.DataFrame(
            {
                "seed": [kwargs["metadata"]["seed"].iat[0]] * 3,
                "household_id": [0, 1, 2],
                "income": [200.0, 200.0, 200.0],
                "baseline_consumption_impact": [100.0, 100.0, 100.0],
                "shock_consumption_impact": [105.0, 105.0, 105.0],
                "debt_asset_ratio": [0.0, 0.0, 0.0],
                "real_cmpc_4q": [0.2, 0.2, 0.2],
                "cmpc_4q": [0.2, 0.2, 0.2],
                "shock_amount": [100.0, 100.0, 100.0],
                "mpc_impact": [0.1, 0.2, 0.3],
                "mpc_regime": ["unconstrained", "unconstrained", "liquidity_constrained"],
                "activity_bracket_t0": [1, 1, 1],
                "activity_bracket_t1": [1, 1, 1],
                "activity_bracket_shock_t0": [1, 1, 1],
                "activity_bracket_shock_t1": [1, 1, 1],
                "worked_hours_t0": [1.0, 1.0, 1.0],
                "worked_hours_t1": [1.0, 2.0, 1.0],
                "worked_hours_shock_t0": [1.0, 1.0, 1.0],
                "worked_hours_shock_t1": [1.0, 2.0, 1.0],
                "labor_income_t0": [100.0, 100.0, 100.0],
                "labor_income_t1": [100.0, 100.0, 100.0],
                "labor_income_shock_t0": [100.0, 100.0, 100.0],
                "labor_income_shock_t1": [100.0, 100.0, 120.0],
            }
        ),
    )
    monkeypatch.setattr(mpc_runner, "add_mpc_bins", lambda panel: panel.assign(income_bin="all"))
    monkeypatch.setattr(
        mpc_runner,
        "filter_mpc_panel",
        lambda panel, config, required_mpc_columns=None: (
            panel.copy(),
            pd.DataFrame({"filter": ["identity"], "before": [len(panel)], "dropped": [0], "remaining": [len(panel)]}),
        ),
    )
    monkeypatch.setattr(
        mpc_runner,
        "summarize_mpc_bins",
        lambda panel, **kwargs: pd.DataFrame(
            {"rows": [len(panel)], "ids": [",".join(panel["household_id"].astype(str))]}
        ),
    )

    paths = mpc_runner.run_mpc_experiment(
        seeds=[1],
        t_max=10,
        shock_period=2,
        horizon_periods=4,
        shock_fraction=0.01,
        output_dir=tmp_path,
        country_iso3="ESP",
        apply_mpc_filters=False,
        labor_income_tolerance=0.05,
    )

    panel = pd.read_csv(paths["analysis_dir"] / "household_mpc_panel.csv")
    summary = pd.read_csv(paths["analysis_dir"] / "household_mpc_summary.csv")

    # Household 1 changes hours over time but not between runs: it stays.
    # Household 2 differs between runs by 20% of the shock: kept, not identified.
    assert panel["household_id"].tolist() == [0, 1, 2]
    # The cumulative-MPC bin summary uses the identified households only.
    assert str(summary.loc[0, "ids"]) == "0,1"
    assert panel["labor_income_gap_share"].tolist() == pytest.approx([0.0, 0.0, 0.2])
    assert panel["cumulative_mpc_identified"].tolist() == [True, True, False]
    regimes = pd.read_csv(paths["analysis_dir"] / "household_mpc_regime_summary.csv").set_index("regime")
    assert regimes.loc["all", "n"] == 3 and regimes.loc["all", "n_cumulative_identified"] == 2
    assert regimes.loc["liquidity_constrained", "n_cumulative_identified"] == 0


def test_run_mpc_experiment_rejects_tolerance_with_a_row_dropping_filter(tmp_path):
    with pytest.raises(ValueError, match="labor_income_tolerance"):
        mpc_runner.run_mpc_experiment(
            seeds=[1],
            t_max=10,
            shock_period=2,
            horizon_periods=4,
            shock_fraction=0.01,
            output_dir=tmp_path,
            stable_effective_labor_state_only=True,
            labor_income_tolerance=0.05,
        )
    with pytest.raises(ValueError, match="non-negative"):
        mpc_runner.run_mpc_experiment(
            seeds=[1],
            t_max=10,
            shock_period=2,
            horizon_periods=4,
            shock_fraction=0.01,
            output_dir=tmp_path,
            labor_income_tolerance=-0.1,
        )


@pytest.mark.parametrize("keep_full_h5", [False, True])
@pytest.mark.parametrize("capture_diagnostics", [False, True])
@pytest.mark.parametrize("shock_period", [0, 1])
def test_run_one_writes_window_and_only_keeps_full_h5_on_request(
    tmp_path, monkeypatch, keep_full_h5, capture_diagnostics, shock_period
):
    import h5py
    import numpy as np

    from macromodel.timeseries import TimeSeries

    class FakeCountry:
        def __init__(self):
            self.households = SimpleNamespace(ts=TimeSeries(income=np.ones(2)))

    class FakeModel:
        def __init__(self):
            self.posthooks = []
            self.prehooks = []
            self.countries = {"FRA": FakeCountry()}

        def run(self):
            for t in range(4):
                for hook in self.posthooks:
                    hook(self, t, 2014, 1)

        def save(self, save_dir, file_name):
            with h5py.File(Path(save_dir) / file_name, "w") as handle:
                handle.create_dataset("FRA/households/consumption", data=np.ones((5, 2)))
                handle.create_dataset("FRA/economy/cpi_fixed_basket", data=np.ones((5, 1)))
                handle.create_dataset("FRA/households/target_consumption", data=np.ones((5, 2)))
                handle.create_dataset("FRA/households/target_consumption_columns", data=np.array([[0, 0], [1, 0]]))

    monkeypatch.setattr(
        mpc_runner.SimulationConfiguration, "model_validate", staticmethod(lambda payload: SimpleNamespace())
    )
    monkeypatch.setattr(mpc_runner.Simulation, "from_datawrapper", staticmethod(lambda **kwargs: FakeModel()))
    monkeypatch.setattr(
        mpc_runner,
        "make_household_metadata",
        lambda country, seed, pre_shock_row: pd.DataFrame({"household_id": [0, 1]}),
    )
    monkeypatch.setattr(
        mpc_runner,
        "make_household_labor_state_snapshot",
        lambda country: {name: np.zeros(2) for name in mpc_runner.HOUSEHOLD_LABOR_STATE_COLUMNS},
    )
    monkeypatch.setattr(
        mpc_runner, "write_selected", lambda model, path, country, paths: model.save(path.parent, path.name)
    )
    support_seen = []
    real_extract = mpc_runner.extract_mpc_window

    def spy(*args, **kwargs):
        support_seen.append(kwargs["household_support"])
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(mpc_runner, "extract_mpc_window", spy)
    seed_dir = tmp_path / "baseline" / "seed-1"
    seed_dir.mkdir(parents=True)
    stale = seed_dir / "multi_country_simulation.h5"
    stale.write_bytes(b"previous full run")

    metadata = mpc_runner._run_one(
        data=SimpleNamespace(configuration=SimpleNamespace(year=2014), time_unit=3),
        country_configurations={
            "FRA": SimpleNamespace(
                model_dump=lambda: {}, household_support={"mode": "minimum_income", "goods_floor": False}
            )
        },
        country_code="FRA",
        seed=1,
        t_max=4,
        output_dir=tmp_path / "baseline",
        shock=False,
        shock_period=shock_period,
        shock_fraction=0.01,
        horizon_periods=2,
        keep_full_h5=keep_full_h5,
        output_variables=["cash"],
        capture_diagnostics=capture_diagnostics,
    )
    assert ("diagnostic_income_t0" in metadata.columns) == capture_diagnostics
    assert support_seen == [{"mode": "minimum_income", "goods_floor": False}]

    window = np.load(seed_dir / "mpc_window.npz")
    assert {k for k in window.files if k.startswith("outcome_")} == {"outcome_cash"}
    assert window["consumption"].shape == (2, 2)
    assert int(window["shock_row"]) == shock_period + 1
    assert not list(seed_dir.glob(".scratch-*"))
    if keep_full_h5:
        with h5py.File(stale, "r") as handle:
            assert "FRA/households/consumption" in handle
    else:
        assert stale.read_bytes() == b"previous full run"  # fixed-path file is never touched


def test_run_one_captures_credit_and_illiquid_asset_diagnostics(tmp_path, monkeypatch):
    """Consumer-credit and illiquid-asset series are exported per offset, to trace unspent transfers."""
    import h5py
    import numpy as np

    from macromodel.timeseries import TimeSeries

    traced = {
        "target_consumption_loans": np.array([5.0, 0.0]),
        "received_consumption_loans": np.array([4.0, 0.0]),
        "illiquid_financial_assets": np.array([10.0, 20.0]),
    }

    class FakeModel:
        def __init__(self):
            self.posthooks = []
            self.prehooks = []
            ts = TimeSeries(income=np.ones(2), **traced)
            self.countries = {"FRA": SimpleNamespace(households=SimpleNamespace(ts=ts))}

        def run(self):
            for t in range(4):
                for hook in self.posthooks:
                    hook(self, t, 2014, 1)

        def save(self, save_dir, file_name):
            with h5py.File(Path(save_dir) / file_name, "w") as handle:
                handle.create_dataset("FRA/households/consumption", data=np.ones((5, 2)))
                handle.create_dataset("FRA/economy/cpi_fixed_basket", data=np.ones((5, 1)))
                handle.create_dataset("FRA/households/target_consumption", data=np.ones((5, 2)))
                handle.create_dataset("FRA/households/target_consumption_columns", data=np.array([[0, 0], [1, 0]]))

    monkeypatch.setattr(
        mpc_runner.SimulationConfiguration, "model_validate", staticmethod(lambda payload: SimpleNamespace())
    )
    monkeypatch.setattr(mpc_runner.Simulation, "from_datawrapper", staticmethod(lambda **kwargs: FakeModel()))
    monkeypatch.setattr(
        mpc_runner,
        "make_household_metadata",
        lambda country, seed, pre_shock_row: pd.DataFrame({"household_id": [0, 1]}),
    )
    monkeypatch.setattr(
        mpc_runner,
        "make_household_labor_state_snapshot",
        lambda country: {name: np.zeros(2) for name in mpc_runner.HOUSEHOLD_LABOR_STATE_COLUMNS},
    )
    monkeypatch.setattr(
        mpc_runner, "write_selected", lambda model, path, country, paths: model.save(path.parent, path.name)
    )
    (tmp_path / "baseline" / "seed-1").mkdir(parents=True)

    metadata = mpc_runner._run_one(
        data=SimpleNamespace(configuration=SimpleNamespace(year=2014), time_unit=3),
        country_configurations={
            "FRA": SimpleNamespace(
                model_dump=lambda: {}, household_support={"mode": "minimum_income", "goods_floor": False}
            )
        },
        country_code="FRA",
        seed=1,
        t_max=4,
        output_dir=tmp_path / "baseline",
        shock=False,
        shock_period=1,
        shock_fraction=0.01,
        horizon_periods=2,
        keep_full_h5=False,
        output_variables=["cash"],
        capture_diagnostics=True,
    )

    for field, values in traced.items():
        assert field in mpc_runner.MPC_DIAGNOSTIC_FIELDS
        for offset in range(3):
            np.testing.assert_allclose(metadata[f"diagnostic_{field}_t{offset}"], values)
