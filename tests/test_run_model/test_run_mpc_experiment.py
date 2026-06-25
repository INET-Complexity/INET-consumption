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
    plot_calls = []

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
                "real_cmpc_4q": [0.4],
                "cmpc_4q": [0.5],
            }
        ),
    )
    monkeypatch.setattr(mpc_runner, "add_mpc_bins", lambda panel: panel.assign(income_bin="all"))
    monkeypatch.setattr(mpc_runner, "summarize_mpc_bins", lambda panel, **kwargs: pd.DataFrame({"rows": [len(panel)]}))

    def fake_write_distribution_plots(panel, output_dir, **kwargs):
        plot_calls.append(kwargs)
        output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mpc_runner, "write_distribution_plots", fake_write_distribution_plots)

    paths = mpc_runner.run_mpc_experiment(
        seeds=[1, 2],
        t_max=10,
        shock_period=2,
        horizon_periods=4,
        shock_fraction=0.01,
        output_dir=str(tmp_path),
        country_iso3="ESP",
        n_jobs=1,
        distribution_plot_kind="box",
        apply_mpc_filters=False,
        mpc_plot_y_quantiles=(0.01, 0.99),
        mpc_winsor_quantiles=(0.05, 0.95),
    )

    assert paths["analysis_dir"] == tmp_path / "analysis"
    assert [call[0] for call in calls] == [1, 2]
    assert plot_calls == [
        {
            "mpc_column": "real_cmpc_4q",
            "plot_kind": "box",
            "y_quantiles": (0.01, 0.99),
            "y_range": None,
            "winsor_quantiles": (0.05, 0.95),
        },
        {
            "mpc_column": "cmpc_4q",
            "plot_kind": "box",
            "y_quantiles": (0.01, 0.99),
            "y_range": None,
            "winsor_quantiles": (0.05, 0.95),
        },
    ]
    assert (tmp_path / "analysis" / "household_mpc_panel.csv").exists()
    assert (tmp_path / "analysis" / "household_mpc_panel_raw.csv").exists()
    assert (tmp_path / "analysis" / "household_mpc_panel_filtered.csv").exists()
    assert (tmp_path / "analysis" / "household_mpc_filter_report.csv").exists()


def test_run_mpc_experiment_routes_filtered_panel_to_default_outputs(tmp_path, monkeypatch):
    data = SimpleNamespace(configuration=SimpleNamespace(year=2014))
    country_cfg = SimpleNamespace(assume_zero_growth=False)
    plot_panels = []

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

    def fake_write_distribution_plots(panel, output_dir, **kwargs):
        plot_panels.append(panel.copy())
        output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mpc_runner, "write_distribution_plots", fake_write_distribution_plots)

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
    assert plot_panels[0]["household_id"].tolist() == [0]


def test_run_mpc_experiment_respects_explicit_nominal_plot_measure(tmp_path, monkeypatch):
    data = SimpleNamespace(configuration=SimpleNamespace(year=2014))
    country_cfg = SimpleNamespace(assume_zero_growth=False)
    plot_calls = []

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
                "real_cmpc_4q": [0.2],
                "cmpc_4q": [0.3],
            }
        ),
    )
    monkeypatch.setattr(mpc_runner, "add_mpc_bins", lambda panel: panel.assign(income_bin="all"))
    monkeypatch.setattr(mpc_runner, "summarize_mpc_bins", lambda panel, **kwargs: pd.DataFrame({"rows": [len(panel)]}))

    def fake_write_distribution_plots(panel, output_dir, **kwargs):
        plot_calls.append(kwargs)
        output_dir.mkdir(parents=True)

    monkeypatch.setattr(mpc_runner, "write_distribution_plots", fake_write_distribution_plots)

    mpc_runner.run_mpc_experiment(
        seeds=[1],
        t_max=10,
        shock_period=2,
        horizon_periods=4,
        shock_fraction=0.01,
        output_dir=tmp_path,
        country_iso3="ESP",
        mpc_plot_measure="nominal",
        mpc_winsor_quantiles=(0.05, 0.95),
    )

    assert [call["mpc_column"] for call in plot_calls] == ["cmpc_4q"]
    assert plot_calls[0]["winsor_quantiles"] == (0.05, 0.95)


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
