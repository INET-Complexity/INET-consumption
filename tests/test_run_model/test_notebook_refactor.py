import json
import sys
from pathlib import Path
from types import SimpleNamespace

import nbformat
import numpy as np

RUN_MODEL_PATH = Path(__file__).resolve().parents[2] / "run_model"
if str(RUN_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL_PATH))

from src.diagnostics.firm_finance import (  # noqa: E402
    FIRM_FINANCE_SERIES,
    build_firm_balance_sheet_ratios,
    summarize_firm_balance_sheet_ratios,
)
from src.diagnostics.income import permanent_income_by_decile  # noqa: E402
from config import CALIBRATED_CONSUMPTION_OVERRIDES, SCENARIO_PRESETS  # noqa: E402
from src.notebook_state import (  # noqa: E402
    NotebookRunState,
    validate_notebook_state,
    write_run_manifest,
)
from src.notebook_workflow import NotebookRunConfig  # noqa: E402


def test_refactored_notebook_suite_and_legacy_banner_are_present():
    expected = {
        "run_model.ipynb",
        "run_model_exploration.ipynb",
        "run_sensitivity.ipynb",
        "run_mpc.ipynb",
        "run_irf.ipynb",
        "run_model_legacy_2026-08-18.ipynb",
    }
    assert expected.issubset({path.name for path in RUN_MODEL_PATH.glob("*.ipynb")})

    legacy = nbformat.read(RUN_MODEL_PATH / "run_model_legacy_2026-08-18.ipynb", as_version=4)
    assert "LEGACY SNAPSHOT" in legacy.cells[0].source
    assert "DO NOT USE FOR NEW WORK" in legacy.cells[0].source

    runner = nbformat.read(RUN_MODEL_PATH / "run_model.ipynb", as_version=4)
    assert len(runner.cells) < 25
    runner_source = "\n".join(cell.source for cell in runner.cells)
    assert "run_notebook_workflow" in runner_source
    assert "plot_cumulative_insolvent_firms_by_sector" in runner_source
    assert "plot_permanent_income_log_ratio_decomposition" in runner_source
    assert "run_parameter_sensitivity" not in runner_source
    assert "run_mpc_experiment" not in runner_source
    assert "run_irf_experiment" not in runner_source
    assert not (RUN_MODEL_PATH / "run_model_diagnostics.ipynb").exists()


def test_calibrated_override_preset_retains_notebook_values():
    assert SCENARIO_PRESETS["calibrated_consumption"] is CALIBRATED_CONSUMPTION_OVERRIDES
    assert CALIBRATED_CONSUMPTION_OVERRIDES[
        "households.functions.consumption.parameters.['long_run_intercept']"
    ] == -0.6
    assert CALIBRATED_CONSUMPTION_OVERRIDES[
        "households.functions.consumption.parameters.['partial_adjustment_speed']"
    ] == 0.56
    assert CALIBRATED_CONSUMPTION_OVERRIDES[
        "households.functions.consumption.parameters.['long_run_mpc_upper_bound']"
    ] == 2


def test_permanent_income_by_decile_handles_tied_income_deterministically():
    households = SimpleNamespace(
        ts=SimpleNamespace(
            income=[np.ones(20) * 10_000],
            target_consumption_permanent_income_log_ratio=[np.linspace(-0.1, 0.1, 20)],
        )
    )

    result = permanent_income_by_decile(
        households,
        scale=5_000,
        period=0,
        periods_per_year=4,
        n_quantiles=10,
    )

    assert result["quantile"].astype(int).tolist() == list(range(1, 11))
    assert len(result) == 10
    assert np.isfinite(result["mean_permanent_income"]).all()


def test_firm_finance_diagnostic_builds_ratios_and_summary():
    arrays = {
        name: [np.array([2.0, -1.0]), np.array([4.0, 2.0])]
        for name in FIRM_FINANCE_SERIES
    }
    arrays.update(
        {
            "equity": [np.array([4.0, 2.0]), np.array([8.0, 4.0])],
            "profits": [np.array([1.0, -1.0]), np.array([2.0, 1.0])],
            "debt": [np.array([2.0, 3.0]), np.array([4.0, 2.0])],
            "capital_inputs_stock_value": [np.array([10.0, 8.0]), np.array([12.0, 9.0])],
            "inventory": [np.array([1.0, 2.0]), np.array([1.0, 2.0])],
            "price": [np.ones(2), np.ones(2)],
            "credit_budget_hard_obligations": [np.array([2.0, 0.0]), np.array([4.0, 2.0])],
            "credit_budget_remaining_internal_finance_after_working_capital": [
                np.array([3.0, 0.0]),
                np.array([3.0, 1.0]),
            ],
            "credit_budget_capital_costs": [np.array([2.0, 2.0]), np.array([4.0, 2.0])],
            "credit_budget_technical_investment_costs": [np.array([1.0, 1.0]), np.array([1.0, 1.0])],
            "credit_budget_tfp_costs": [np.array([1.0, 1.0]), np.array([1.0, 1.0])],
            "target_long_term_credit": [np.array([0.0, 2.0]), np.array([2.0, 0.0])],
        }
    )
    model = SimpleNamespace(
        countries={"FRA": SimpleNamespace(firms=SimpleNamespace(ts=SimpleNamespace(dicts=arrays)))}
    )

    panel = build_firm_balance_sheet_ratios(model, "FRA")
    summary = summarize_firm_balance_sheet_ratios(panel, periods=[0, 1], money_scale=1.0)

    assert len(panel) == 4
    assert panel.loc[(panel["period"] == 0) & (panel["firm"] == 1), "positive_deposits"].iat[0] == 0
    assert summary.index.tolist() == [0, 1]
    assert summary.loc[0, "firms_with_capital_gap"] == 0.5


def test_notebook_state_validation_and_manifest(tmp_path):
    h5_path = tmp_path / "simulation.h5"
    h5_path.touch()
    model = SimpleNamespace(countries={"ESP": object()})
    prepared = SimpleNamespace(
        cfg=SimpleNamespace(country_iso3="ESP", seed=7, t_max=3),
        output_dir=tmp_path,
        data_pkl_path=tmp_path / "data.pkl",
    )
    simulation = SimpleNamespace(
        model=model,
        df_base=__import__("pandas").DataFrame({"gdp": [1.0]}),
        model_h5_path=h5_path,
    )
    state = NotebookRunState(
        config=NotebookRunConfig(seed=7, t_max=3, country_iso3="ESP"),
        scenario_name="test",
        scenario_overrides={"x.y": 1},
        prepared=prepared,
        country_configurations={"ESP": object()},
        simulation=simulation,
        benchmark=None,
    )

    validate_notebook_state(state)
    path = write_run_manifest(state)
    manifest = json.loads(path.read_text())

    assert manifest["country_code"] == "ESP"
    assert manifest["scenario"] == {"name": "test", "overrides": {"x.y": 1}}
    assert manifest["outputs"]["model_h5"] == str(h5_path)
