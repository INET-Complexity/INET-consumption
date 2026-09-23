import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

RUN_MODEL_PATH = Path(__file__).resolve().parents[2] / "run_model"
if str(RUN_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL_PATH))

from src.diagnostics.firm_policy_rate import (  # noqa: E402
    FIRM_POLICY_RATE_SERIES,
    aggregate_firm_policy_rate_irf,
    build_firm_policy_rate_irf_panel,
    load_firm_policy_rate_panel,
    summarize_firm_policy_rate_irf,
)


def _write_firm_h5(path: Path, *, shock: bool) -> None:
    values = {name: np.zeros((3, 2), dtype=float) for name in FIRM_POLICY_RATE_SERIES}
    values["deposits"][:] = [[10.0, -1.0], [10.0, -1.0], [10.0, -1.0]]
    values["credit_budget_hard_obligations"][:] = 5.0
    values["credit_budget_cash_after_hard_obligations"][:] = [[5.0, -2.0], [5.0, -2.0], [5.0, -2.0]]
    values["credit_budget_remaining_internal_finance_after_working_capital"][:] = [[6.0, 0.0]] * 3
    values["credit_budget_capital_costs"][:] = [[4.0, 3.0]] * 3
    values["credit_budget_technical_investment_costs"][:] = 1.0
    values["credit_budget_tfp_costs"][:] = 2.0
    values["credit_budget_investment_budget"][:] = 7.0
    values["target_short_term_credit"][:] = [[2.0, 3.0]] * 3
    values["target_debt_rollover_credit"][:] = [[0.0, 3.0]] * 3
    values["target_overdraft_refinance_credit"][:] = [[0.0, 1.0]] * 3
    values["target_long_term_credit"][:] = [[4.0, 5.0]] * 3
    values["planned_tfp_investment"][:] = [[2.0, 3.0]] * 3
    values["executed_tfp_investment"][:] = [[2.0, 1.0]] * 3
    values["planned_productivity_investment"][:] = values["planned_tfp_investment"]
    values["planned_tfp_investment_expected_costs"][:] = values["planned_tfp_investment"]
    values["feasible_tfp_investment_expected_costs"][:] = values["executed_tfp_investment"]
    values["tfp_investment_effective_cost_rate"][:] = 0.02 + (0.01 if shock else 0.0)
    values["tfp_investment_cash_binding"][:] = [[0.0, 1.0]] * 3
    values["tfp_investment_cap_binding"][:] = 0.0
    values["tfp_multiplier"][:] = 1.0
    values["production"][:] = 100.0
    values["production_nominal"][:] = 100.0
    values["capital_inputs_stock_value"][:] = 20.0
    values["credit_market_firm_lt_binding_reason"][:] = [[0.0, 3.0]] * 3
    values["credit_market_firm_st_binding_reason"][:] = [[0.0, 2.0]] * 3

    if shock:
        values["planned_tfp_investment"][1:] = [[1.0, 2.0], [1.0, 2.0]]
        values["executed_tfp_investment"][1:] = [[1.0, 1.0], [1.0, 1.0]]
        values["planned_productivity_investment"][1:] = values["planned_tfp_investment"][1:]

    with h5py.File(path, "w") as handle:
        for name, value in values.items():
            handle.create_dataset(f"FRA/firms/{name}", data=value)


def test_load_firm_policy_rate_panel_classifies_liquidity_and_derives_gaps(tmp_path):
    path = tmp_path / "firm.h5"
    _write_firm_h5(path, shock=False)

    panel = load_firm_policy_rate_panel(path)

    assert panel.loc[(panel["period"] == 0) & (panel["firm"] == 0), "liquidity_group"].iat[0] == "liquid"
    assert panel.loc[(panel["period"] == 0) & (panel["firm"] == 1), "liquidity_group"].iat[0] == "overdraft"
    assert bool(panel.loc[(panel["period"] == 0) & (panel["firm"] == 0), "debt_stressed"].iat[0]) is False
    assert bool(panel.loc[(panel["period"] == 0) & (panel["firm"] == 1), "debt_stressed"].iat[0]) is True
    assert panel.loc[(panel["period"] == 0) & (panel["firm"] == 0), "capital_funding_gap"].iat[0] == pytest.approx(0.0)
    assert panel.loc[(panel["period"] == 0) & (panel["firm"] == 1), "capital_funding_gap"].iat[0] == pytest.approx(3.0)


def test_firm_policy_rate_irf_aggregates_paired_tfp_and_credit_responses(tmp_path):
    baseline = tmp_path / "baseline.h5"
    shock = tmp_path / "shock.h5"
    _write_firm_h5(baseline, shock=False)
    _write_firm_h5(shock, shock=True)

    panel = build_firm_policy_rate_irf_panel(
        baseline,
        shock,
        seed=12,
        shock_name="policy_rate_100bp_30q",
        shock_kind="policy_rate",
        shock_period=1,
        shock_magnitude=0.01,
        shock_duration=30,
        horizon_periods=2,
        variables=("planned_tfp_investment", "tfp_investment_effective_cost_rate"),
        dscr_enabled=True,
    )
    aggregate = aggregate_firm_policy_rate_irf(panel)
    summary = summarize_firm_policy_rate_irf(aggregate, n_bootstrap=5, random_state=1)

    tfp = summary[
        (summary["variable"] == "planned_tfp_investment")
        & (summary["liquidity_group"] == "all")
        & (summary["debt_stressed"] == "all")
        & (summary["horizon"] == 0)
    ]
    assert tfp["delta_mean"].iat[0] == pytest.approx(-2.0)
    assert tfp["n_seeds"].iat[0] == 1
    assert tfp["negative_seed_share"].iat[0] == pytest.approx(1.0)
