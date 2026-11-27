import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "run_model"))
from src.household_classes import (  # noqa: E402
    LIQUIDITY_GROUP_COLORS,
    LIQUIDITY_GROUP_LABELS,
    LIQUIDITY_GROUP_SHORT,
    classify_liquidity_groups,
    consumer_credit_headroom,
)

# Quarterly income 600 -> tau = 600 / 3 / 2 = 100. DSTI 0.35 -> service room 210 without a mortgage.
# With annuity factor 0.1 the DSTI cap allows a total consumer balance of 2100.
PHI = 0.1
COLUMNS = {
    "income": 600.0,
    "liquid_financial_assets": 0.0,
    "consumption_loan_debt": 0.0,
    "illiquid_financial_assets": 0.0,
    "housing_wealth": 0.0,
    "mortgage_debt": 0.0,
    "diagnostic_actual_mortgage_payment_t0": 0.0,
    "diagnostic_unpaid_mortgage_payment_t0": 0.0,
    "diagnostic_scheduled_consumer_payment_t0": 0.0,
    "diagnostic_consumption_cut_amount_t0": 0.0,
    "baseline_subsistence_floor": 10.0,
}


def _panel(*rows):
    panel = pd.DataFrame([{**COLUMNS, **row} for row in rows])
    panel["seed"] = 1
    return panel


def _groups(*rows, **kwargs):
    kwargs.setdefault("consumer_annuity_factor", PHI)
    return classify_liquidity_groups(_panel(*rows), **kwargs)


def test_constants_cover_the_same_groups():
    assert list(LIQUIDITY_GROUP_LABELS) == ["PHTM", "WHTM", "APU", "ILU", "DW"]
    assert set(LIQUIDITY_GROUP_SHORT) == set(LIQUIDITY_GROUP_LABELS) == set(LIQUIDITY_GROUP_COLORS)


def test_binding_cut_is_constrained_whatever_the_headroom():
    out = _groups(
        {"diagnostic_consumption_cut_amount_t0": 5.0, "liquid_financial_assets": 1000.0},  # a = 0
        {"diagnostic_consumption_cut_amount_t0": 5.0, "housing_wealth": 500.0},  # a > 0
    )
    assert out.liquidity_group.tolist() == ["PHTM", "WHTM"]
    assert out.consumption_cut_binding.tolist() == [True, True]
    assert (out.credit_headroom > 0).all()


def test_cut_tolerance_is_relative_to_the_floor():
    tol = 1e-9 * 10.0
    out = _groups(
        {"diagnostic_consumption_cut_amount_t0": tol, "liquid_financial_assets": 1000.0},
        {"diagnostic_consumption_cut_amount_t0": 2 * tol, "liquid_financial_assets": 1000.0},
    )
    assert out.consumption_cut_binding.tolist() == [False, True]
    assert out.liquidity_group.tolist() == ["APU", "PHTM"]


def test_available_liquidity_boundary_at_tau():
    out = _groups(
        {"consumption_loan_debt": 2100.0},  # K = 0, available 0 -> constrained
        {"consumption_loan_debt": 2000.0},  # K = 100 = tau -> constrained (<=)
        {"consumption_loan_debt": 1999.0},  # K = 101 > tau -> unconstrained
        {"consumption_loan_debt": 1950.0, "liquid_financial_assets": -50.0},  # overdraft: 150 - 50 = tau
        {"consumption_loan_debt": 1950.0, "liquid_financial_assets": -49.0},  # 150 - 49 > tau
    )
    np.testing.assert_allclose(out.credit_headroom, [0.0, 100.0, 101.0, 150.0, 150.0])
    np.testing.assert_allclose(out.available_liquidity, [0.0, 100.0, 101.0, 100.0, 101.0])
    assert out.liquidity_constrained.tolist() == [True, True, False, True, False]
    assert out.liquidity_group.tolist() == ["PHTM", "PHTM", "APU", "PHTM", "APU"]


def test_mortgage_service_paid_and_unpaid_uses_up_dsti_room():
    out = _groups(
        {
            "liquid_financial_assets": 50.0,
            "housing_wealth": 1000.0,
            "mortgage_debt": 500.0,
            "diagnostic_actual_mortgage_payment_t0": 150.0,
            "diagnostic_unpaid_mortgage_payment_t0": 60.0,
        },
        {
            "liquid_financial_assets": 50.0,
            "housing_wealth": 1000.0,
            "mortgage_debt": 500.0,
            "diagnostic_actual_mortgage_payment_t0": 150.0,
        },
    )
    np.testing.assert_allclose(out.credit_headroom, [0.0, 600.0])
    assert out.liquidity_group.tolist() == ["WHTM", "ILU"]


def test_net_illiquid_wealth_boundary_at_zero():
    underwater = {"housing_wealth": 500.0, "mortgage_debt": 500.0}
    out = _groups(
        {**underwater, "consumption_loan_debt": 2100.0},  # constrained, a = 0
        {**underwater, "consumption_loan_debt": 2100.0, "illiquid_financial_assets": 1.0},  # a = 1
        underwater,  # unconstrained, a = 0
        {**underwater, "illiquid_financial_assets": 1.0},  # unconstrained, a = 1, LFA = 0
    )
    np.testing.assert_allclose(out.net_illiquid_wealth, [0.0, 1.0, 0.0, 1.0])
    assert out.liquidity_group.tolist() == ["PHTM", "WHTM", "APU", "ILU"]


def test_liquid_buffer_boundary_splits_illiquid_wealthy_from_doubly_wealthy():
    out = _groups(
        {"housing_wealth": 900.0, "liquid_financial_assets": 100.0},  # LFA = tau -> ILU
        {"housing_wealth": 900.0, "liquid_financial_assets": 101.0},  # LFA > tau -> DW
        {"liquid_financial_assets": 5000.0},  # liquid only, a = 0 -> APU
    )
    np.testing.assert_allclose(out.liquidity_threshold, 100.0)
    assert out.liquidity_group.tolist() == ["ILU", "DW", "APU"]
    assert out.liquidity_group_label.tolist() == [LIQUIDITY_GROUP_LABELS[c] for c in ["ILU", "DW", "APU"]]


def test_pay_period_sets_tau():
    out = _groups({"liquid_financial_assets": 150.0, "housing_wealth": 900.0}, pay_periods_per_quarter=1.5)
    assert out.liquidity_threshold.iloc[0] == pytest.approx(200.0)
    assert out.liquidity_group.iloc[0] == "ILU"


def test_alternative_cut_column():
    panel = _panel({"liquid_financial_assets": 1000.0})
    panel["baseline_consumption_cut_amount"] = 5.0
    default = classify_liquidity_groups(panel, consumer_annuity_factor=PHI)
    alternative = classify_liquidity_groups(
        panel, consumer_annuity_factor=PHI, cut_column="baseline_consumption_cut_amount"
    )
    assert default.liquidity_group.iloc[0] == "APU"
    assert alternative.liquidity_group.iloc[0] == "PHTM"


def test_invalid_income_is_unclassified():
    out = _groups({"income": 0.0}, {"income": np.nan}, {"liquid_financial_assets": np.nan}, {})
    assert out.liquidity_group.isna().tolist() == [True, True, True, False]
    assert out.liquidity_constrained.isna().tolist() == [True, True, True, False]


def test_annuity_factor_is_estimated_within_seed():
    panel = pd.concat(
        [
            _panel(
                {"consumption_loan_debt": 1000.0, "diagnostic_scheduled_consumer_payment_t0": 100.0},
                {"consumption_loan_debt": 1000.0, "diagnostic_scheduled_consumer_payment_t0": 50.0},
                {"consumption_loan_debt": 1000.0, "diagnostic_scheduled_consumer_payment_t0": 150.0},
                {},  # non-borrower: excluded from the median
            ),
            _panel({"consumption_loan_debt": 1000.0, "diagnostic_scheduled_consumer_payment_t0": 70.0}).assign(seed=2),
        ],
        ignore_index=True,
    )
    headroom = consumer_credit_headroom(panel)
    # seed 1: phi = 0.1 -> 210 / 0.1 - balance; seed 2: phi = 0.07 -> 3000 - 1000
    np.testing.assert_allclose(headroom, [1100.0, 1100.0, 1100.0, 2100.0, 2000.0])


def test_annuity_factor_errors():
    with pytest.raises(ValueError, match="no serviced consumer loans"):
        consumer_credit_headroom(_panel({}))
    with pytest.raises(ValueError, match="Pass consumer_annuity_factor"):
        consumer_credit_headroom(_panel({}).drop(columns="diagnostic_scheduled_consumer_payment_t0"))
    with pytest.raises(ValueError, match="must be positive"):
        consumer_credit_headroom(_panel({}), consumer_annuity_factor=0.0)
