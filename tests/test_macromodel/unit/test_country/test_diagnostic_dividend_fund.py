from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.country.country import Country
from macromodel.country.diagnostic_dividend_fund import (
    compute_cash_feasible_firm_distribution_settlement,
    compute_diagnostic_dividend_fund,
)
from macromodel.timeseries import TimeSeries


def _compute(**overrides):
    inputs = {
        "beginning_ifa": np.array([-10.0, 100.0, 300.0]),
        "firm_cash_profit_after_settlement": np.array([50.0, 40.0, -5.0]),
        "firm_closing_deposits": np.array([100.0, 100.0, 100.0]),
        "firm_unpaid_interest": np.array([0.0, 1.0, 0.0]),
        "firm_closing_principal_arrears": np.zeros(3),
        "firm_residual_overdraft_exposure": np.zeros(3),
        "firm_default_flag": np.zeros(3, dtype=bool),
        "bank_cash_distributable_profits": np.array([25.0, -2.0]),
        "bank_resolved_flag": np.zeros(2, dtype=bool),
        "ownership_quota": np.array([0.0, 0.25, 0.75]),
    }
    inputs.update(overrides)
    return compute_diagnostic_dividend_fund(**inputs)


def test__firm_distribution_settlement_is_junior_and_cash_feasible():
    settled, shortfall = compute_cash_feasible_firm_distribution_settlement(
        declared_distribution=np.array([20.0, 20.0, 20.0]),
        cash_available_after_debt_service=np.array([30.0, 15.0, 5.0]),
        operating_revolving_repayment=np.array([5.0, 5.0, 10.0]),
    )

    np.testing.assert_array_equal(settled, [20.0, 10.0, 0.0])
    np.testing.assert_array_equal(shortfall, [0.0, 10.0, 20.0])
    np.testing.assert_array_equal(settled + shortfall, [20.0, 20.0, 20.0])


def test__diagnostic_dividend_fund_allocates_separate_cash_backed_candidates():
    result = _compute()

    np.testing.assert_allclose(result.ifa_weight, [0.0, 0.25, 0.75])
    np.testing.assert_allclose(result.firm_distributable_profit_candidate, [50.0, 0.0, 0.0])
    np.testing.assert_array_equal(result.firm_distress_gate_passed, [True, False, True])
    np.testing.assert_allclose(result.bank_distributable_profit_candidate, [25.0, 0.0])
    np.testing.assert_allclose(result.hypothetical_firm_distribution, [0.0, 12.5, 37.5])
    np.testing.assert_allclose(result.hypothetical_bank_distribution, [0.0, 6.25, 18.75])
    assert result.hypothetical_fund_inflow == 75.0
    assert result.hypothetical_fund_outflow == 75.0
    assert result.aggregate_distribution_yield == 75.0 / 400.0
    assert result.distribution_by_ifa_quintile.sum() == 75.0
    np.testing.assert_allclose(result.distribution_by_ifa_quintile, [0.0, 0.0, 18.75, 0.0, 56.25])
    assert abs(result.fund_identity_error) < 1e-12


def test__diagnostic_dividend_fund_uses_fixed_ownership_quota_instead_of_current_ifa():
    result = _compute(ownership_quota=np.array([1.0, 0.0, 0.0]))

    np.testing.assert_allclose(result.ifa_weight, [0.0, 0.25, 0.75])
    np.testing.assert_allclose(result.ownership_quota, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(result.hypothetical_total_distribution, [75.0, 0.0, 0.0])


def test__diagnostic_dividend_fund_rejects_non_conserving_ownership_quota():
    with pytest.raises(ValueError, match="sum to one"):
        _compute(ownership_quota=np.array([0.2, 0.2, 0.2]))


def test__zero_aggregate_direct_shares_create_no_fallback_owner_or_distribution():
    result = _compute(ownership_quota=np.zeros(3))

    assert result.total_firm_distributable_profit_candidate == 50.0
    assert result.total_bank_distributable_profit_candidate == 25.0
    np.testing.assert_array_equal(result.ownership_quota, np.zeros(3))
    np.testing.assert_array_equal(result.hypothetical_total_distribution, np.zeros(3))
    assert result.hypothetical_fund_inflow == 0.0


def test__resolved_bank_cannot_declare_a_new_distribution():
    result = _compute(bank_resolved_flag=np.array([True, False]))

    np.testing.assert_array_equal(result.bank_distributable_profit_candidate, [0.0, 0.0])
    np.testing.assert_array_equal(result.hypothetical_bank_distribution, np.zeros(3))


def test__diagnostic_dividend_fund_zero_holdings_produce_no_inflow_or_receipts():
    result = _compute(
        beginning_ifa=np.array([0.0, -1.0, 0.0]),
        ownership_quota=np.zeros(3),
    )

    np.testing.assert_array_equal(result.ifa_weight, np.zeros(3))
    np.testing.assert_array_equal(result.hypothetical_total_distribution, np.zeros(3))
    assert result.total_firm_distributable_profit_candidate == 50.0
    assert result.total_bank_distributable_profit_candidate == 25.0
    assert result.hypothetical_fund_inflow == 0.0
    assert result.hypothetical_fund_outflow == 0.0
    assert result.aggregate_distribution_yield == 0.0
    assert result.fund_identity_error == 0.0


def test__diagnostic_dividend_fund_zero_candidates_produce_no_receipts():
    result = _compute(
        firm_cash_profit_after_settlement=np.zeros(3),
        bank_cash_distributable_profits=np.zeros(2),
    )

    np.testing.assert_array_equal(result.hypothetical_total_distribution, np.zeros(3))
    assert result.hypothetical_fund_inflow == 0.0
    assert result.hypothetical_fund_outflow == 0.0


def test__diagnostic_dividend_fund_caps_firm_candidate_at_closing_cash():
    result = _compute(firm_closing_deposits=np.array([10.0, 100.0, 100.0]))

    np.testing.assert_array_equal(result.firm_distributable_profit_candidate, [10.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("beginning_ifa", np.array([0.0, np.nan, 1.0])),
        ("firm_cash_profit_after_settlement", np.array([0.0, np.inf, 1.0])),
        ("bank_cash_distributable_profits", np.array([np.nan, 1.0])),
    ],
)
def test__diagnostic_dividend_fund_rejects_non_finite_inputs(name, value):
    with pytest.raises(ValueError, match="non-finite"):
        _compute(**{name: value})


@pytest.mark.parametrize(
    "name",
    [
        "firm_unpaid_interest",
        "firm_closing_principal_arrears",
        "firm_residual_overdraft_exposure",
    ],
)
def test__diagnostic_dividend_fund_rejects_negative_distress_state(name):
    with pytest.raises(ValueError, match="materially negative"):
        _compute(**{name: np.array([0.0, -1.0, 0.0])})


def test__diagnostic_dividend_fund_accepts_empty_agent_populations():
    result = compute_diagnostic_dividend_fund(
        beginning_ifa=np.array([]),
        firm_cash_profit_after_settlement=np.array([]),
        firm_closing_deposits=np.array([]),
        firm_unpaid_interest=np.array([]),
        firm_closing_principal_arrears=np.array([]),
        firm_residual_overdraft_exposure=np.array([]),
        firm_default_flag=np.array([], dtype=bool),
        bank_cash_distributable_profits=np.array([]),
        bank_resolved_flag=np.array([], dtype=bool),
        ownership_quota=np.array([]),
    )

    assert result.hypothetical_fund_outflow == 0.0
    assert result.positive_ifa_household_share == 0.0
    assert result.firm_distress_gate_passed_share == 0.0
    np.testing.assert_array_equal(result.distribution_by_ifa_quintile, np.zeros(5))


def test__record_diagnostic_dividend_fund_uses_cash_flow_and_preserves_core_series():
    households_ts = TimeSeries(
        n_households=3,
        wealth_other_financial_assets=np.array([100.0, 200.0, 300.0]),
        income=np.array([1.0, 2.0, 3.0]),
        income_financial_assets_distribution=np.zeros(3),
        wealth_deposits=np.array([10.0, 20.0, 30.0]),
    )
    households_ts.wealth_other_financial_assets.append(np.array([999.0, 999.0, 999.0]))
    firms_ts = TimeSeries(
        nominal_amount_sold_in_lcu=np.array([100.0]),
        total_wage=np.array([10.0]),
        nominal_amount_spent_in_lcu=np.array([[5.0, 4.0]]),
        taxes_paid_on_production=np.array([3.0]),
        corporate_taxes_paid=np.array([2.0]),
        interest_paid=np.array([6.0]),
        debt_installments=np.array([7.0]),
        operating_revolving_repayment=np.array([8.0]),
        deposits=np.array([10.0]),
        firm_settlement_unpaid_interest=np.zeros(1),
        firm_settlement_closing_principal_arrears=np.zeros(1),
        firm_settlement_residual_overdraft_exposure=np.zeros(1),
        firm_settlement_default_flag=np.zeros(1, dtype=bool),
        dividend_fund_cash_distributable_profit_candidate=np.zeros(1),
        dividend_fund_distress_gate_passed=np.zeros(1, dtype=bool),
        dividend_fund_declared_distribution=np.zeros(1),
        dividend_fund_retained_capacity=np.zeros(1),
        dividend_fund_settlement_debit=np.zeros(1),
        dividend_fund_settlement_shortfall=np.zeros(1),
    )
    firms_ts.deposits.append(np.array([60.0]))
    banks_ts = TimeSeries(
        cash_distributable_profits=np.array([20.0]),
        dividend_fund_cash_distributable_profit_candidate=np.zeros(1),
        dividend_fund_declared_distribution=np.zeros(1),
        dividend_fund_retained_capacity=np.zeros(1),
        dividend_fund_settlement_debit=np.zeros(1),
    )
    banks_ts.cash_distributable_profits.append(np.array([40.0]))
    firms = SimpleNamespace(
        ts=firms_ts,
        direct_tfp_investment_cash_expense=lambda: np.array([5.0]),
    )
    country = SimpleNamespace(
        households=SimpleNamespace(
            ts=households_ts,
            functions={
                "wealth": SimpleNamespace(
                    dividend_fund_payout_ratio=0.25,
                    dividend_fund_empirical_proxy_ratio=0.30,
                )
            },
            states={
                "dividend_fund_ownership_quota": np.array([0.0, 0.25, 0.75]),
                "dividend_fund_initial_direct_share_fraction": np.array([0.0, 0.5, 1.0]),
            },
        ),
        firms=firms,
        banks=SimpleNamespace(ts=banks_ts, states={"is_insolvent": np.zeros(1, dtype=bool)}),
    )
    watched = {
        "income": households_ts.current("income").copy(),
        "wealth_deposits": households_ts.current("wealth_deposits").copy(),
        "wealth_other_financial_assets": households_ts.current("wealth_other_financial_assets").copy(),
        "firm_deposits": firms_ts.current("deposits").copy(),
        "bank_cash_profits": banks_ts.current("cash_distributable_profits").copy(),
    }

    result = Country.record_diagnostic_dividend_fund(country)

    # 100 receipts - 10 wages - 9 goods - 5 TFP - 3 production tax
    # - 2 corporate tax - 6 interest - 7 principal - 8 facility repayment.
    np.testing.assert_array_equal(result.firm_distributable_profit_candidate, [50.0])
    np.testing.assert_array_equal(result.beginning_ifa, [100.0, 200.0, 300.0])
    np.testing.assert_allclose(result.ownership_quota, [0.0, 0.25, 0.75])
    np.testing.assert_array_equal(firms_ts.current("dividend_fund_cash_distributable_profit_candidate"), [50.0])
    np.testing.assert_array_equal(banks_ts.current("dividend_fund_cash_distributable_profit_candidate"), [40.0])
    assert len(firms_ts.dividend_fund_cash_distributable_profit_candidate) == len(firms_ts.deposits)
    assert len(firms_ts.dividend_fund_distress_gate_passed) == len(firms_ts.deposits)
    assert len(banks_ts.dividend_fund_cash_distributable_profit_candidate) == len(banks_ts.cash_distributable_profits)
    np.testing.assert_array_equal(households_ts.current("income"), watched["income"])
    np.testing.assert_array_equal(households_ts.current("wealth_deposits"), watched["wealth_deposits"])
    np.testing.assert_array_equal(
        households_ts.current("wealth_other_financial_assets"),
        watched["wealth_other_financial_assets"],
    )
    np.testing.assert_array_equal(firms_ts.current("deposits"), watched["firm_deposits"])
    np.testing.assert_array_equal(banks_ts.current("cash_distributable_profits"), watched["bank_cash_profits"])
