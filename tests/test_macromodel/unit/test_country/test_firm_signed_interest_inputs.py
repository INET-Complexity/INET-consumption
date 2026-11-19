"""Check that country finance inputs preserve deposit receipts and facility costs."""

import numpy as np
import pytest

from macromodel.country import country as country_module


@pytest.mark.parametrize("deposit_interest,operating,expected", [(-10, 0, -10), (-10, 3, -7), (2, 3, 5)])
def test_finance_potential_inputs_include_signed_receipts_and_operating_costs(
    test_country, monkeypatch, deposit_interest, operating, expected
):
    n = test_country.firms.ts.current("n_firms")

    def full(value):
        return np.full(n, value, dtype=float)

    test_country.firms.ts.override_current("firm_settlement_scheduled_interest_due", full(5))
    test_country.firms.ts.override_current("operating_revolving_interest_paid", full(operating))
    monkeypatch.setattr(test_country.firms, "compute_total_wage_obligation", lambda **kwargs: full(0))
    monkeypatch.setattr(
        test_country.firms, "compute_interest_paid_on_deposits", lambda **kwargs: full(deposit_interest)
    )
    monkeypatch.setattr(
        country_module,
        "compute_firm_borrower_credit_room",
        lambda **kwargs: {
            "short_term": full(0),
            "long_term": full(0),
            "total": full(0),
        },
    )
    result = test_country._excess_demand_finance_potential_inputs()
    np.testing.assert_allclose(result["non_loan_interest_obligation_preview"], expected)


def test_signed_country_inputs_reach_finance_capacity_consumer(test_country, monkeypatch):
    firms = test_country.firms
    n = firms.ts.current("n_firms")

    def full(value):
        return np.full(n, value, dtype=float)

    firms.ts.override_current("activity_finance_opening_deposits", full(100))
    firms.ts.override_current("target_debt_rollover_credit", full(0))
    firms.ts.override_current("target_overdraft_refinance_credit", full(0))
    firms.ts.override_current("operating_revolving_interest_paid", full(3))
    monkeypatch.setattr(firms, "compute_total_wage_obligation", lambda **kwargs: full(0))
    monkeypatch.setattr(firms, "compute_interest_paid_on_deposits", lambda **kwargs: full(-10))
    monkeypatch.setattr(
        country_module,
        "compute_firm_borrower_credit_room",
        lambda **kwargs: {
            "short_term": full(0),
            "long_term": full(0),
            "total": full(0),
        },
    )
    firms.compute_excess_demand_finance_potential_capacities(
        **test_country._excess_demand_finance_potential_inputs(),
        q_sold=full(0),
    )
    np.testing.assert_allclose(firms._last_excess_demand_finance_potential["finance_cash"], 107)
