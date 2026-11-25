"""One-off MPC receipts fund consumption without expanding eligible income."""

import numpy as np
import pytest


@pytest.mark.parametrize("exempt_from_support", [False, True])
def test_mpc_receipt_excluded_from_underwriting(test_country, exempt_from_support):
    country = test_country
    eligible_before = country.household_gross_cash_income().copy()
    cash_income = country.households.ts.current("expected_income").copy()
    transfer = np.linspace(10.0, 100.0, cash_income.size)

    country.stage_household_income_shock(transfer, exempt_from_support_test=exempt_from_support)

    # Both consumer and housing underwriting use this common eligible-income base.
    np.testing.assert_array_equal(country.household_gross_cash_income(), eligible_before)
    # Excluding it from underwriting must not remove or duplicate the cash receipt.
    np.testing.assert_allclose(country._apply_household_income_shock(cash_income), cash_income + transfer)
