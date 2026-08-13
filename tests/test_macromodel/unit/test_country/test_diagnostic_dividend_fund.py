from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.country.country import Country
from macromodel.country.diagnostic_dividend_fund import (
    compute_cash_feasible_firm_distribution_settlement,
    compute_diagnostic_dividend_fund,
    compute_dividend_income_after_withholding,
)
from macromodel.timeseries import TimeSeries


def _diagnostic_inputs() -> dict[str, np.ndarray]:
    return {
        "beginning_ifa": np.array([10.0, 90.0]),
        "firm_cash_profit_after_settlement": np.array([12.0, 8.0]),
        "firm_closing_deposits": np.array([10.0, 20.0]),
        "firm_unpaid_interest": np.array([0.0, 0.0]),
        "firm_closing_principal_arrears": np.array([0.0, 0.0]),
        "firm_residual_overdraft_exposure": np.array([0.0, 0.0]),
        "firm_default_flag": np.array([False, False]),
        "bank_cash_distributable_profits": np.array([5.0, -1.0]),
        "ownership_quota": np.array([0.25, 0.75]),
    }


def test_diagnostic_dividend_fund_uses_fixed_quota_and_closing_cash() -> None:
    result = compute_diagnostic_dividend_fund(**_diagnostic_inputs())

    np.testing.assert_allclose(result.firm_distributable_profit_candidate, [10.0, 8.0])
    np.testing.assert_allclose(result.bank_distributable_profit_candidate, [5.0, 0.0])
    np.testing.assert_allclose(result.hypothetical_firm_distribution, [4.5, 13.5])
    np.testing.assert_allclose(result.hypothetical_bank_distribution, [1.25, 3.75])
    assert result.fund_identity_error == pytest.approx(0.0)


def test_cash_feasible_settlement_is_junior_to_revolving_repayment() -> None:
    settled, shortfall = compute_cash_feasible_firm_distribution_settlement(
        declared_distribution=np.array([8.0, 6.0]),
        cash_available_after_debt_service=np.array([10.0, 4.0]),
        operating_revolving_repayment=np.array([3.0, 1.0]),
    )

    np.testing.assert_allclose(settled, [7.0, 3.0])
    np.testing.assert_allclose(shortfall, [1.0, 3.0])


def test_dividend_income_is_net_of_source_withholding() -> None:
    net_income, income_tax_withheld = compute_dividend_income_after_withholding(
        gross_distribution=np.array([10.0, 30.0]),
        income_tax_rate=0.25,
    )

    np.testing.assert_allclose(net_income, [7.5, 22.5])
    np.testing.assert_allclose(income_tax_withheld, [2.5, 7.5])
    np.testing.assert_allclose(net_income + income_tax_withheld, [10.0, 30.0])


@pytest.mark.parametrize("income_tax_rate", (-0.01, 1.01, np.nan))
def test_dividend_income_withholding_rejects_invalid_income_tax_rate(income_tax_rate: float) -> None:
    with pytest.raises(ValueError, match="income_tax_rate"):
        compute_dividend_income_after_withholding(
            gross_distribution=np.array([1.0]),
            income_tax_rate=income_tax_rate,
        )


def test_diagnostic_dividend_fund_with_no_owners_declares_no_distribution() -> None:
    inputs = _diagnostic_inputs()
    inputs["ownership_quota"] = np.zeros(2)

    result = compute_diagnostic_dividend_fund(**inputs)

    np.testing.assert_allclose(result.hypothetical_total_distribution, [0.0, 0.0])
    assert result.fund_identity_error == pytest.approx(0.0)


def test_settlement_receipts_reconcile_separately_without_deposit_credit() -> None:
    country = Country.__new__(Country)
    country.households = SimpleNamespace(
        states={"dividend_fund_ownership_quota": np.array([0.25, 0.75])},
        ts=TimeSeries(
            dividend_fund_settled_firm_distribution=np.zeros(2),
            dividend_fund_settled_bank_distribution=np.zeros(2),
            dividend_fund_ownership_quota=np.array([0.25, 0.75]),
            dividend_fund_total_settled_distribution=[0.0],
            dividend_fund_quota_sum=[0.0],
            dividend_fund_firm_settlement_identity_error=[0.0],
            dividend_fund_bank_settlement_identity_error=[0.0],
            dividend_fund_settlement_identity_error=[0.0],
        ),
    )
    country.firms = SimpleNamespace(ts=TimeSeries(dividend_fund_settlement_debit=np.array([8.0, 4.0])))
    country.banks = SimpleNamespace(ts=TimeSeries(dividend_fund_settlement_debit=np.array([6.0])))

    country.record_dividend_fund_settlement_receipts()

    np.testing.assert_allclose(country.households.ts.current("dividend_fund_settled_firm_distribution"), [3.0, 9.0])
    np.testing.assert_allclose(country.households.ts.current("dividend_fund_settled_bank_distribution"), [1.5, 4.5])
    np.testing.assert_allclose(country.households.ts.current("dividend_fund_ownership_quota"), [0.25, 0.75])
    assert country.households.ts.current("dividend_fund_settlement_identity_error")[0] == pytest.approx(0.0)


@pytest.mark.parametrize("quota", (np.array([0.20, 0.70]), np.array([0.20, -0.20])))
def test_settlement_receipts_reject_invalid_ownership_quota(quota: np.ndarray) -> None:
    country = Country.__new__(Country)
    country.households = SimpleNamespace(states={"dividend_fund_ownership_quota": quota})
    country.firms = SimpleNamespace(ts=TimeSeries(dividend_fund_settlement_debit=np.array([1.0])))
    country.banks = SimpleNamespace(ts=TimeSeries(dividend_fund_settlement_debit=np.array([0.0])))

    with pytest.raises(ValueError, match="ownership quotas"):
        country.record_dividend_fund_settlement_receipts()
