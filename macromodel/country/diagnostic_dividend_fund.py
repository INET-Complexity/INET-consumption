"""Ownership-based firm and bank profit-distribution calculations."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiagnosticDividendFundResult:
    """Eligible profit capacity and its allocation across household owners."""

    beginning_ifa: np.ndarray
    ownership_quota: np.ndarray
    firm_distributable_profit_candidate: np.ndarray
    firm_distress_gate_passed: np.ndarray
    bank_distributable_profit_candidate: np.ndarray
    hypothetical_firm_distribution: np.ndarray
    hypothetical_bank_distribution: np.ndarray
    hypothetical_total_distribution: np.ndarray
    firm_identity_error: float
    bank_identity_error: float
    fund_identity_error: float


def _finite_1d(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional agent array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _boolean_1d(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional agent array.")
    if array.dtype != np.bool_:
        numeric = _finite_1d(name, array)
        if not np.all((numeric == 0.0) | (numeric == 1.0)):
            raise ValueError(f"{name} must contain only boolean or 0/1 values.")
    return array.astype(bool)


def _non_negative_state_1d(name: str, values: np.ndarray) -> np.ndarray:
    array = _finite_1d(name, values)
    if np.any(array < -1e-9):
        raise ValueError(f"{name} contains materially negative accounting state.")
    return np.maximum(array, 0.0)


def compute_dividend_income_after_withholding(
    *,
    gross_distribution: np.ndarray,
    income_tax_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return household net dividend income and tax withheld at source.

    The payer-side settlement debit is the gross distribution.  The household
    receives the net amount as disposable income, while the difference is
    remitted to government as income tax.
    """
    gross = _non_negative_state_1d("gross_distribution", gross_distribution)
    try:
        tax_rate = float(income_tax_rate)
    except (TypeError, ValueError) as error:
        raise ValueError("income_tax_rate must be a finite scalar between zero and one.") from error
    if not np.isfinite(tax_rate) or not 0.0 <= tax_rate <= 1.0:
        raise ValueError("income_tax_rate must be a finite scalar between zero and one.")
    income_tax_withheld = gross * tax_rate
    return gross - income_tax_withheld, income_tax_withheld


def compute_cash_feasible_firm_distribution_settlement(
    *,
    declared_distribution: np.ndarray,
    cash_available_after_debt_service: np.ndarray,
    operating_revolving_repayment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Settle declared firm payouts only from cash left after senior claims."""
    declared = _non_negative_state_1d("declared_distribution", declared_distribution)
    available = _finite_1d("cash_available_after_debt_service", cash_available_after_debt_service)
    revolving_repayment = _non_negative_state_1d(
        "operating_revolving_repayment",
        operating_revolving_repayment,
    )
    if available.shape != declared.shape or revolving_repayment.shape != declared.shape:
        raise ValueError("Firm distribution settlement inputs must contain one value per firm.")
    distributable_cash = np.maximum(available - revolving_repayment, 0.0)
    settled = np.minimum(declared, distributable_cash)
    return settled, declared - settled


def compute_diagnostic_dividend_fund(
    *,
    beginning_ifa: np.ndarray,
    firm_cash_profit_after_settlement: np.ndarray,
    firm_closing_deposits: np.ndarray,
    firm_unpaid_interest: np.ndarray,
    firm_closing_principal_arrears: np.ndarray,
    firm_residual_overdraft_exposure: np.ndarray,
    firm_default_flag: np.ndarray,
    bank_cash_distributable_profits: np.ndarray,
    ownership_quota: np.ndarray,
) -> DiagnosticDividendFundResult:
    """Calculate eligible capacity and allocate it by fixed ownership quota."""
    beginning_ifa = _finite_1d("beginning_ifa", beginning_ifa)
    firm_cash_profit_after_settlement = _finite_1d(
        "firm_cash_profit_after_settlement", firm_cash_profit_after_settlement
    )
    firm_closing_deposits = _finite_1d("firm_closing_deposits", firm_closing_deposits)
    firm_unpaid_interest = _non_negative_state_1d("firm_unpaid_interest", firm_unpaid_interest)
    firm_closing_principal_arrears = _non_negative_state_1d(
        "firm_closing_principal_arrears",
        firm_closing_principal_arrears,
    )
    firm_residual_overdraft_exposure = _non_negative_state_1d(
        "firm_residual_overdraft_exposure",
        firm_residual_overdraft_exposure,
    )
    firm_default_flag = _boolean_1d("firm_default_flag", firm_default_flag)
    bank_cash_distributable_profits = _finite_1d(
        "bank_cash_distributable_profits",
        bank_cash_distributable_profits,
    )
    firm_inputs = (
        firm_closing_deposits,
        firm_unpaid_interest,
        firm_closing_principal_arrears,
        firm_residual_overdraft_exposure,
        firm_default_flag,
    )
    if any(values.shape != firm_cash_profit_after_settlement.shape for values in firm_inputs):
        raise ValueError("All firm dividend-fund inputs must contain one value per firm.")

    allocation_weight = _non_negative_state_1d("ownership_quota", ownership_quota)
    if allocation_weight.shape != beginning_ifa.shape:
        raise ValueError("ownership_quota must contain one value per household.")
    quota_sum = float(allocation_weight.sum())
    if quota_sum > 0.0 and not np.isclose(quota_sum, 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("Positive ownership quotas must sum to one.")

    firm_distress_gate_passed = (
        (firm_unpaid_interest <= 1e-9)
        & (firm_closing_principal_arrears <= 1e-9)
        & (firm_residual_overdraft_exposure <= 1e-9)
        & ~firm_default_flag
    )
    firm_candidate = np.where(
        firm_distress_gate_passed,
        np.minimum(
            np.maximum(firm_cash_profit_after_settlement, 0.0),
            np.maximum(firm_closing_deposits, 0.0),
        ),
        0.0,
    )
    bank_candidate = np.maximum(bank_cash_distributable_profits, 0.0)
    has_owners = quota_sum > 0.0
    firm_inflow = float(firm_candidate.sum()) if has_owners else 0.0
    bank_inflow = float(bank_candidate.sum()) if has_owners else 0.0
    firm_distribution = allocation_weight * firm_inflow
    bank_distribution = allocation_weight * bank_inflow
    total_distribution = firm_distribution + bank_distribution
    fund_inflow = firm_inflow + bank_inflow

    return DiagnosticDividendFundResult(
        beginning_ifa=beginning_ifa.copy(),
        ownership_quota=allocation_weight.copy(),
        firm_distributable_profit_candidate=firm_candidate,
        firm_distress_gate_passed=firm_distress_gate_passed,
        bank_distributable_profit_candidate=bank_candidate,
        hypothetical_firm_distribution=firm_distribution,
        hypothetical_bank_distribution=bank_distribution,
        hypothetical_total_distribution=total_distribution,
        firm_identity_error=float(firm_distribution.sum() - firm_inflow),
        bank_identity_error=float(bank_distribution.sum() - bank_inflow),
        fund_identity_error=float(total_distribution.sum() - fund_inflow),
    )
