"""Read-only household financial-asset dividend-fund diagnostics."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiagnosticDividendFundResult:
    """Read-only result of the hypothetical household dividend-fund allocation."""

    beginning_ifa: np.ndarray
    ifa_weight: np.ndarray
    firm_distributable_profit_candidate: np.ndarray
    firm_distress_gate_passed: np.ndarray
    bank_distributable_profit_candidate: np.ndarray
    hypothetical_firm_distribution: np.ndarray
    hypothetical_bank_distribution: np.ndarray
    hypothetical_total_distribution: np.ndarray
    distribution_by_ifa_quintile: np.ndarray
    total_positive_ifa: float
    positive_ifa_household_count: int
    positive_ifa_household_share: float
    total_firm_distributable_profit_candidate: float
    total_bank_distributable_profit_candidate: float
    firm_distress_gate_passed_count: int
    firm_distress_gate_passed_share: float
    hypothetical_firm_fund_inflow: float
    hypothetical_bank_fund_inflow: float
    hypothetical_fund_inflow: float
    hypothetical_fund_outflow: float
    aggregate_distribution_yield: float
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


def _ifa_quintile_totals(positive_ifa: np.ndarray, distribution: np.ndarray) -> np.ndarray:
    """Sum receipts by stable midpoint-ranked IFA quintile for all households."""
    totals = np.zeros(5)
    if positive_ifa.size == 0:
        return totals
    ordered = np.argsort(positive_ifa, kind="stable")
    quintiles = np.minimum(
        ((2 * np.arange(positive_ifa.size) + 1) * 5) // (2 * positive_ifa.size),
        4,
    )
    np.add.at(totals, quintiles, distribution[ordered])
    return totals


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
) -> DiagnosticDividendFundResult:
    """Compute a hypothetical, behaviourally inert dividend-fund allocation.

    Firm candidates use current-period cash receipts net of production,
    investment, tax, interest, principal, and facility-repayment cash outflows.
    Opening cash and new credit are excluded by the caller. Existing settlement
    flags provide a distress gate. Increment 1 applies no calibrated payout
    ratio or bank-capital policy; those belong to the later fixed-policy step.
    """
    beginning_ifa = _finite_1d("beginning_ifa", beginning_ifa)
    firm_cash_profit_after_settlement = _finite_1d(
        "firm_cash_profit_after_settlement", firm_cash_profit_after_settlement
    )
    firm_closing_deposits = _finite_1d("firm_closing_deposits", firm_closing_deposits)
    firm_unpaid_interest = _non_negative_state_1d("firm_unpaid_interest", firm_unpaid_interest)
    firm_closing_principal_arrears = _non_negative_state_1d(
        "firm_closing_principal_arrears", firm_closing_principal_arrears
    )
    firm_residual_overdraft_exposure = _non_negative_state_1d(
        "firm_residual_overdraft_exposure", firm_residual_overdraft_exposure
    )
    firm_default_flag = _boolean_1d("firm_default_flag", firm_default_flag)
    bank_cash_distributable_profits = _finite_1d("bank_cash_distributable_profits", bank_cash_distributable_profits)

    firm_shape = firm_cash_profit_after_settlement.shape
    firm_inputs = (
        firm_closing_deposits,
        firm_unpaid_interest,
        firm_closing_principal_arrears,
        firm_residual_overdraft_exposure,
        firm_default_flag,
    )
    if any(values.shape != firm_shape for values in firm_inputs):
        raise ValueError("All firm dividend-fund inputs must contain one value per firm.")

    positive_ifa = np.maximum(beginning_ifa, 0.0)
    total_positive_ifa = float(positive_ifa.sum())
    ifa_weight = np.divide(
        positive_ifa,
        total_positive_ifa,
        out=np.zeros_like(positive_ifa),
        where=total_positive_ifa > 0.0,
    )

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

    total_firm_candidate = float(firm_candidate.sum())
    total_bank_candidate = float(bank_candidate.sum())
    holdings_available = total_positive_ifa > 0.0
    firm_inflow = total_firm_candidate if holdings_available else 0.0
    bank_inflow = total_bank_candidate if holdings_available else 0.0
    firm_distribution = ifa_weight * firm_inflow
    bank_distribution = ifa_weight * bank_inflow
    total_distribution = firm_distribution + bank_distribution
    fund_inflow = firm_inflow + bank_inflow
    fund_outflow = float(total_distribution.sum())
    positive_holding_count = int(np.count_nonzero(positive_ifa > 0.0))

    return DiagnosticDividendFundResult(
        beginning_ifa=beginning_ifa.copy(),
        ifa_weight=ifa_weight,
        firm_distributable_profit_candidate=firm_candidate,
        firm_distress_gate_passed=firm_distress_gate_passed,
        bank_distributable_profit_candidate=bank_candidate,
        hypothetical_firm_distribution=firm_distribution,
        hypothetical_bank_distribution=bank_distribution,
        hypothetical_total_distribution=total_distribution,
        distribution_by_ifa_quintile=_ifa_quintile_totals(positive_ifa, total_distribution),
        total_positive_ifa=total_positive_ifa,
        positive_ifa_household_count=positive_holding_count,
        positive_ifa_household_share=(positive_holding_count / beginning_ifa.size if beginning_ifa.size > 0 else 0.0),
        total_firm_distributable_profit_candidate=total_firm_candidate,
        total_bank_distributable_profit_candidate=total_bank_candidate,
        firm_distress_gate_passed_count=int(np.count_nonzero(firm_distress_gate_passed)),
        firm_distress_gate_passed_share=(
            float(np.mean(firm_distress_gate_passed)) if firm_distress_gate_passed.size > 0 else 0.0
        ),
        hypothetical_firm_fund_inflow=firm_inflow,
        hypothetical_bank_fund_inflow=bank_inflow,
        hypothetical_fund_inflow=fund_inflow,
        hypothetical_fund_outflow=fund_outflow,
        aggregate_distribution_yield=(fund_outflow / total_positive_ifa if holdings_available else 0.0),
        firm_identity_error=float(firm_distribution.sum() - firm_inflow),
        bank_identity_error=float(bank_distribution.sum() - bank_inflow),
        fund_identity_error=float(fund_outflow - fund_inflow),
    )
