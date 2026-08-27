"""Reconcile a baseline and household-investment-off HDF5 pair.

The four named channels are reported, but they are not forced to close.  The
remaining difference is derived from the household cash-ledger terms that
actually changed between runs and any still-unexplained residual is shown.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def _period_values(h5: h5py.File, country: str, series: str, period: int) -> np.ndarray:
    """Return one finite HDF5 period, failing loudly on malformed input."""
    path = f"{country}/{series}"
    if path not in h5:
        raise KeyError(f"Missing HDF5 series: {path}")

    dataset = h5[path]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"HDF5 path is not a dataset: {path}")
    if dataset.ndim == 0:
        raise ValueError(f"HDF5 series is scalar, expected a time dimension: {path}")
    if period < 0 or period >= dataset.shape[0]:
        raise IndexError(f"Period {period} is unavailable for {path}")

    values = np.asarray(dataset[period], dtype=float)
    if values.size == 0:
        raise ValueError(f"HDF5 series has an empty household panel: {path}, period {period}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite values found in {path}, period {period}")
    return values


def _period_total(h5: h5py.File, country: str, series: str, period: int) -> float:
    return float(_period_values(h5, country, series, period).sum())


def _financial_saving_flow(h5: h5py.File, country: str, period: int) -> float:
    if period < 1:
        raise ValueError("period must be at least 1 to compute a financial-saving flow")
    closing_assets = _period_total(h5, country, "households/total_liquid_financial_assets", period)
    closing_assets += _period_total(h5, country, "households/total_illiquid_financial_assets", period)
    opening_assets = _period_total(h5, country, "households/total_liquid_financial_assets", period - 1)
    opening_assets += _period_total(h5, country, "households/total_illiquid_financial_assets", period - 1)
    valuation_gain = _period_total(
        h5,
        country,
        "households/total_illiquid_financial_asset_capital_gains",
        period,
    )
    return closing_assets - opening_assets - valuation_gain


def reconcile_investment_counterfactual(
    baseline_path: str | Path,
    investment_off_path: str | Path,
    *,
    country: str = "FRA",
    period: int = 1,
) -> dict[str, float]:
    """Derive the investment-off decomposition from the saved cash identities."""

    with h5py.File(baseline_path) as baseline, h5py.File(investment_off_path) as off:

        def base(series: str) -> float:
            return _period_total(baseline, country, series, period)

        def counterfactual(series: str) -> float:
            return _period_total(off, country, series, period)

        def off_minus_base(series: str) -> float:
            return counterfactual(series) - base(series)

        forgone_pre_tax_investment = base("households/total_investment_before_vat") - counterfactual(
            "households/total_investment_before_vat"
        )
        baseline_investment_taxes = base("households/total_investment") - base("households/total_investment_before_vat")
        off_investment_taxes = counterfactual("households/total_investment") - counterfactual(
            "households/total_investment_before_vat"
        )
        avoided_investment_taxes = baseline_investment_taxes - off_investment_taxes

        increased_net_financial_saving = _financial_saving_flow(off, country, period) - _financial_saving_flow(
            baseline, country, period
        )
        reduced_borrowing = sum(
            base(series) - counterfactual(series)
            for series in (
                "households/total_received_consumption_loans",
                "households/total_received_mortgages",
            )
        )
        reduced_liquidation = base("households/forced_liquidation_amount") - counterfactual(
            "households/forced_liquidation_amount"
        )

        left_hand_side = forgone_pre_tax_investment + avoided_investment_taxes
        named_channels = increased_net_financial_saving + reduced_borrowing + reduced_liquidation
        four_channel_residual = left_hand_side - named_channels

        # These are the remaining aggregate terms in the realised household
        # cash identity. Positive deltas are additional uses in the off arm;
        # positive income deltas are additional sources. The portfolio flows
        # between LFA and IFA cancel in aggregate financial saving; only their
        # committed adjustment cost is a cash use.
        baseline_noninvestment_expenditure = base("households/realised_household_expenditure") - base(
            "households/total_investment_before_vat"
        )
        off_noninvestment_expenditure = counterfactual("households/realised_household_expenditure") - counterfactual(
            "households/total_investment_before_vat"
        )
        tracked_other_cash_uses = off_noninvestment_expenditure - baseline_noninvestment_expenditure
        tracked_other_cash_uses += sum(
            off_minus_base(series)
            for series in (
                "households/interest_paid",
                "households/debt_installments",
                "households/price_paid_for_property",
                "households/portfolio_settlement_committed_adjustment_cost",
            )
        )
        tracked_other_cash_sources = off_minus_base("households/income_for_residual_saving")
        tracked_other_net_use = tracked_other_cash_uses - tracked_other_cash_sources
        # Liquidation exchanges IFA for LFA and is not a net source when saving
        # is measured as acquisition of total financial assets. Keep the user's
        # intended channel visible, but remove it when testing the actual cash
        # identity rather than forcing the four terms to close.
        liquidation_internal_reclassification = reduced_liquidation
        unexplained_cash_identity_residual = (
            four_channel_residual - tracked_other_net_use + liquidation_internal_reclassification
        )

        baseline_stage5_residual = _period_values(baseline, country, "households/stage5_cash_ledger_residual", period)
        off_stage5_residual = _period_values(off, country, "households/stage5_cash_ledger_residual", period)

    return {
        "forgone_pre_tax_investment": forgone_pre_tax_investment,
        "avoided_investment_taxes": avoided_investment_taxes,
        "increased_net_financial_saving": increased_net_financial_saving,
        "reduced_borrowing": reduced_borrowing,
        "reduced_asset_liquidation": reduced_liquidation,
        "four_channel_residual": four_channel_residual,
        "tracked_other_cash_uses": tracked_other_cash_uses,
        "tracked_other_cash_sources": tracked_other_cash_sources,
        "tracked_other_net_use": tracked_other_net_use,
        "liquidation_internal_reclassification": liquidation_internal_reclassification,
        "unexplained_cash_identity_residual": unexplained_cash_identity_residual,
        "baseline_stage5_max_abs_residual": float(np.max(np.abs(baseline_stage5_residual))),
        "investment_off_stage5_max_abs_residual": float(np.max(np.abs(off_stage5_residual))),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_h5", type=Path)
    parser.add_argument("investment_off_h5", type=Path)
    parser.add_argument("--country", default="FRA")
    parser.add_argument("--period", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = reconcile_investment_counterfactual(
        args.baseline_h5,
        args.investment_off_h5,
        country=args.country,
        period=args.period,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
