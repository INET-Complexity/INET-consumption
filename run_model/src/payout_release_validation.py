"""Release-gate validation for household dividend-fund simulations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class PayoutReleaseResult:
    seed: int
    period_count: int
    payout_ratio: float
    max_distribution_target_ratio: float
    max_realised_identity_error: float
    max_expected_identity_error: float
    max_absolute_capital_gains: float


def _maximum_absolute(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def validate_payout_release_file(
    path: str | Path,
    *,
    seed: int,
    country: str = "FRA",
    expected_periods: int = 50,
    expected_payout_ratio: float = 0.10,
    absolute_tolerance: float = 2e-5,
    relative_tolerance: float = 1e-10,
) -> PayoutReleaseResult:
    """Validate one saved simulation against the Increment 2 release contract."""
    path = Path(path)
    with h5py.File(path) as h5_file:
        households = h5_file[f"{country}/households"]
        distribution = households["total_income_financial_assets_distribution"][1:, 0]
        residual = households["total_income_financial_assets_residual_portfolio_return"][1:, 0]
        total_income = households["total_income_financial_assets"][1:, 0]
        target = households["total_income_financial_assets_calibration_target"][1:, 0]
        expected_distribution = households["total_expected_income_financial_assets_distribution"][1:, 0]
        expected_residual = households["total_expected_income_financial_assets_residual"][1:, 0]
        expected_total = households["expected_income_financial_assets"][1:].sum(axis=1)
        capital_gains = households["total_wealth_other_financial_assets_capital_gains"][1:, 0]
        payout_ratio = households["dividend_fund_payout_ratio"][1:, 0]

    arrays = (
        distribution,
        residual,
        total_income,
        target,
        expected_distribution,
        expected_residual,
        expected_total,
        capital_gains,
        payout_ratio,
    )
    period_count = len(distribution)
    if period_count != expected_periods:
        raise AssertionError(f"Seed {seed} has {period_count} periods; expected {expected_periods}.")
    if any(len(values) != period_count for values in arrays):
        raise AssertionError(f"Seed {seed} has inconsistent payout diagnostic lengths.")
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise AssertionError(f"Seed {seed} contains non-finite payout release diagnostics.")
    if not np.allclose(
        payout_ratio,
        expected_payout_ratio,
        atol=absolute_tolerance,
        rtol=relative_tolerance,
    ):
        raise AssertionError(f"Seed {seed} payout ratio does not match {expected_payout_ratio:.10g}.")

    ratio = np.divide(
        distribution,
        target,
        out=np.where(distribution > absolute_tolerance, np.inf, 0.0),
        where=target > absolute_tolerance,
    )
    max_ratio = float(np.max(ratio)) if ratio.size else 0.0
    if max_ratio > 1.0 + relative_tolerance:
        raise AssertionError(
            f"Seed {seed} distribution exceeds the financial-income target: max ratio={max_ratio:.6f}."
        )

    realised_error = total_income - distribution - residual
    expected_error = expected_total - expected_distribution - expected_residual
    if not np.allclose(realised_error, 0.0, atol=absolute_tolerance, rtol=relative_tolerance):
        raise AssertionError(f"Seed {seed} fails the realised financial-income component identity.")
    if not np.allclose(expected_error, 0.0, atol=absolute_tolerance, rtol=relative_tolerance):
        raise AssertionError(f"Seed {seed} fails the expected financial-income component identity.")
    if _maximum_absolute(capital_gains) > absolute_tolerance:
        raise AssertionError(f"Seed {seed} records non-zero capital gains in the income-only increment.")
    if np.any(payout_ratio > 0.0) and not np.any(expected_distribution > absolute_tolerance):
        raise AssertionError(f"Seed {seed} does not hand the enabled lagged payout into expected income.")

    return PayoutReleaseResult(
        seed=seed,
        period_count=period_count,
        payout_ratio=float(payout_ratio[-1]),
        max_distribution_target_ratio=max_ratio,
        max_realised_identity_error=_maximum_absolute(realised_error),
        max_expected_identity_error=_maximum_absolute(expected_error),
        max_absolute_capital_gains=_maximum_absolute(capital_gains),
    )


def validate_payout_release_envelope(
    h5_root: str | Path,
    *,
    seeds: tuple[int, ...] = (12, 13, 14, 15, 16),
    country: str = "FRA",
    expected_periods: int = 50,
    expected_payout_ratio: float = 0.10,
) -> list[PayoutReleaseResult]:
    """Validate the declared multi-seed release envelope."""
    root = Path(h5_root)
    results = []
    for seed in seeds:
        path = root / f"seed-{seed}" / "multi_country_simulation.h5"
        if not path.exists():
            raise FileNotFoundError(f"Missing release output for seed {seed}: {path}")
        results.append(
            validate_payout_release_file(
                path,
                seed=seed,
                country=country,
                expected_periods=expected_periods,
                expected_payout_ratio=expected_payout_ratio,
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5_root", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[12, 13, 14, 15, 16])
    parser.add_argument("--country", default="FRA")
    parser.add_argument("--expected-periods", type=int, default=50)
    parser.add_argument("--expected-payout-ratio", type=float, default=0.10)
    args = parser.parse_args()
    for result in validate_payout_release_envelope(
        args.h5_root,
        seeds=tuple(args.seeds),
        country=args.country,
        expected_periods=args.expected_periods,
        expected_payout_ratio=args.expected_payout_ratio,
    ):
        print(result)


if __name__ == "__main__":
    main()
