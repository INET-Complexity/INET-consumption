"""Release-gate validation for ownership-based household payout simulations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class PayoutReleaseResult:
    """Reconciliation summary for one saved payout simulation."""

    seed: int
    period_count: int
    max_firm_receipt_identity_error: float
    max_bank_receipt_identity_error: float
    max_household_income_identity_error: float
    max_expected_dividend_timing_error: float
    max_absolute_capital_gain: float


def _maximum_absolute(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def _read_dataset(group: h5py.Group, name: str) -> np.ndarray:
    """Read a release dataset, reporting its full HDF5 path if it is missing."""
    if name not in group:
        raise AssertionError(f"Missing payout release diagnostic: {group.name}/{name}.")
    values = np.asarray(group[name], dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise AssertionError(f"Payout release diagnostic must be two-dimensional: {group.name}/{name}.")
    return values


def _require_same_period_count(seed: int, arrays: dict[str, np.ndarray]) -> int:
    """Ensure all payout arrays contain the same initial row and periods."""
    lengths = {name: values.shape[0] for name, values in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise AssertionError(f"Seed {seed} has inconsistent payout diagnostic lengths: {lengths}.")
    period_count = next(iter(lengths.values())) - 1
    if period_count < 0:
        raise AssertionError(f"Seed {seed} has no initial payout diagnostic row.")
    return period_count


def _assert_finite(seed: int, arrays: dict[str, np.ndarray]) -> None:
    for name, values in arrays.items():
        if not np.all(np.isfinite(values)):
            raise AssertionError(f"Seed {seed} contains non-finite {name} diagnostics.")


def _assert_non_negative(
    seed: int,
    arrays: dict[str, np.ndarray],
    *,
    absolute_tolerance: float,
) -> None:
    for name, values in arrays.items():
        if np.any(values < -absolute_tolerance):
            raise AssertionError(f"Seed {seed} has a negative {name} component.")


def validate_payout_release_file(
    path: str | Path,
    *,
    seed: int,
    country: str = "FRA",
    expected_periods: int = 50,
    absolute_tolerance: float = 2e-5,
    relative_tolerance: float = 1e-10,
) -> PayoutReleaseResult:
    """Validate payer-funded settlement, withholding, and income timing identities."""
    path = Path(path)
    with h5py.File(path) as h5_file:
        households = h5_file[f"{country}/households"]
        firms = h5_file[f"{country}/firms"]
        banks = h5_file[f"{country}/banks"]
        arrays = {
            "household firm receipt": _read_dataset(households, "dividend_fund_settled_firm_distribution"),
            "household bank receipt": _read_dataset(households, "dividend_fund_settled_bank_distribution"),
            "household dividend income": _read_dataset(households, "income_dividend_distributions"),
            "household dividend income tax withheld": _read_dataset(
                households,
                "dividend_fund_income_tax_withheld",
            ),
            "expected household dividend income": _read_dataset(
                households,
                "expected_income_dividend_distributions",
            ),
            "ownership quota": _read_dataset(households, "dividend_fund_ownership_quota"),
            "firm payer settlement debit": _read_dataset(firms, "dividend_fund_settlement_debit"),
            "bank payer settlement debit": _read_dataset(banks, "dividend_fund_settlement_debit"),
            "firm settlement identity": _read_dataset(
                households,
                "dividend_fund_firm_settlement_identity_error",
            ),
            "bank settlement identity": _read_dataset(
                households,
                "dividend_fund_bank_settlement_identity_error",
            ),
            "combined settlement identity": _read_dataset(
                households,
                "dividend_fund_settlement_identity_error",
            ),
            "firm settlement shortfall": _read_dataset(
                firms,
                "dividend_fund_settlement_shortfall",
            ),
            "IFA capital gain": _read_dataset(
                households,
                "illiquid_financial_asset_capital_gains",
            ),
        }

    period_count = _require_same_period_count(seed, arrays)
    if period_count != expected_periods:
        raise AssertionError(f"Seed {seed} has {period_count} periods; expected {expected_periods}.")
    _assert_finite(seed, arrays)
    _assert_non_negative(
        seed,
        {name: values for name, values in arrays.items() if name != "IFA capital gain"},
        absolute_tolerance=absolute_tolerance,
    )

    firm_receipts = arrays["household firm receipt"][1:].sum(axis=1)
    bank_receipts = arrays["household bank receipt"][1:].sum(axis=1)
    dividend_income = arrays["household dividend income"][1:].sum(axis=1)
    dividend_income_tax_withheld = arrays["household dividend income tax withheld"][1:].sum(axis=1)
    expected_dividend_income = arrays["expected household dividend income"][1:].sum(axis=1)
    previous_dividend_income = arrays["household dividend income"][:-1].sum(axis=1)
    firm_debits = arrays["firm payer settlement debit"][1:].sum(axis=1)
    bank_debits = arrays["bank payer settlement debit"][1:].sum(axis=1)
    quota_sums = arrays["ownership quota"][1:].sum(axis=1)

    firm_receipt_identity_error = firm_receipts - firm_debits
    bank_receipt_identity_error = bank_receipts - bank_debits
    household_income_identity_error = dividend_income + dividend_income_tax_withheld - firm_receipts - bank_receipts
    expected_dividend_timing_error = expected_dividend_income - previous_dividend_income
    payer_activity = np.abs(firm_debits) + np.abs(bank_debits)
    receipt_activity = np.abs(firm_receipts) + np.abs(bank_receipts)
    invalid_quota = ~(
        np.isclose(quota_sums, 0.0, atol=absolute_tolerance, rtol=relative_tolerance)
        | np.isclose(quota_sums, 1.0, atol=absolute_tolerance, rtol=relative_tolerance)
    )
    if np.any(invalid_quota):
        raise AssertionError(f"Seed {seed} has invalid fixed ownership quota sums.")
    no_owner_payout = np.isclose(quota_sums, 0.0, atol=absolute_tolerance, rtol=relative_tolerance) & (
        (payer_activity > absolute_tolerance) | (receipt_activity > absolute_tolerance)
    )
    if np.any(no_owner_payout):
        raise AssertionError(f"Seed {seed} pays a dividend despite having no direct-share owners.")
    checks = {
        "firm payer-to-receipt identity": firm_receipt_identity_error,
        "bank payer-to-receipt identity": bank_receipt_identity_error,
        "household dividend income identity": household_income_identity_error,
        "expected dividend timing": expected_dividend_timing_error,
        "firm settlement identity diagnostic": arrays["firm settlement identity"][1:],
        "bank settlement identity diagnostic": arrays["bank settlement identity"][1:],
        "combined settlement identity diagnostic": arrays["combined settlement identity"][1:],
    }
    for name, error in checks.items():
        if not np.allclose(error, 0.0, atol=absolute_tolerance, rtol=relative_tolerance):
            raise AssertionError(f"Seed {seed} fails the {name}.")

    return PayoutReleaseResult(
        seed=seed,
        period_count=period_count,
        max_firm_receipt_identity_error=_maximum_absolute(firm_receipt_identity_error),
        max_bank_receipt_identity_error=_maximum_absolute(bank_receipt_identity_error),
        max_household_income_identity_error=_maximum_absolute(household_income_identity_error),
        max_expected_dividend_timing_error=_maximum_absolute(expected_dividend_timing_error),
        max_absolute_capital_gain=_maximum_absolute(arrays["IFA capital gain"][1:]),
    )


def validate_payout_release_envelope(
    h5_root: str | Path,
    *,
    seeds: tuple[int, ...] = (12, 13, 14, 15, 16),
    country: str = "FRA",
    expected_periods: int = 50,
) -> list[PayoutReleaseResult]:
    """Validate every declared seed in a payout release envelope."""
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
            )
        )
    return results


def main() -> None:
    """Run the payout release validator from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5_root", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[12, 13, 14, 15, 16])
    parser.add_argument("--country", default="FRA")
    parser.add_argument("--expected-periods", type=int, default=50)
    args = parser.parse_args()
    for result in validate_payout_release_envelope(
        args.h5_root,
        seeds=tuple(args.seeds),
        country=args.country,
        expected_periods=args.expected_periods,
    ):
        print(result)


if __name__ == "__main__":
    main()
