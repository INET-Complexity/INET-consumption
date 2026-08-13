import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

RUN_MODEL_PATH = Path(__file__).resolve().parents[2] / "run_model"
if str(RUN_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL_PATH))

from src.payout_release_validation import validate_payout_release_envelope  # noqa: E402


def _write_release_file(path: Path, *, periods: int = 3) -> None:
    """Write a minimal HDF5 payout ledger with declaration/settlement lag."""
    path.parent.mkdir(parents=True, exist_ok=True)
    household_firm_receipts = np.array(
        [
            [0.0, 0.0],
            [1.0, 3.0],
            [0.0, 0.0],
            [2.0, 6.0],
        ]
    )[: periods + 1]
    household_bank_receipts = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ]
    )[: periods + 1]
    dividend_income = household_firm_receipts + household_bank_receipts
    expected_dividend_income = np.vstack((np.zeros(2), dividend_income[:-1]))
    firm_debits = np.array([[0.0], [4.0], [0.0], [8.0]])[: periods + 1]
    bank_debits = np.array([[0.0], [2.0], [0.0], [1.0]])[: periods + 1]
    quota = np.tile(np.array([[0.25, 0.75]]), (periods + 1, 1))
    with h5py.File(path, "w") as h5_file:
        households = h5_file.create_group("FRA/households")
        firms = h5_file.create_group("FRA/firms")
        banks = h5_file.create_group("FRA/banks")
        households["dividend_fund_settled_firm_distribution"] = household_firm_receipts
        households["dividend_fund_settled_bank_distribution"] = household_bank_receipts
        households["income_dividend_distributions"] = dividend_income
        households["expected_income_dividend_distributions"] = expected_dividend_income
        households["dividend_fund_ownership_quota"] = quota
        households["illiquid_financial_asset_capital_gains"] = np.array(
            [[0.0, 0.0], [8.0, -4.0], [0.0, 0.0], [6.0, 2.0]]
        )[: periods + 1]
        for name in (
            "dividend_fund_firm_settlement_identity_error",
            "dividend_fund_bank_settlement_identity_error",
            "dividend_fund_settlement_identity_error",
        ):
            households[name] = np.zeros((periods + 1, 1))
        firms["dividend_fund_settlement_debit"] = firm_debits
        firms["dividend_fund_settlement_shortfall"] = np.zeros_like(firm_debits)
        banks["dividend_fund_settlement_debit"] = bank_debits


def test__release_validator_accepts_separate_payer_receipts_and_capital_gains(tmp_path: Path) -> None:
    for seed in (12, 13):
        _write_release_file(tmp_path / f"seed-{seed}" / "multi_country_simulation.h5")

    results = validate_payout_release_envelope(tmp_path, seeds=(12, 13), expected_periods=3)

    assert [result.seed for result in results] == [12, 13]
    assert all(result.max_firm_receipt_identity_error == 0.0 for result in results)
    assert all(result.max_bank_receipt_identity_error == 0.0 for result in results)
    assert all(result.max_household_income_identity_error == 0.0 for result in results)
    assert all(result.max_expected_dividend_timing_error == 0.0 for result in results)
    assert all(result.max_absolute_capital_gain == pytest.approx(8.0) for result in results)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda h5_file: h5_file["FRA/households/income_dividend_distributions"].__setitem__(
                (1, 0),
                9.0,
            ),
            "household dividend income identity",
        ),
        (
            lambda h5_file: h5_file["FRA/households/dividend_fund_settled_firm_distribution"].__setitem__(
                (1, 0),
                2.0,
            ),
            "firm payer-to-receipt identity",
        ),
        (
            lambda h5_file: h5_file["FRA/households/expected_income_dividend_distributions"].__setitem__(
                (2, 0),
                0.0,
            ),
            "expected dividend timing",
        ),
        (
            lambda h5_file: h5_file["FRA/households/dividend_fund_ownership_quota"].__setitem__(
                (1, 1),
                0.50,
            ),
            "invalid fixed ownership quota sums",
        ),
    ),
)
def test__release_validator_rejects_unfunded_duplicate_bad_quota_and_timing(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    path = tmp_path / "seed-13" / "multi_country_simulation.h5"
    _write_release_file(path)
    with h5py.File(path, "r+") as h5_file:
        mutate(h5_file)

    with pytest.raises(AssertionError, match=message):
        validate_payout_release_envelope(tmp_path, seeds=(13,), expected_periods=3)


def test__release_validator_rejects_payout_when_no_household_owns_direct_shares(tmp_path: Path) -> None:
    path = tmp_path / "seed-13" / "multi_country_simulation.h5"
    _write_release_file(path)
    with h5py.File(path, "r+") as h5_file:
        h5_file["FRA/households/dividend_fund_ownership_quota"][1] = np.zeros(2)

    with pytest.raises(AssertionError, match="no direct-share owners"):
        validate_payout_release_envelope(tmp_path, seeds=(13,), expected_periods=3)
