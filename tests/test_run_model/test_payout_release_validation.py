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
    quota = np.tile(np.array([[0.25, 0.75]]), (periods + 1, 1))
    firm_declarations = np.array(
        [
            [0.0, 0.0],
            [1.0, 3.0],
            [0.0, 0.0],
            [2.0, 6.0],
        ]
    )[: periods + 1]
    bank_declarations = np.array(
        [
            [0.0, 0.0],
            [0.5, 1.5],
            [0.0, 0.0],
            [0.25, 0.75],
        ]
    )[: periods + 1]
    firm_retained_capacity = np.array(
        [
            [0.0, 0.0],
            [0.5, 1.5],
            [0.0, 0.0],
            [2.0, 4.0],
        ]
    )[: periods + 1]
    bank_retained_capacity = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.5],
            [0.0, 0.0],
            [0.25, 0.25],
        ]
    )[: periods + 1]
    firm_debits = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 3.0],
            [0.0, 0.0],
        ]
    )[: periods + 1]
    bank_debits = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.5, 1.5],
            [0.0, 0.0],
        ]
    )[: periods + 1]
    firm_shortfall = np.zeros_like(firm_debits)
    household_firm_receipts = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 3.0],
            [0.0, 0.0],
        ]
    )[: periods + 1]
    household_bank_receipts = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.5, 1.5],
            [0.0, 0.0],
        ]
    )[: periods + 1]
    gross_dividend_income = household_firm_receipts + household_bank_receipts
    dividend_income_tax_withheld = 0.25 * gross_dividend_income
    dividend_income = gross_dividend_income - dividend_income_tax_withheld
    expected_dividend_income = np.vstack((np.zeros(2), dividend_income[:-1]))
    with h5py.File(path, "w") as h5_file:
        households = h5_file.create_group("FRA/households")
        firms = h5_file.create_group("FRA/firms")
        banks = h5_file.create_group("FRA/banks")
        households["dividend_fund_settled_firm_distribution"] = household_firm_receipts
        households["dividend_fund_settled_bank_distribution"] = household_bank_receipts
        households["income_dividend_distributions"] = dividend_income
        households["dividend_fund_income_tax_withheld"] = dividend_income_tax_withheld
        households["expected_income_dividend_distributions"] = expected_dividend_income
        households["dividend_fund_ownership_quota"] = quota
        households["dividend_fund_declared_firm_distribution"] = quota * firm_declarations.sum(axis=1, keepdims=True)
        households["dividend_fund_declared_bank_distribution"] = quota * bank_declarations.sum(axis=1, keepdims=True)
        households["illiquid_financial_asset_capital_gains"] = np.array(
            [[0.0, 0.0], [8.0, -4.0], [0.0, 0.0], [6.0, 2.0]]
        )[: periods + 1]
        for name in (
            "dividend_fund_firm_settlement_identity_error",
            "dividend_fund_bank_settlement_identity_error",
            "dividend_fund_settlement_identity_error",
        ):
            households[name] = np.zeros((periods + 1, 1))
        firms["dividend_fund_cash_distributable_profit_candidate"] = firm_declarations + firm_retained_capacity
        firms["dividend_fund_declared_distribution"] = firm_declarations
        firms["dividend_fund_retained_capacity"] = firm_retained_capacity
        firms["dividend_fund_settlement_debit"] = firm_debits
        firms["dividend_fund_settlement_shortfall"] = firm_shortfall
        banks["dividend_fund_cash_distributable_profit_candidate"] = bank_declarations + bank_retained_capacity
        banks["dividend_fund_declared_distribution"] = bank_declarations
        banks["dividend_fund_retained_capacity"] = bank_retained_capacity
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
    assert all(result.max_firm_declaration_settlement_error == 0.0 for result in results)
    assert all(result.max_bank_declaration_settlement_error == 0.0 for result in results)
    assert all(result.max_absolute_capital_gain == pytest.approx(8.0) for result in results)


def test__release_validator_allows_final_unsettled_declarations(tmp_path: Path) -> None:
    path = tmp_path / "seed-13" / "multi_country_simulation.h5"
    _write_release_file(path)

    with h5py.File(path) as h5_file:
        assert h5_file["FRA/firms/dividend_fund_declared_distribution"][-1].sum() > 0.0
        assert h5_file["FRA/banks/dividend_fund_declared_distribution"][-1].sum() > 0.0

    results = validate_payout_release_envelope(tmp_path, seeds=(13,), expected_periods=3)

    assert results[0].max_firm_declaration_settlement_error == 0.0
    assert results[0].max_bank_declaration_settlement_error == 0.0


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda h5_file: h5_file["FRA/households/income_dividend_distributions"].__setitem__(
                (2, 0),
                9.0,
            ),
            "household dividend income identity",
        ),
        (
            lambda h5_file: h5_file["FRA/households/dividend_fund_income_tax_withheld"].__setitem__(
                (2, 0),
                0.0,
            ),
            "household dividend income identity",
        ),
        (
            lambda h5_file: h5_file["FRA/households/dividend_fund_settled_firm_distribution"].__setitem__(
                (2, 0),
                2.0,
            ),
            "firm payer-to-receipt identity",
        ),
        (
            lambda h5_file: h5_file["FRA/households/expected_income_dividend_distributions"].__setitem__(
                (3, 0),
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
        (
            lambda h5_file: h5_file["FRA/firms/dividend_fund_settlement_shortfall"].__setitem__(
                (2, 0),
                1.0,
            ),
            "firm declaration-to-settlement identity",
        ),
        (
            lambda h5_file: h5_file["FRA/banks/dividend_fund_settlement_debit"].__setitem__(
                (2, 0),
                1.0,
            ),
            "bank declaration-to-settlement identity",
        ),
        (
            lambda h5_file: h5_file["FRA/households/dividend_fund_declared_firm_distribution"].__setitem__(
                (1, 0),
                2.0,
            ),
            "firm declaration allocation",
        ),
        (
            lambda h5_file: h5_file["FRA/firms/dividend_fund_retained_capacity"].__setitem__(
                (1, 0),
                1.0,
            ),
            "firm declaration capacity identity",
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
