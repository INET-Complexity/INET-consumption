import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

RUN_MODEL_PATH = Path(__file__).resolve().parents[2] / "run_model"
if str(RUN_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL_PATH))

from src.payout_release_validation import validate_payout_release_envelope  # noqa: E402


def _write_release_file(path, *, distribution_ratio: float, capital_gains: float = 0.0):
    path.parent.mkdir(parents=True)
    target = np.array([100.0, 100.0])
    distribution = target * distribution_ratio
    residual = target - distribution
    with h5py.File(path, "w") as h5_file:
        households = h5_file.create_group("FRA/households")
        households["total_income_financial_assets_distribution"] = distribution[:, None]
        households["total_income_financial_assets_residual_portfolio_return"] = residual[:, None]
        households["total_income_financial_assets"] = target[:, None]
        households["total_income_financial_assets_calibration_target"] = target[:, None]
        households["total_expected_income_financial_assets_distribution"] = distribution[:, None]
        households["total_expected_income_financial_assets_residual"] = residual[:, None]
        households["expected_income_financial_assets"] = target[:, None]
        households["total_wealth_other_financial_assets_capital_gains"] = np.array([[0.0], [capital_gains]])
        households["dividend_fund_payout_ratio"] = np.array([[0.04], [0.04]])


def test__multi_seed_payout_release_envelope_accepts_income_only_handoff(tmp_path):
    for seed in (12, 13):
        _write_release_file(
            tmp_path / f"seed-{seed}" / "multi_country_simulation.h5",
            distribution_ratio=0.4,
        )

    results = validate_payout_release_envelope(tmp_path, seeds=(12, 13))

    assert [result.seed for result in results] == [12, 13]
    assert all(result.max_distribution_target_ratio == pytest.approx(0.4) for result in results)
    assert all(result.max_absolute_capital_gains == 0.0 for result in results)


def test__multi_seed_payout_release_envelope_rejects_target_breach(tmp_path):
    _write_release_file(
        tmp_path / "seed-13" / "multi_country_simulation.h5",
        distribution_ratio=1.01,
    )

    with pytest.raises(AssertionError, match="distribution exceeds"):
        validate_payout_release_envelope(tmp_path, seeds=(13,))


def test__multi_seed_payout_release_envelope_rejects_capital_gains(tmp_path):
    _write_release_file(
        tmp_path / "seed-13" / "multi_country_simulation.h5",
        distribution_ratio=0.4,
        capital_gains=1.0,
    )

    with pytest.raises(AssertionError, match="non-zero capital gains"):
        validate_payout_release_envelope(tmp_path, seeds=(13,))
