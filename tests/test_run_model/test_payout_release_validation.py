import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

RUN_MODEL_PATH = Path(__file__).resolve().parents[2] / "run_model"
if str(RUN_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL_PATH))

from src.payout_release_validation import validate_payout_release_envelope  # noqa: E402


def _write_release_file(
    path,
    *,
    distribution_ratio: float,
    capital_gains: float = 0.0,
    periods: int = 50,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    target = np.full(periods + 1, 100.0)
    distribution = target * distribution_ratio
    residual = target.copy()
    total_income = distribution + residual
    distribution_excess = np.maximum(distribution - target, 0.0)
    with h5py.File(path, "w") as h5_file:
        households = h5_file.create_group("FRA/households")
        households["total_income_financial_assets_distribution"] = distribution[:, None]
        households["total_income_financial_assets_residual_portfolio_return"] = residual[:, None]
        households["total_income_financial_assets"] = total_income[:, None]
        households["total_income_financial_assets_calibration_target"] = target[:, None]
        households["income_financial_assets_distribution_excess_over_target"] = distribution_excess[:, None]
        households["total_expected_income_financial_assets_distribution"] = distribution[:, None]
        households["total_expected_income_financial_assets_residual"] = residual[:, None]
        households["expected_income_financial_assets"] = total_income[:, None]
        capital_gain_series = np.zeros(periods + 1)
        capital_gain_series[-1] = capital_gains
        households["total_wealth_other_financial_assets_capital_gains"] = capital_gain_series[:, None]


def test__multi_seed_payout_release_envelope_accepts_income_only_handoff(tmp_path):
    for seed in (12, 13):
        _write_release_file(
            tmp_path / f"seed-{seed}" / "multi_country_simulation.h5",
            distribution_ratio=0.4,
        )

    results = validate_payout_release_envelope(tmp_path, seeds=(12, 13))

    assert [result.seed for result in results] == [12, 13]
    assert all(result.period_count == 50 for result in results)
    assert all(result.max_distribution_target_ratio == pytest.approx(0.4) for result in results)
    assert all(result.max_distribution_excess_over_target == 0.0 for result in results)
    assert all(result.max_absolute_capital_gains == 0.0 for result in results)


def test__multi_seed_payout_release_envelope_reports_target_breach(tmp_path):
    _write_release_file(
        tmp_path / "seed-13" / "multi_country_simulation.h5",
        distribution_ratio=1.01,
    )

    result = validate_payout_release_envelope(tmp_path, seeds=(13,))[0]

    assert result.max_distribution_target_ratio == pytest.approx(1.01)
    assert result.max_distribution_excess_over_target == pytest.approx(1.0)


def test__multi_seed_payout_release_envelope_rejects_wrong_excess_diagnostic(tmp_path):
    path = tmp_path / "seed-13" / "multi_country_simulation.h5"
    _write_release_file(path, distribution_ratio=1.01)
    with h5py.File(path, "r+") as h5_file:
        h5_file["FRA/households/income_financial_assets_distribution_excess_over_target"][-1, 0] = 0.0

    with pytest.raises(AssertionError, match="distribution-excess-over-target identity"):
        validate_payout_release_envelope(tmp_path, seeds=(13,))


def test__multi_seed_payout_release_envelope_rejects_capital_gains(tmp_path):
    _write_release_file(
        tmp_path / "seed-13" / "multi_country_simulation.h5",
        distribution_ratio=0.4,
        capital_gains=1.0,
    )

    with pytest.raises(AssertionError, match="non-zero capital gains"):
        validate_payout_release_envelope(tmp_path, seeds=(13,))


@pytest.mark.parametrize(
    ("dataset", "identity", "message"),
    (
        (
            "total_income_financial_assets_distribution",
            "realised",
            "negative distribution component",
        ),
        (
            "total_income_financial_assets_residual_portfolio_return",
            "realised",
            "negative residual portfolio return component",
        ),
        (
            "total_income_financial_assets_calibration_target",
            None,
            "negative calibration target component",
        ),
        (
            "income_financial_assets_distribution_excess_over_target",
            None,
            "negative distribution excess over target component",
        ),
        (
            "total_expected_income_financial_assets_distribution",
            "expected",
            "negative expected distribution component",
        ),
        (
            "total_expected_income_financial_assets_residual",
            "expected",
            "negative expected residual component",
        ),
    ),
)
def test__multi_seed_payout_release_envelope_rejects_negative_components(tmp_path, dataset, identity, message):
    path = tmp_path / "seed-13" / "multi_country_simulation.h5"
    _write_release_file(path, distribution_ratio=0.4)
    with h5py.File(path, "r+") as h5_file:
        households = h5_file["FRA/households"]
        households[dataset][-1, 0] = -1.0
        if identity == "realised":
            households["total_income_financial_assets"][-1, 0] = (
                households["total_income_financial_assets_distribution"][-1, 0]
                + households["total_income_financial_assets_residual_portfolio_return"][-1, 0]
            )
        elif identity == "expected":
            households["expected_income_financial_assets"][-1, 0] = (
                households["total_expected_income_financial_assets_distribution"][-1, 0]
                + households["total_expected_income_financial_assets_residual"][-1, 0]
            )

    with pytest.raises(AssertionError, match=message):
        validate_payout_release_envelope(tmp_path, seeds=(13,))


def test__multi_seed_payout_release_envelope_rejects_wrong_horizon(tmp_path):
    path = tmp_path / "seed-13" / "multi_country_simulation.h5"
    _write_release_file(path, distribution_ratio=0.4, periods=49)
    with pytest.raises(AssertionError, match="49 periods"):
        validate_payout_release_envelope(tmp_path, seeds=(13,))
