from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.agents.households.func.financial_asset_income import compose_financial_asset_income
from macromodel.country.country import Country
from macromodel.timeseries import TimeSeries


def _compose(**overrides):
    inputs = {
        "lagged_distribution_income": np.array([0.0, 20.0, 30.0]),
        "residual_profile": np.array([-10.0, 10.0, 30.0]),
        "calibration_target": np.array([10.0, 30.0, 60.0]),
    }
    inputs.update(overrides)
    return compose_financial_asset_income(**inputs)


def test__financial_asset_income_reduces_residual_without_double_counting():
    result = _compose()

    np.testing.assert_array_equal(result.distribution_income, [0.0, 20.0, 30.0])
    np.testing.assert_allclose(result.residual_portfolio_return, [0.0, 12.5, 37.5])
    np.testing.assert_allclose(result.total_income, [0.0, 32.5, 67.5])
    assert result.aggregate_distribution_income == 50.0
    assert result.aggregate_residual_portfolio_return == 50.0
    assert result.aggregate_calibration_target == 100.0
    assert result.target_gap == pytest.approx(0.0)
    assert result.calibration_error == pytest.approx(0.0)
    assert np.all(result.residual_portfolio_return >= 0.0)


def test__zero_distribution_leaves_full_target_as_non_negative_residual():
    result = _compose(lagged_distribution_income=np.zeros(3))

    assert result.aggregate_distribution_income == 0.0
    assert result.aggregate_residual_portfolio_return == pytest.approx(100.0)
    assert result.total_income.sum() == pytest.approx(100.0)


def test__distribution_at_target_leaves_zero_residual():
    result = _compose(
        lagged_distribution_income=np.array([10.0, 30.0, 60.0]),
        residual_profile=np.full(3, -1.0),
    )

    np.testing.assert_array_equal(result.residual_portfolio_return, np.zeros(3))
    np.testing.assert_array_equal(result.total_income, result.distribution_income)


def test__distribution_above_target_requires_recalibration():
    with pytest.raises(ValueError, match="recalibrate payout ratios"):
        _compose(lagged_distribution_income=np.array([10.0, 30.0, 61.0]))


def test__non_positive_realised_profile_falls_back_to_expected_income_weights():
    result = _compose(
        lagged_distribution_income=np.zeros(3),
        residual_profile=np.array([-30.0, -20.0, -10.0]),
    )

    np.testing.assert_allclose(result.residual_portfolio_return, [10.0, 30.0, 60.0])


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("lagged_distribution_income", np.array([0.0, np.nan, 0.0])),
        ("residual_profile", np.array([0.0, np.inf, 0.0])),
        ("calibration_target", np.array([0.0, np.nan, 0.0])),
    ],
)
def test__financial_asset_income_rejects_non_finite_inputs(name, values):
    with pytest.raises(ValueError, match="non-finite"):
        _compose(**{name: values})


def test__country_records_lagged_components_without_moving_deposits():
    ts = TimeSeries(
        dividend_fund_hypothetical_total_distribution=np.array([0.0, 10.0]),
        expected_income_financial_assets=np.array([40.0, 60.0]),
        income_financial_assets=np.zeros(2),
        total_income_financial_assets=[0.0],
        income_financial_assets_distribution=np.zeros(2),
        income_financial_assets_residual_portfolio_return=np.zeros(2),
        income_financial_assets_calibration_target=np.zeros(2),
        total_income_financial_assets_distribution=[0.0],
        total_income_financial_assets_residual_portfolio_return=[0.0],
        total_income_financial_assets_calibration_target=[0.0],
        income_financial_assets_target_gap=[0.0],
        income_financial_assets_calibration_error=[0.0],
        wealth_deposits=np.array([100.0, 200.0]),
    )
    deposits_before = ts.current("wealth_deposits").copy()
    expected_before = ts.current("expected_income_financial_assets").copy()
    households = SimpleNamespace(
        ts=ts,
        compute_income_from_financial_assets=lambda period_index=None: np.array([20.0, 30.0]),
    )
    country = SimpleNamespace(households=households)

    result = Country.record_household_financial_asset_income(country, period_index=2)

    np.testing.assert_array_equal(ts.current("income_financial_assets_distribution"), [0.0, 10.0])
    np.testing.assert_allclose(
        ts.current("income_financial_assets"),
        ts.current("income_financial_assets_distribution")
        + ts.current("income_financial_assets_residual_portfolio_return"),
    )
    assert ts.current("total_income_financial_assets")[0] == pytest.approx(100.0)
    assert result.calibration_error == pytest.approx(0.0)
    np.testing.assert_array_equal(ts.current("wealth_deposits"), deposits_before)
    np.testing.assert_array_equal(ts.current("expected_income_financial_assets"), expected_before)
