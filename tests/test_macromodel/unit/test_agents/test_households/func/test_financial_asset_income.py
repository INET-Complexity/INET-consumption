from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.agents.households.func.financial_asset_income import (
    compose_financial_asset_income,
    empirical_financial_income_target,
)
from macromodel.agents.households.households import Households
from macromodel.country.country import Country
from macromodel.timeseries import TimeSeries


def _compose(**overrides):
    inputs = {
        "lagged_distribution_income": np.array([0.0, 20.0, 30.0]),
        "residual_profile": np.array([-10.0, 10.0, 30.0]),
        "expected_non_negative_residual_profile": np.array([0.0, 10.0, 30.0]),
        "calibration_target": np.array([10.0, 30.0, 60.0]),
    }
    inputs.update(overrides)
    return compose_financial_asset_income(**inputs)


def test__residual_return_ifa_proxy_uses_fixed_initial_share_fraction():
    households = SimpleNamespace(
        ts=TimeSeries(wealth_other_financial_assets=np.array([100.0, 200.0, -10.0])),
        states={"dividend_fund_initial_direct_share_fraction": np.array([0.25, 1.0, 0.5])},
    )

    first = Households.residual_return_ifa_proxy(households)
    households.ts.wealth_other_financial_assets.append(np.array([200.0, 50.0, 40.0]))
    second = Households.residual_return_ifa_proxy(households)

    np.testing.assert_allclose(first, [75.0, 0.0, 0.0])
    np.testing.assert_allclose(second, [150.0, 0.0, 20.0])
    np.testing.assert_allclose(
        households.states["dividend_fund_initial_direct_share_fraction"],
        [0.25, 1.0, 0.5],
    )


def test__financial_asset_income_adds_distribution_without_crowding_out_residual():
    result = _compose()

    np.testing.assert_array_equal(result.distribution_income, [0.0, 20.0, 30.0])
    np.testing.assert_allclose(result.residual_portfolio_return, [10.0, 30.0, 60.0])
    np.testing.assert_allclose(result.total_income, [10.0, 50.0, 90.0])
    assert result.aggregate_distribution_income == 50.0
    assert result.aggregate_residual_portfolio_return == 100.0
    assert result.aggregate_expected_residual_portfolio_return == 100.0
    assert result.aggregate_calibration_target == 100.0
    assert result.distribution_excess_over_target == 0.0
    assert result.target_gap == pytest.approx(0.0)
    assert result.residual_calibration_scale == pytest.approx(2.5)
    assert result.calibration_error == pytest.approx(60.0)
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


def test__distribution_above_target_is_preserved_without_reducing_residual():
    result = _compose(lagged_distribution_income=np.array([10.0, 30.0, 61.0]))

    np.testing.assert_array_equal(result.distribution_income, [10.0, 30.0, 61.0])
    np.testing.assert_array_equal(result.residual_portfolio_return, [10.0, 30.0, 60.0])
    np.testing.assert_array_equal(result.total_income, [20.0, 60.0, 121.0])
    assert result.distribution_excess_over_target == pytest.approx(1.0)
    assert result.target_gap == pytest.approx(0.0)


def test__non_positive_realised_profile_produces_zero_current_residual():
    result = _compose(
        lagged_distribution_income=np.zeros(3),
        residual_profile=np.array([-30.0, -20.0, -10.0]),
    )

    np.testing.assert_array_equal(result.residual_portfolio_return, np.zeros(3))
    assert result.target_gap == pytest.approx(100.0)
    assert result.calibration_error == pytest.approx(60.0)


def test__stochastic_residual_does_not_force_period_target_identity():
    result = _compose(
        residual_profile=np.array([0.0, 5.0, 15.0]),
    )

    assert result.stochastic_multiplier == pytest.approx(0.5)
    assert result.total_income.sum() == pytest.approx(100.0)
    assert result.target_gap == pytest.approx(50.0)
    assert result.calibration_error == pytest.approx(60.0)


def test__empirical_target_scales_initial_observed_yield_by_current_ifa():
    result = empirical_financial_income_target(
        current_ifa=np.array([50.0, 150.0]),
        initial_ifa=np.array([100.0, 100.0]),
        initial_financial_income=np.array([4.0, 6.0]),
    )

    np.testing.assert_allclose(result, [2.5, 7.5])


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("lagged_distribution_income", np.array([0.0, np.nan, 0.0])),
        ("residual_profile", np.array([0.0, np.inf, 0.0])),
        ("expected_non_negative_residual_profile", np.array([0.0, np.inf, 0.0])),
        ("calibration_target", np.array([0.0, np.nan, 0.0])),
    ],
)
def test__financial_asset_income_rejects_non_finite_inputs(name, values):
    with pytest.raises(ValueError, match="non-finite"):
        _compose(**{name: values})


def test__country_records_lagged_components_without_moving_deposits():
    ts = TimeSeries(
        dividend_fund_calibrated_total_distribution=np.array([0.0, 10.0]),
        dividend_fund_settled_firm_distribution=np.zeros(2),
        dividend_fund_settled_bank_distribution=np.zeros(2),
        expected_income_financial_assets=np.array([40.0, 60.0]),
        income_financial_assets=np.zeros(2),
        total_income_financial_assets=[0.0],
        income_financial_assets_distribution=np.zeros(2),
        income_financial_assets_residual_portfolio_return=np.zeros(2),
        income_financial_assets_calibration_target=np.array([40.0, 60.0]),
        total_income_financial_assets_distribution=[0.0],
        total_income_financial_assets_residual_portfolio_return=[0.0],
        total_income_financial_assets_calibration_target=[0.0],
        income_financial_assets_distribution_excess_over_target=[0.0],
        income_financial_assets_target_gap=[0.0],
        income_financial_assets_calibration_error=[0.0],
        income_financial_assets_residual_calibration_scale=[1.0],
        wealth_other_financial_assets_capital_gains=np.zeros(2),
        total_wealth_other_financial_assets_capital_gains=[0.0],
        wealth_deposits=np.array([100.0, 200.0]),
        wealth_other_financial_assets=np.array([40.0, 60.0]),
    )
    deposits_before = ts.current("wealth_deposits").copy()
    expected_before = ts.current("expected_income_financial_assets").copy()
    households = SimpleNamespace(
        ts=ts,
        states={"dividend_fund_ownership_quota": np.array([0.0, 1.0])},
        stage_illiquid_valuation_return=lambda period_index=None: np.array([20.0, 30.0]),
        expected_non_negative_valuation_return=lambda: np.array([20.0, 30.0]),
    )
    firms = SimpleNamespace(ts=TimeSeries(deposits=np.array([500.0]), dividend_fund_settlement_debit=np.array([0.0])))
    banks = SimpleNamespace(ts=TimeSeries(deposits=np.array([700.0]), dividend_fund_settlement_debit=np.array([10.0])))
    country = SimpleNamespace(households=households, firms=firms, banks=banks)
    firm_deposits_before = firms.ts.current("deposits").copy()
    bank_deposits_before = banks.ts.current("deposits").copy()

    result = Country.record_household_financial_asset_income(country, period_index=2)

    np.testing.assert_array_equal(ts.current("income_financial_assets_distribution"), [0.0, 10.0])
    np.testing.assert_allclose(
        ts.current("income_financial_assets"),
        ts.current("income_financial_assets_distribution")
        + ts.current("income_financial_assets_residual_portfolio_return"),
    )
    assert ts.current("total_income_financial_assets")[0] == pytest.approx(110.0)
    assert result.calibration_error == pytest.approx(50.0)
    assert result.residual_calibration_scale == pytest.approx(2.0)
    assert ts.current("income_financial_assets_distribution_excess_over_target")[0] == 0.0
    np.testing.assert_array_equal(ts.current("wealth_other_financial_assets_capital_gains"), np.zeros(2))
    np.testing.assert_array_equal(ts.current("wealth_deposits"), deposits_before)
    np.testing.assert_array_equal(ts.current("expected_income_financial_assets"), expected_before)
    np.testing.assert_array_equal(firms.ts.current("deposits"), firm_deposits_before)
    np.testing.assert_array_equal(banks.ts.current("deposits"), bank_deposits_before)


def test__country_consumes_lagged_distribution_and_records_target_excess():
    ts = TimeSeries(
        dividend_fund_calibrated_total_distribution=np.zeros(2),
        dividend_fund_settled_firm_distribution=np.zeros(2),
        dividend_fund_settled_bank_distribution=np.zeros(2),
        income_financial_assets=np.array([40.0, 60.0]),
        total_income_financial_assets=[100.0],
        expected_income_financial_assets=np.array([40.0, 60.0]),
        income_financial_assets_distribution=np.zeros(2),
        income_financial_assets_residual_portfolio_return=np.array([40.0, 60.0]),
        income_financial_assets_calibration_target=np.array([40.0, 60.0]),
        total_income_financial_assets_distribution=[0.0],
        total_income_financial_assets_residual_portfolio_return=[100.0],
        total_income_financial_assets_calibration_target=[100.0],
        income_financial_assets_distribution_excess_over_target=[0.0],
        income_financial_assets_target_gap=[0.0],
        income_financial_assets_calibration_error=[0.0],
        income_financial_assets_residual_calibration_scale=[1.0],
        wealth_other_financial_assets_capital_gains=np.zeros(2),
        total_wealth_other_financial_assets_capital_gains=[0.0],
        wealth_other_financial_assets=np.array([40.0, 60.0]),
    )
    households = SimpleNamespace(
        ts=ts,
        states={"dividend_fund_ownership_quota": np.array([40.0 / 101.0, 61.0 / 101.0])},
        stage_illiquid_valuation_return=lambda period_index=None: np.array([20.0, 30.0]),
        expected_non_negative_valuation_return=lambda: np.array([20.0, 30.0]),
    )
    firms = SimpleNamespace(ts=TimeSeries(dividend_fund_settlement_debit=np.array([0.0])))
    banks = SimpleNamespace(ts=TimeSeries(dividend_fund_settlement_debit=np.array([0.0])))
    country = SimpleNamespace(households=households, firms=firms, banks=banks)

    first = Country.record_household_financial_asset_income(country, period_index=1)
    banks.ts.dividend_fund_settlement_debit.append(np.array([101.0]))
    second = Country.record_household_financial_asset_income(country, period_index=2)

    np.testing.assert_array_equal(first.distribution_income, np.zeros(2))
    np.testing.assert_array_equal(second.distribution_income, [40.0, 61.0])
    np.testing.assert_array_equal(second.residual_portfolio_return, [40.0, 60.0])
    assert second.distribution_excess_over_target == pytest.approx(1.0)
    assert ts.current("income_financial_assets_distribution_excess_over_target")[0] == pytest.approx(1.0)


def test__known_distribution_is_handed_into_expected_income_composition():
    target = np.array([40.0, 60.0])

    without_distribution = compose_financial_asset_income(
        lagged_distribution_income=np.zeros(2),
        residual_profile=target,
        expected_non_negative_residual_profile=target,
        calibration_target=target,
    )
    with_distribution = compose_financial_asset_income(
        lagged_distribution_income=np.array([4.0, 6.0]),
        residual_profile=target,
        expected_non_negative_residual_profile=target,
        calibration_target=target,
    )

    np.testing.assert_array_equal(with_distribution.distribution_income, [4.0, 6.0])
    np.testing.assert_array_equal(with_distribution.residual_portfolio_return, target)
    np.testing.assert_array_equal(with_distribution.total_income, [44.0, 66.0])
    np.testing.assert_array_equal(without_distribution.total_income, target)


def test__distribution_changes_realised_saving_income_when_residual_shock_is_not_at_expectation():
    target = np.array([40.0, 60.0])
    realised_profile = np.array([20.0, 30.0])

    without_distribution = compose_financial_asset_income(
        lagged_distribution_income=np.zeros(2),
        residual_profile=realised_profile,
        expected_non_negative_residual_profile=target,
        calibration_target=target,
    )
    with_distribution = compose_financial_asset_income(
        lagged_distribution_income=np.array([4.0, 6.0]),
        residual_profile=realised_profile,
        expected_non_negative_residual_profile=target,
        calibration_target=target,
    )

    assert with_distribution.total_income.sum() == pytest.approx(60.0)
    assert without_distribution.total_income.sum() == pytest.approx(50.0)
