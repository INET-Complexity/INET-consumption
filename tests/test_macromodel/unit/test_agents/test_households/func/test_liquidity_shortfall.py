import numpy as np

from macromodel.agents.households.func.liquidity_shortfall import (
    LiquidityShortfallResult,
    compute_liquidity_shortfall,
)


def _run(income, target_consumption, scheduled_debt_service) -> LiquidityShortfallResult:
    return compute_liquidity_shortfall(
        income=np.asarray(income, dtype=float),
        target_consumption=np.asarray(target_consumption, dtype=float),
        scheduled_debt_service=np.asarray(scheduled_debt_service, dtype=float),
    )


def test__shortfall_is_positive_when_consumption_plus_debt_service_exceeds_income():
    result = _run(income=[1000.0], target_consumption=[800.0], scheduled_debt_service=[300.0])
    # L^d = target_consumption + scheduled_debt_service - income = 800 + 300 - 1000 = 100
    np.testing.assert_allclose(result.liquidity_shortfall, [100.0])


def test__shortfall_is_zero_or_negative_when_income_covers_consumption_and_debt_service():
    result = _run(income=[1000.0], target_consumption=[500.0], scheduled_debt_service=[200.0])
    # L^d = 500 + 200 - 1000 = -300 (a surplus, not a shortfall)
    np.testing.assert_allclose(result.liquidity_shortfall, [-300.0])


def test__household_saving_is_income_minus_target_consumption_independent_of_debt_service():
    result = _run(income=[1000.0], target_consumption=[700.0], scheduled_debt_service=[50.0])
    np.testing.assert_allclose(result.household_saving, [300.0])


def test__zero_debt_service_reduces_shortfall_to_consumption_minus_income():
    result = _run(income=[900.0], target_consumption=[950.0], scheduled_debt_service=[0.0])
    np.testing.assert_allclose(result.liquidity_shortfall, [50.0])


def test__vectorized_across_households():
    result = _run(
        income=[1000.0, 500.0, 2000.0],
        target_consumption=[800.0, 600.0, 1000.0],
        scheduled_debt_service=[300.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(result.liquidity_shortfall, [100.0, 100.0, -1000.0])
    np.testing.assert_allclose(result.household_saving, [200.0, -100.0, 1000.0])


def test__returns_liquidity_shortfall_result_dataclass():
    result = _run(income=[1.0], target_consumption=[1.0], scheduled_debt_service=[0.0])
    assert isinstance(result, LiquidityShortfallResult)
