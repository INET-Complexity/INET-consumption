"""Minimum income pure functions and household_support configuration (PLAN.md increment 1)."""

import numpy as np
import pytest

from macromodel.agents.households.func.minimum_income import (
    assessed_income,
    income_guard,
    minimum_income_level,
    minimum_income_top_up,
    notional_rate_period,
)
from macromodel.configurations import CountryConfiguration
from macromodel.country.country import (
    HouseholdSupportParameters,
    _read_household_support_parameters,
    _validate_minimum_income_base_year,
)

RSA = 499.31


def level(cu=1.0, time_unit=3, scale=1.0, transfer_index=1.0, cpi_ratio=1.0):
    return minimum_income_level(
        rsa_base_monthly_model_units=RSA * scale,
        time_unit=time_unit,
        consumption_units=np.atleast_1d(np.asarray(cu, dtype=float)),
        transfer_index=transfer_index,
        cpi_ratio=cpi_ratio,
    )


def assess(assets=0.0, *, flows=(0.0, 0.0, 0.0, 0.0, 0.0), liquid_share=1.0, time_unit=3):
    employee, unemployment, pension, other, rental = (np.array([float(f)]) for f in flows)
    return assessed_income(
        employee=employee,
        unemployment=unemployment,
        pension=pension,
        other_transfers=other,
        rental=rental,
        liquid_assets=np.array([assets * liquid_share]),
        illiquid_assets=np.array([assets * (1 - liquid_share)]),
        notional_rate_period=notional_rate_period(0.03, time_unit),
    )


def top_up(assets=0.0, **kwargs):
    return minimum_income_top_up(level=level(), assessed=assess(assets, **kwargs))


def test_notional_rate_period():
    assert notional_rate_period(0.03, 3) == pytest.approx(0.0075)
    assert notional_rate_period(0.03, 1) == pytest.approx(0.0025)


def test_zero_income_gets_full_level():
    np.testing.assert_allclose(top_up(0.0), [1497.93])
    np.testing.assert_allclose(top_up(0.0), 3 * RSA)


def test_income_above_level_gets_nothing():
    np.testing.assert_array_equal(top_up(0.0, flows=(2000, 0, 0, 0, 0)), [0.0])
    np.testing.assert_array_equal(top_up(0.0, flows=(1497.93, 0, 0, 0, 0)), [0.0])


def test_euro_for_euro_withdrawal_across_components():
    for position in range(5):
        flows = [0.0] * 5
        flows[position] = 400.0
        np.testing.assert_allclose(top_up(0.0, flows=tuple(flows)), [1497.93 - 400.0])


@pytest.mark.parametrize(
    "assets, expected",
    [(0.0, 1497.93), (10_000.0, 1422.93), (200_000.0, 0.0)],
)
def test_financial_assets_notional_income(assets, expected):
    np.testing.assert_allclose(top_up(assets), [expected], atol=1e-9)
    np.testing.assert_allclose(top_up(assets), [max(3 * RSA - 0.0075 * assets, 0.0)], atol=1e-9)


def test_assets_cutoff_and_liquid_illiquid_are_summed():
    cutoff = 3 * RSA / 0.0075
    assert cutoff == pytest.approx(199_724.0)
    np.testing.assert_allclose(top_up(cutoff, liquid_share=0.3), [0.0], atol=1e-9)
    np.testing.assert_allclose(top_up(10_000.0, liquid_share=0.4), top_up(10_000.0, liquid_share=1.0))


@pytest.mark.parametrize("cu", [1.0, 1.8, 2.1])
def test_level_scales_with_consumption_units(cu):
    np.testing.assert_allclose(level(cu), [3 * RSA * cu])


def test_level_vector_of_households():
    np.testing.assert_allclose(level([1.0, 1.8, 2.1]), 3 * RSA * np.array([1.0, 1.8, 2.1]))


def test_monthly_vs_quarterly():
    np.testing.assert_allclose(level(time_unit=1), [RSA])
    np.testing.assert_allclose(level(time_unit=3), 3 * level(time_unit=1))
    # Monthly notional rate is a third of the quarterly one; the cutoff in assets is unchanged.
    monthly = minimum_income_top_up(level=level(time_unit=1), assessed=assess(10_000.0, time_unit=1))
    np.testing.assert_allclose(monthly, [RSA - 0.0025 * 10_000.0])
    np.testing.assert_allclose(monthly * 3, top_up(10_000.0))


def test_transfer_index_and_cpi_ratio():
    np.testing.assert_allclose(level(transfer_index=1.1, cpi_ratio=1.25), [3 * RSA * 1.1 * 1.25])
    np.testing.assert_allclose(level(transfer_index=1.1), [3 * RSA * 1.1])
    np.testing.assert_allclose(level(cpi_ratio=1.25), [3 * RSA * 1.25])


def test_population_scale_enters_level_once():
    np.testing.assert_allclose(level(scale=5000.0), 5000.0 * level())


def test_negative_rental_income_can_push_top_up_above_level():
    m = level()
    t = minimum_income_top_up(level=m, assessed=assess(0.0, flows=(0, 0, 0, 0, -200.0)))
    np.testing.assert_allclose(t, m + 200.0)
    assert t[0] > m[0]


def test_guard_is_one_percent_of_level():
    m = level([1.0, 1.8])
    np.testing.assert_allclose(income_guard(level=m, share=0.01), 0.01 * 3 * RSA * np.array([1.0, 1.8]))


@pytest.mark.parametrize("share", [0.0, 1.0, -0.1, np.nan])
def test_guard_share_must_lie_in_open_unit_interval(share):
    with pytest.raises(ValueError):
        income_guard(level=level(), share=share)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="consumption_units"):
        level(cu=0.0)
    with pytest.raises(ValueError, match="finite"):
        level(cu=np.nan)
    with pytest.raises(ValueError, match="transfer_index"):
        level(transfer_index=0.0)
    with pytest.raises(ValueError, match="cpi_ratio"):
        level(cpi_ratio=np.inf)
    with pytest.raises(ValueError, match="equal shapes"):
        assessed_income(
            employee=np.zeros(2),
            unemployment=np.zeros(1),
            pension=np.zeros(1),
            other_transfers=np.zeros(1),
            rental=np.zeros(1),
            liquid_assets=np.zeros(1),
            illiquid_assets=np.zeros(1),
            notional_rate_period=0.0075,
        )
    with pytest.raises(ValueError, match="non-negative"):
        assess(-1.0)
    with pytest.raises(ValueError, match="finite"):
        assess(0.0, flows=(np.nan, 0, 0, 0, 0))
    with pytest.raises(ValueError, match="equal shapes"):
        minimum_income_top_up(level=np.ones(2), assessed=np.ones(3))


def test_purity_inputs_are_not_modified():
    m, y = level([1.0, 2.0]), np.array([100.0, 5000.0])
    m0, y0 = m.copy(), y.copy()
    t = minimum_income_top_up(level=m, assessed=y)
    np.testing.assert_array_equal(m, m0)
    np.testing.assert_array_equal(y, y0)
    np.testing.assert_allclose(t, [m0[0] - 100.0, 0.0])


# --- configuration ---------------------------------------------------------------------------------


def test_default_block_is_legacy_with_goods_floor():
    parsed = _read_household_support_parameters({})
    assert parsed == HouseholdSupportParameters()
    assert parsed.mode == "legacy_stage5" and parsed.goods_floor is True
    assert _read_household_support_parameters(None) == parsed
    assert CountryConfiguration().household_support == {}


def test_full_block_is_parsed():
    parsed = _read_household_support_parameters(
        {
            "mode": "minimum_income",
            "goods_floor": False,
            "minimum_income": {
                "rsa_base_monthly": 499.31,
                "base_year": 2014,
                "notional_asset_return_annual": 0.03,
                "guard_share_of_m": 0.01,
            },
        }
    )
    assert parsed.mode == "minimum_income" and parsed.goods_floor is False
    assert parsed.minimum_income.rsa_base_monthly == 499.31
    assert parsed.minimum_income.base_year == 2014
    assert parsed.minimum_income.guard_share_of_m == 0.01


@pytest.mark.parametrize(
    "block, message",
    [
        ({"unknown": 1}, "Unknown household_support keys"),
        ({"minimum_income": {"unknown": 1}}, "Unknown household_support.minimum_income keys"),
        ({"mode": "other"}, "mode"),
        ({"mode": "legacy_stage5", "goods_floor": False}, "requires goods_floor"),
        ({"goods_floor": False}, "requires goods_floor"),
        ({"goods_floor": "yes"}, "boolean"),
        ({"minimum_income": {"rsa_base_monthly": float("nan")}}, "rsa_base_monthly"),
        ({"minimum_income": {"rsa_base_monthly": float("inf")}}, "rsa_base_monthly"),
        ({"minimum_income": {"rsa_base_monthly": 0.0}}, "rsa_base_monthly"),
        ({"minimum_income": {"rsa_base_monthly": -1.0}}, "rsa_base_monthly"),
        ({"minimum_income": {"rsa_base_monthly": "499"}}, "rsa_base_monthly"),
        ({"minimum_income": {"base_year": float("nan")}}, "base_year"),
        ({"minimum_income": {"base_year": 2014.5}}, "base_year"),
        ({"minimum_income": {"notional_asset_return_annual": float("inf")}}, "notional_asset_return_annual"),
        ({"minimum_income": {"notional_asset_return_annual": -0.03}}, "notional_asset_return_annual"),
        ({"minimum_income": {"guard_share_of_m": 0.0}}, "guard_share_of_m"),
        ({"minimum_income": {"guard_share_of_m": 1.0}}, "guard_share_of_m"),
        ({"minimum_income": {"guard_share_of_m": float("nan")}}, "guard_share_of_m"),
    ],
)
def test_invalid_blocks_are_rejected(block, message):
    with pytest.raises(ValueError, match=message):
        _read_household_support_parameters(block)


def test_base_year_mismatch_raises_only_in_minimum_income_mode():
    minimum = _read_household_support_parameters({"mode": "minimum_income"})
    _validate_minimum_income_base_year(minimum, 2014)
    with pytest.raises(ValueError, match="initial_year"):
        _validate_minimum_income_base_year(minimum, 2016)
    with pytest.raises(ValueError, match="initial_year"):
        _validate_minimum_income_base_year(minimum, None)
    legacy = _read_household_support_parameters({})
    _validate_minimum_income_base_year(legacy, 2016)
    _validate_minimum_income_base_year(legacy, None)
