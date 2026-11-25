"""DNA quintile opening budget (vault row 25, option A): common lambda over total C incl. housing."""

import numpy as np
import pytest

from macro_data.configuration.dataconfiguration import HouseholdConsumptionInitialisationConfiguration
from macro_data.util.consumption_initialisation import allocate_dna_quintile_consumption

DNA = np.array([1.034, 0.984, 0.934, 0.901, 0.707])
VAT = 0.2


def _households(n=100, seed=0):
    rng = np.random.default_rng(seed)
    income = rng.lognormal(3.0, 0.6, n)
    size = rng.integers(1, 5, n).astype(float)
    c_y = rng.uniform(0.6, 1.2, n)
    # Housing share falls with income, so it varies by quintile.
    housing_share = 0.35 - 0.25 * (np.argsort(np.argsort(income)) / n)
    cash_rent = 0.5 * housing_share * income
    imputed_rent = 0.5 * housing_share * income
    legacy = rng.uniform(5.0, 10.0, n)
    return income, size, c_y, cash_rent, imputed_rent, legacy


def _allocate(income, size, c_y, cash_rent, imputed_rent, legacy, anchor=1500.0):
    return allocate_dna_quintile_consumption(
        hfcs_c_y=c_y,
        model_income=income,
        household_size=size,
        cash_rent=cash_rent,
        imputed_rent=imputed_rent,
        vat=VAT,
        aggregate_net_goods=anchor,
        legacy_goods=legacy,
        target_c_y=DNA,
    )


def _quintile_ratio(result, income, size):
    ok = ~result.legacy_fallback
    pos = income > 0
    rank = np.full(len(income), -1)
    rank[pos] = np.argsort(np.argsort(income[pos] / np.sqrt(size[pos]), kind="stable")) * 5 // pos.sum()
    return np.array([result.total[ok & (rank == q)].sum() / income[ok & (rank == q)].sum() for q in range(5)]) / DNA


def test_shape_holds_with_quintile_varying_rents_and_goods_equal_anchor():
    hh = _households()
    result = _allocate(*hh)
    ratio = _quintile_ratio(result, hh[0], hh[1])
    np.testing.assert_allclose(ratio, result.goods_scale, rtol=1e-12)
    np.testing.assert_allclose(result.goods.sum(), 1500.0, rtol=1e-12)
    np.testing.assert_allclose(result.total, (1 + VAT) * result.goods + hh[3] + hh[4])


def test_fallback_and_negative_income_households_keep_legacy_goods():
    income, size, c_y, cash, imputed, legacy = _households()
    income[0] = -5.0  # negative income
    c_y[1], cash[1], imputed[1] = 0.1, 0.4 * income[1], 0.4 * income[1]  # HFCS total below housing
    result = _allocate(income, size, c_y, cash, imputed, legacy)
    assert result.legacy_fallback[0] and result.legacy_fallback[1]
    np.testing.assert_allclose(result.goods[:2], legacy[:2])
    np.testing.assert_allclose(result.goods.sum(), 1500.0, rtol=1e-12)
    ratio = _quintile_ratio(result, income, size)
    np.testing.assert_allclose(ratio, result.goods_scale, rtol=1e-12)
    assert (result.total[~result.legacy_fallback] >= cash[~result.legacy_fallback]).all()


def test_misaligned_household_size_raises():
    income, size, *rest = _households()
    with pytest.raises(ValueError, match="aligned"):
        _allocate(income, size[:-1], *rest)


def test_config_default_off_and_validation():
    assert HouseholdConsumptionInitialisationConfiguration().dna_quintile_c_y is None
    with pytest.raises(ValueError):
        HouseholdConsumptionInitialisationConfiguration(dna_quintile_c_y=[1.0, 1.0])
    with pytest.raises(ValueError):
        HouseholdConsumptionInitialisationConfiguration(dna_quintile_c_y=[1.0, 1.0, 1.0, 1.0, 0.0])
