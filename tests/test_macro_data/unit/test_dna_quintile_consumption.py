"""DNA quintile shape factors for the HFCS-ratio opening budget (vault row 25, option A)."""

import numpy as np
import pytest

from macro_data.configuration.dataconfiguration import HouseholdConsumptionInitialisationConfiguration
from macro_data.util.consumption_initialisation import allocate_hfcs_ratio_consumption, dna_quintile_factors

DNA = np.array([1.034, 0.984, 0.934, 0.901, 0.707])


def _households(n=50, seed=0):
    rng = np.random.default_rng(seed)
    income = rng.lognormal(3.0, 0.6, n)
    size = rng.integers(1, 5, n).astype(float)
    c_y = rng.uniform(0.5, 1.2, n)
    return income, size, c_y


def _quintile_cy(total, income, size):
    rank = np.argsort(np.argsort(income / np.sqrt(size), kind="stable"), kind="stable")
    q = rank * 5 // len(rank)
    return np.array([total[q == k].sum() / income[q == k].sum() for k in range(5)])


def _allocate(c_y, income, anchor):
    zeros = np.zeros_like(income)
    return allocate_hfcs_ratio_consumption(
        hfcs_c_y=c_y, model_income=income, cash_rent=zeros, imputed_rent=zeros, vat=0.2, aggregate_net_goods=anchor
    )


def test_off_reproduces_current_allocation():
    income, size, c_y = _households()
    assert HouseholdConsumptionInitialisationConfiguration().dna_quintile_c_y is None
    np.testing.assert_allclose(_allocate(c_y, income, 300.0).total, _allocate(c_y.copy(), income, 300.0).total)


def test_quintile_shape_follows_dna_and_goods_equal_anchor():
    income, size, c_y = _households()
    factors = dna_quintile_factors(c_y, income, size, DNA)
    result = _allocate(c_y * factors, income, 300.0)
    cy = _quintile_cy(result.total, income, size)
    np.testing.assert_allclose(cy / cy[0], DNA / DNA[0])
    np.testing.assert_allclose(result.goods.sum(), 300.0)
    np.testing.assert_allclose(cy / DNA, result.goods_scale)


def test_config_rejects_wrong_length_or_nonpositive():
    with pytest.raises(ValueError):
        HouseholdConsumptionInitialisationConfiguration(dna_quintile_c_y=[1.0, 1.0])
    with pytest.raises(ValueError):
        HouseholdConsumptionInitialisationConfiguration(dna_quintile_c_y=[1.0, 1.0, 1.0, 1.0, 0.0])
