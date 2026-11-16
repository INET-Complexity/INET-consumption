import numpy as np
import pytest

from macro_data.util.consumption_initialisation import (
    allocate_hfcs_ratio_consumption,
    allocate_hfcs_with_legacy_reserve,
)


def inputs():
    return dict(
        hfcs_c_y=np.array([0.6, 0.9, 0.5, 0.9]),
        model_income=np.array([0.0, -10.0, 10.0, 200.0]),
        cash_rent=np.array([5.0, 0.0, 20.0, 30.0]),
        imputed_rent=np.zeros(4),
        vat=0.2,
        aggregate_net_goods=200.0,
        legacy_goods=np.array([10.0, 20.0, 30.0, 40.0]),
    )


def test_reserves_legacy_goods_and_scales_only_feasible_households():
    args = inputs()
    before = {k: v.copy() for k, v in args.items() if isinstance(v, np.ndarray)}
    result = allocate_hfcs_with_legacy_reserve(**args)
    np.testing.assert_array_equal(result.legacy_fallback, [True, True, True, False])
    np.testing.assert_array_equal(result.goods[:3], args["legacy_goods"][:3])
    np.testing.assert_allclose(result.goods, [10, 20, 30, 140])
    assert result.goods.sum() == pytest.approx(200.0)
    np.testing.assert_allclose(result.total, 1.2 * result.goods + args["cash_rent"])
    np.testing.assert_allclose(result.candidate_total, [0, -9, 5, 180])
    for k, v in before.items():
        np.testing.assert_array_equal(args[k], v)


def test_no_fallback_matches_strict_allocation():
    args = inputs()
    args["model_income"] = np.array([100.0, 200.0, 100.0, 200.0])
    expected = allocate_hfcs_ratio_consumption(**{k: v for k, v in args.items() if k != "legacy_goods"})
    actual = allocate_hfcs_with_legacy_reserve(**args)
    assert not actual.legacy_fallback.any()
    np.testing.assert_array_equal(actual.goods, expected.goods)
    np.testing.assert_array_equal(actual.total, expected.total)


def test_all_fallback_keeps_exact_legacy_goods_when_budget_matches():
    args = inputs()
    args["model_income"] = np.zeros(4)
    args["aggregate_net_goods"] = 100.0
    result = allocate_hfcs_with_legacy_reserve(**args)
    assert result.legacy_fallback.all()
    np.testing.assert_array_equal(result.goods, args["legacy_goods"])


@pytest.mark.parametrize("budget,match", [(50.0, "exceed"), (110.0, "No feasible")])
def test_impossible_reserve_does_not_rescale_fallback_households(budget, match):
    args = inputs()
    args["model_income"] = np.zeros(4)
    args["aggregate_net_goods"] = budget
    with pytest.raises(ValueError, match=match):
        allocate_hfcs_with_legacy_reserve(**args)


@pytest.mark.parametrize("field", ["model_income", "hfcs_c_y", "cash_rent", "imputed_rent", "legacy_goods"])
def test_missing_values_are_not_treated_as_fallback(field):
    args = inputs()
    args[field][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        allocate_hfcs_with_legacy_reserve(**args)


def test_zero_aggregate_cannot_hide_a_positive_legacy_reserve_in_absolute_tolerance():
    args = inputs()
    args["aggregate_net_goods"] = 0.0
    args["legacy_goods"] = np.array([1e-13, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="exceed"):
        allocate_hfcs_with_legacy_reserve(**args)
