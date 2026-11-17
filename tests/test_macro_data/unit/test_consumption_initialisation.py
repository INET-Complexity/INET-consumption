"""Economic and accounting contracts for the HFCS-ratio opening allocation."""

import numpy as np
import pytest

from macro_data.util.consumption_initialisation import allocate_hfcs_ratio_consumption


def inputs():
    return {
        "hfcs_c_y": np.array([0.6, 0.9]),
        "model_income": np.array([100.0, 200.0]),
        "cash_rent": np.array([0.0, 30.0]),
        "imputed_rent": np.array([12.0, 0.0]),
        "vat": 0.2,
        "aggregate_net_goods": 165.0,
    }


def test_preserves_source_ratios_when_aggregate_already_matches():
    args = inputs()
    saved = {k: v.copy() for k, v in args.items() if isinstance(v, np.ndarray)}
    result = allocate_hfcs_ratio_consumption(**args)
    np.testing.assert_allclose(result.goods, [40.0, 125.0])
    np.testing.assert_allclose(result.total / args["model_income"], args["hfcs_c_y"])
    assert result.goods_scale == 1.0
    for name, before in saved.items():
        np.testing.assert_array_equal(args[name], before)


def test_reconciles_goods_without_scaling_housing_or_applying_vat_twice():
    args = inputs()
    args["aggregate_net_goods"] = 330.0
    result = allocate_hfcs_ratio_consumption(**args)
    assert result.goods.sum() == pytest.approx(330.0)
    assert result.goods_scale == pytest.approx(2.0)
    np.testing.assert_allclose(result.goods, [80.0, 250.0])
    np.testing.assert_allclose(result.total, [108.0, 330.0])
    np.testing.assert_allclose(result.candidate_total, [60.0, 180.0])
    assert not np.allclose(result.total / args["model_income"], args["hfcs_c_y"])


def test_common_period_and_population_scaling_commutes_with_allocation():
    args = inputs()
    reference = allocate_hfcs_ratio_consumption(**args)
    for name in ("model_income", "cash_rent", "imputed_rent", "aggregate_net_goods"):
        args[name] = args[name] * 5000 / 4
    result = allocate_hfcs_ratio_consumption(**args)
    np.testing.assert_allclose(result.total, reference.total * 5000 / 4)
    np.testing.assert_allclose(result.goods, reference.goods * 5000 / 4)
    assert result.goods_scale == reference.goods_scale


def test_rejects_housing_shortfall_without_clipping_or_redistribution():
    args = inputs()
    args["cash_rent"] = np.array([100.0, 30.0])
    with pytest.raises(ValueError, match="below housing expenditure for 1 households"):
        allocate_hfcs_ratio_consumption(**args)


@pytest.mark.parametrize("name", ["hfcs_c_y", "model_income", "cash_rent", "imputed_rent"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0])
def test_rejects_invalid_household_observations(name, bad):
    args = inputs()
    args[name][0] = bad
    with pytest.raises(ValueError, match="finite nonnegative"):
        allocate_hfcs_ratio_consumption(**args)


def test_rejects_nonpositive_income_instead_of_using_subsistence_income():
    args = inputs()
    args["model_income"][0] = 0.0
    with pytest.raises(ValueError, match="no income floor"):
        allocate_hfcs_ratio_consumption(**args)


@pytest.mark.parametrize("name", ["vat", "aggregate_net_goods"])
@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf, [1.0]])
def test_rejects_invalid_scalar_controls(name, bad):
    args = inputs()
    args[name] = bad
    with pytest.raises(ValueError, match="finite nonnegative scalar"):
        allocate_hfcs_ratio_consumption(**args)


def test_rejects_broadcastable_but_misaligned_vectors():
    args = inputs()
    args["cash_rent"] = np.zeros(1)
    with pytest.raises(ValueError, match="identical shapes"):
        allocate_hfcs_ratio_consumption(**args)


def test_zero_goods_requires_zero_aggregate_anchor():
    args = inputs()
    args["cash_rent"] = args["hfcs_c_y"] * args["model_income"]
    args["imputed_rent"] = np.zeros(2)
    with pytest.raises(ValueError, match="positive household goods shares"):
        allocate_hfcs_ratio_consumption(**args)
    args["aggregate_net_goods"] = 0.0
    result = allocate_hfcs_ratio_consumption(**args)
    np.testing.assert_array_equal(result.goods, 0.0)
    np.testing.assert_allclose(result.total, result.candidate_total)


def test_winsorisation_matches_calibration_midpoint_interpolation():
    from macro_data.util.consumption_initialisation import winsorise_hfcs_consumption_ratios

    ratios = np.array([0.1, 0.5, 0.9, 2.0])
    weights = np.array([1.0, 8.0, 1.0, 2.0])
    result = winsorise_hfcs_consumption_ratios(
        c_y=ratios,
        survey_weights=weights,
        hard_valid=np.ones(4, dtype=bool),
        lower_quantile=0.1,
        upper_quantile=0.9,
    )
    # Calibration midpoint CDF = [.5, 5, 9.5, 11] / 12.
    assert result.lower == pytest.approx(0.1 + (1.2 - 0.5) / 4.5 * 0.4)
    assert result.upper == pytest.approx(0.9 + (10.8 - 9.5) / 1.5 * 1.1)
    np.testing.assert_allclose(result.ratios[1:3], ratios[1:3])
    np.testing.assert_array_equal(result.capped, [True, False, False, True])
    np.testing.assert_array_equal(result.valid, True)
    np.testing.assert_array_equal(ratios, [0.1, 0.5, 0.9, 2.0])


def test_winsorisation_keeps_tail_rows_and_flags_invalid_rows_without_imputation():
    from macro_data.util.consumption_initialisation import winsorise_hfcs_consumption_ratios

    result = winsorise_hfcs_consumption_ratios(
        c_y=np.array([0.01, 0.5, 1.0, 2.9, np.nan, 9.0]),
        survey_weights=np.ones(6),
        hard_valid=np.array([True, True, True, True, False, False]),
        threshold_pool=np.array([False, True, True, False, False, False]),
    )
    # The threshold pool is selected before percentile trimming. Other rows
    # remain present and are capped, not dropped or used to move the bounds.
    np.testing.assert_allclose(result.ratios[:4], [0.5, 0.5, 1.0, 1.0])
    assert np.isnan(result.ratios[4:]).all()
    np.testing.assert_array_equal(result.valid, [True, True, True, True, False, False])
    np.testing.assert_array_equal(result.capped, [True, False, False, True, False, False])


def test_winsorisation_then_allocation_preserves_capped_ratios_before_macro_alignment():
    from macro_data.util.consumption_initialisation import winsorise_hfcs_consumption_ratios

    result = winsorise_hfcs_consumption_ratios(
        c_y=np.array([0.1, 0.6, 0.9, 2.0]),
        survey_weights=np.ones(4),
        hard_valid=np.ones(4, dtype=bool),
        lower_quantile=0.25,
        upper_quantile=0.75,
    )
    allocation = allocate_hfcs_ratio_consumption(
        hfcs_c_y=result.ratios,
        model_income=np.full(4, 100.0),
        cash_rent=np.zeros(4),
        imputed_rent=np.zeros(4),
        vat=0.2,
        aggregate_net_goods=500.0,
    )
    np.testing.assert_allclose(allocation.candidate_total / 100, result.ratios)
    assert allocation.goods.sum() == pytest.approx(500.0)
    assert not np.allclose(allocation.total / 100, result.ratios)


@pytest.mark.parametrize("change", ["shape", "mask_type", "bad_weight", "bad_ratio", "empty_pool", "quantile_order"])
def test_winsorisation_rejects_inconsistent_preparation(change):
    from macro_data.util.consumption_initialisation import winsorise_hfcs_consumption_ratios

    args = dict(c_y=np.array([0.5, 0.9]), survey_weights=np.ones(2), hard_valid=np.ones(2, dtype=bool))
    if change == "shape":
        args["survey_weights"] = np.ones(1)
    elif change == "mask_type":
        args["hard_valid"] = np.ones(2)
    elif change == "bad_weight":
        args["survey_weights"][0] = 0
    elif change == "bad_ratio":
        args["c_y"][0] = np.nan
    elif change == "empty_pool":
        args["threshold_pool"] = np.zeros(2, dtype=bool)
    else:
        args["lower_quantile"] = 0.99
        args["upper_quantile"] = 0.01
    with pytest.raises(ValueError):
        winsorise_hfcs_consumption_ratios(**args)
