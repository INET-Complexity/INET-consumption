import numpy as np

from macromodel.agents.households.func.borrow_vs_sell import (
    PREFERRED_MARGIN_BORROW,
    PREFERRED_MARGIN_NONE,
    PREFERRED_MARGIN_SELL,
    BorrowVsSellResult,
    compute_borrow_vs_sell_choice,
)


def _compute(**kwargs):
    defaults = {
        "residual_shortfall_after_lfa": np.asarray([10.0], dtype=float),
        "delta_tilde": np.asarray([0.0], dtype=float),
        "opening_tfa_scale": np.asarray([100.0], dtype=float),
        "post_return_ifa": np.asarray([25.0], dtype=float),
        "r_b": np.asarray([0.08], dtype=float),
        "r_kappa": np.asarray([0.02], dtype=float),
        "phi_1": 1.0,
        "lambda_kappa": 0.5,
    }
    defaults.update(kwargs)
    return compute_borrow_vs_sell_choice(**defaults)


def test__positive_spread_prefers_sell():
    result = _compute(r_b=[0.2])

    np.testing.assert_array_equal(result.preferred_margin, [PREFERRED_MARGIN_SELL])
    np.testing.assert_allclose(result.preferred_amount, [10.0])
    assert result.comparison_valid_flag[0]


def test__negative_spread_prefers_borrow():
    result = _compute(r_b=[0.01])

    np.testing.assert_array_equal(result.preferred_margin, [PREFERRED_MARGIN_BORROW])
    np.testing.assert_allclose(result.preferred_amount, [10.0])
    assert result.comparison_valid_flag[0]


def test__exact_tie_defaults_to_borrow():
    result = _compute(r_b=[0.12])

    np.testing.assert_array_equal(result.preferred_margin, [PREFERRED_MARGIN_BORROW])
    np.testing.assert_allclose(result.borrow_vs_sell_spread, [0.0], atol=1e-15)


def test__zero_or_negative_shortfall_returns_none():
    result = _compute(residual_shortfall_after_lfa=[0.0, -1.0], delta_tilde=[0.0, 0.0], opening_tfa_scale=[100.0, 100.0], post_return_ifa=[25.0, 25.0], r_b=[0.08, 0.08], r_kappa=[0.02, 0.02])

    np.testing.assert_array_equal(result.preferred_margin, [PREFERRED_MARGIN_NONE, PREFERRED_MARGIN_NONE])
    np.testing.assert_allclose(result.preferred_amount, [0.0, 0.0])
    np.testing.assert_array_equal(result.comparison_valid_flag, [False, False])


def test__zero_post_return_ifa_cannot_choose_sell():
    result = _compute(post_return_ifa=[0.0], r_b=[0.2])

    np.testing.assert_array_equal(result.preferred_margin, [PREFERRED_MARGIN_BORROW])
    np.testing.assert_allclose(result.preferred_amount, [10.0])


def test__invalid_delta_tilde_falls_back_to_borrow():
    result = _compute(delta_tilde=[np.nan])

    np.testing.assert_array_equal(result.preferred_margin, [PREFERRED_MARGIN_BORROW])
    np.testing.assert_allclose(result.preferred_amount, [10.0])
    np.testing.assert_array_equal(result.comparison_valid_flag, [False])
    np.testing.assert_allclose(result.borrow_vs_sell_threshold, [0.0])


def test__non_positive_opening_scale_falls_back_to_borrow():
    result = _compute(opening_tfa_scale=[0.0])

    np.testing.assert_array_equal(result.preferred_margin, [PREFERRED_MARGIN_BORROW])
    np.testing.assert_allclose(result.preferred_amount, [10.0])
    np.testing.assert_array_equal(result.comparison_valid_flag, [False])
    np.testing.assert_allclose(result.borrow_vs_sell_l_tilde, [0.0])


def test__non_finite_post_return_ifa_falls_back_to_borrow():
    result = _compute(post_return_ifa=[np.nan])

    np.testing.assert_array_equal(result.preferred_margin, [PREFERRED_MARGIN_BORROW])
    np.testing.assert_allclose(result.preferred_amount, [10.0])
    np.testing.assert_array_equal(result.comparison_valid_flag, [False])


def test__invalid_residual_shortfall_returns_none():
    result = _compute(residual_shortfall_after_lfa=[np.nan])

    np.testing.assert_array_equal(result.preferred_margin, [PREFERRED_MARGIN_NONE])
    np.testing.assert_allclose(result.preferred_amount, [0.0])
    np.testing.assert_array_equal(result.comparison_valid_flag, [False])


def test__phi2_is_derived_internally_from_lambda_and_phi1():
    result = _compute(phi_1=2.0, lambda_kappa=0.25, r_b=[0.6])

    expected_l_tilde = 0.1
    expected_threshold = 0.02 + (2.0 / (2.0 * 0.25)) * (2.0 * 0.25 * 0.0 + expected_l_tilde)
    np.testing.assert_allclose(result.borrow_vs_sell_l_tilde, [expected_l_tilde])
    np.testing.assert_allclose(result.borrow_vs_sell_threshold, [expected_threshold])


def test__returns_typed_result_without_side_effects():
    residual = np.asarray([10.0])
    result = _compute(residual_shortfall_after_lfa=residual)

    assert isinstance(result, BorrowVsSellResult)
    np.testing.assert_allclose(residual, [10.0])
