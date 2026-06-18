import numpy as np

from macromodel.agents.households.func.portfolio_diagnostics import (
    Stage4HouseholdDiagnostics,
    compute_stage4_household_diagnostics,
)

PHI_1 = 5.0
LAMBDA_KAPPA = 0.1
FIXED_COST_SHARE = 0.001
DEFAULT_TARGET_SHARE = 0.65


def _run(
    opening_tfa_scale,
    post_surplus_lfa,
    post_return_ifa,
    investable_surplus,
    portfolio_participates,
    target_share_source="scalar",
    default_target_illiquid_share=DEFAULT_TARGET_SHARE,
    phi_1=PHI_1,
    lambda_kappa=LAMBDA_KAPPA,
    fixed_cost_share=FIXED_COST_SHARE,
    frm_covariates=None,
    frm_magnitude_coefficients=None,
    population_scale_factor=None,
    net_wealth_scale_divisor=None,
) -> Stage4HouseholdDiagnostics:
    return compute_stage4_household_diagnostics(
        opening_tfa_scale=np.asarray(opening_tfa_scale, dtype=float),
        post_surplus_lfa=np.asarray(post_surplus_lfa, dtype=float),
        post_return_ifa=np.asarray(post_return_ifa, dtype=float),
        investable_surplus=np.asarray(investable_surplus, dtype=float),
        portfolio_participates=np.asarray(portfolio_participates, dtype=bool),
        target_share_source=target_share_source,
        default_target_illiquid_share=default_target_illiquid_share,
        phi_1=phi_1,
        lambda_kappa=lambda_kappa,
        fixed_cost_share=fixed_cost_share,
        frm_covariates=frm_covariates,
        frm_magnitude_coefficients=frm_magnitude_coefficients,
        population_scale_factor=population_scale_factor,
        net_wealth_scale_divisor=net_wealth_scale_divisor,
    )


def test__post_surplus_lfa_is_passed_through_unchanged_to_target_tfa_base():
    result = _run(
        opening_tfa_scale=[1000.0],
        post_surplus_lfa=[600.0],
        post_return_ifa=[400.0],
        investable_surplus=[100.0],
        portfolio_participates=[True],
    )
    # post_surplus_lfa is supplied directly (already inclusive of any surplus);
    # target_tfa_base = post_surplus_lfa + post_return_ifa = 600 + 400 = 1000.
    np.testing.assert_allclose(result.portfolio_post_return_lfa, [600.0])
    np.testing.assert_allclose(result.portfolio_target_tfa_base, [1000.0])


def test__investable_surplus_is_recorded_as_diagnostic_only_not_added_to_lfa():
    result = _run(
        opening_tfa_scale=[1000.0],
        post_surplus_lfa=[500.0],
        post_return_ifa=[400.0],
        investable_surplus=[-200.0],
        portfolio_participates=[True],
    )
    # investable_surplus does not participate in the target_tfa_base arithmetic
    # at all in this function (the caller is responsible for incorporating any
    # surplus into post_surplus_lfa before calling); it is only echoed back.
    np.testing.assert_allclose(result.portfolio_post_return_lfa, [500.0])
    np.testing.assert_allclose(result.portfolio_target_tfa_base, [900.0])
    np.testing.assert_allclose(result.portfolio_investable_surplus, [-200.0])


def test__nonparticipant_gets_zero_target_share_and_zero_target_assets():
    result = _run(
        opening_tfa_scale=[1000.0],
        post_surplus_lfa=[500.0],
        post_return_ifa=[0.0],
        investable_surplus=[100.0],
        portfolio_participates=[False],
    )
    assert result.portfolio_target_illiquid_share[0] == 0.0
    assert result.rebalancing.target_illiquid_assets[0] == 0.0


def test__participant_gets_default_target_share_applied_to_target_tfa_base():
    result = _run(
        opening_tfa_scale=[1000.0],
        post_surplus_lfa=[600.0],
        post_return_ifa=[400.0],
        investable_surplus=[0.0],
        portfolio_participates=[True],
    )
    assert result.portfolio_target_illiquid_share[0] == DEFAULT_TARGET_SHARE
    np.testing.assert_allclose(
        result.rebalancing.target_illiquid_assets, [DEFAULT_TARGET_SHARE * result.portfolio_target_tfa_base[0]]
    )


def test__rebalancing_helper_receives_target_tfa_base_as_separate_argument():
    # Regression guard for the wrong-denominator bug Increment 2's second review
    # caught: actual_illiquid_share must use target_tfa_base, not opening_tfa_scale.
    result = _run(
        opening_tfa_scale=[200.0],
        post_surplus_lfa=[600.0],
        post_return_ifa=[400.0],
        investable_surplus=[0.0],
        portfolio_participates=[True],
    )
    # target_tfa_base = 600 + 400 = 1000, opening_tfa_scale = 200 (deliberately
    # different so the two denominators are distinguishable in the assertion).
    expected_actual_share = 400.0 / 1000.0
    np.testing.assert_allclose(result.rebalancing.actual_illiquid_share, [expected_actual_share])


def test__vectorized_multi_household_batch_runs_without_cross_contamination():
    result = _run(
        opening_tfa_scale=[1000.0, 0.0, -50.0, 500.0],
        post_surplus_lfa=[600.0, 0.0, -10.0, 300.0],
        post_return_ifa=[400.0, 0.0, 0.0, 100.0],
        investable_surplus=[100.0, -50.0, 5.0, 0.0],
        portfolio_participates=[True, False, True, False],
    )
    assert result.rebalancing.portfolio_valid_flag.tolist() == [True, False, False, True]
    # Nonparticipant with no_financial_assets_flag still gets a zero target
    # share and zero target assets, independent of the other rows' values.
    assert result.portfolio_target_illiquid_share[1] == 0.0
    assert result.portfolio_target_illiquid_share[3] == 0.0


def test__non_finite_post_surplus_lfa_propagates_to_invalid_flag():
    result = _run(
        opening_tfa_scale=[1000.0],
        post_surplus_lfa=[np.nan],
        post_return_ifa=[400.0],
        investable_surplus=[0.0],
        portfolio_participates=[True],
    )
    assert not result.rebalancing.portfolio_valid_flag[0]


def test__non_finite_investable_surplus_does_not_invalidate_the_household():
    # investable_surplus is diagnostic-only and is not part of the Increment 2
    # helper's finite-input check, so a non-finite value here must not flip
    # portfolio_valid_flag (unlike a non-finite post_surplus_lfa/opening_tfa_scale).
    result = _run(
        opening_tfa_scale=[1000.0],
        post_surplus_lfa=[600.0],
        post_return_ifa=[400.0],
        investable_surplus=[np.nan],
        portfolio_participates=[True],
    )
    assert result.rebalancing.portfolio_valid_flag[0]
    assert np.isnan(result.portfolio_investable_surplus[0])


_MAGNITUDE_COEFFICIENTS = {
    "constant": 0.0418,
    "age": 0.0101,
    "household_members_in_employment": -0.1126,
    "investment_attitudes": -0.141,
    "mortgagor": -0.1403,
    "owner": -0.1749,
    "net_wealth": 0.0345,
}


def test__frm_magnitude_source_produces_nonscalar_target_share_via_full_composition():
    # End-to-end through compute_stage4_household_diagnostics, not just the
    # target-share helper in isolation: confirms target_share_source=
    # "frm_magnitude" composes correctly with target_tfa_base/rebalancing.
    result = _run(
        opening_tfa_scale=[1000.0],
        post_surplus_lfa=[600.0],
        post_return_ifa=[400.0],
        investable_surplus=[0.0],
        portfolio_participates=[True],
        target_share_source="frm_magnitude",
        frm_covariates={
            "age": np.array([40.0]),
            "household_members_in_employment": np.array([1.0]),
            "investment_attitudes": np.array([2.0]),
            "mortgagor": np.array([0.0]),
            "owner": np.array([1.0]),
            "net_wealth": np.array([100_000.0]),
        },
        frm_magnitude_coefficients=_MAGNITUDE_COEFFICIENTS,
        population_scale_factor=5000.0,
        net_wealth_scale_divisor=100_000.0,
    )

    assert result.portfolio_target_illiquid_share[0] != DEFAULT_TARGET_SHARE
    np.testing.assert_allclose(
        result.rebalancing.target_illiquid_assets,
        [result.portfolio_target_illiquid_share[0] * result.portfolio_target_tfa_base[0]],
    )
