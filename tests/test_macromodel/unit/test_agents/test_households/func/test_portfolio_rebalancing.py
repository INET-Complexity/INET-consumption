import math

import numpy as np
import pytest

from macromodel.agents.households.func.portfolio_rebalancing import (
    PortfolioRebalancingResult,
    compute_portfolio_rebalancing,
)

PHI_1 = 5.0
LAMBDA_KAPPA = 0.1
FIXED_COST_SHARE = 0.001


def _run(
    opening_tfa_scale,
    post_return_lfa,
    post_return_ifa,
    target_illiquid_assets,
    portfolio_participates,
    target_tfa_base=None,
    phi_1=PHI_1,
    lambda_kappa=LAMBDA_KAPPA,
    fixed_cost_share=FIXED_COST_SHARE,
) -> PortfolioRebalancingResult:
    post_return_lfa = np.asarray(post_return_lfa, dtype=float)
    post_return_ifa = np.asarray(post_return_ifa, dtype=float)
    if target_tfa_base is None:
        # Default: no separate investable surplus modeled in this scenario,
        # so target_tfa_base = post_surplus_lfa + post_return_ifa collapses
        # to post_return_lfa + post_return_ifa (post_surplus_lfa with a
        # zero/negative surplus branch).
        target_tfa_base = post_return_lfa + post_return_ifa
    return compute_portfolio_rebalancing(
        opening_tfa_scale=np.asarray(opening_tfa_scale, dtype=float),
        post_return_lfa=post_return_lfa,
        post_return_ifa=post_return_ifa,
        target_illiquid_assets=np.asarray(target_illiquid_assets, dtype=float),
        target_tfa_base=np.asarray(target_tfa_base, dtype=float),
        portfolio_participates=np.asarray(portfolio_participates, dtype=bool),
        phi_1=phi_1,
        lambda_kappa=lambda_kappa,
        fixed_cost_share=fixed_cost_share,
    )


# ---------------------------------------------------------------------------
# lambda_kappa domain validation
# ---------------------------------------------------------------------------


class TestLambdaKappaValidation:
    def test__lambda_kappa_zero_raises(self):
        with pytest.raises(ValueError):
            _run([1.0], [1.0], [1.0], [1.0], [True], lambda_kappa=0.0)

    def test__lambda_kappa_negative_raises(self):
        with pytest.raises(ValueError):
            _run([1.0], [1.0], [1.0], [1.0], [True], lambda_kappa=-0.1)

    def test__lambda_kappa_above_one_raises(self):
        with pytest.raises(ValueError):
            _run([1.0], [1.0], [1.0], [1.0], [True], lambda_kappa=1.5)

    def test__lambda_kappa_exactly_one_is_valid(self):
        result = _run([1.0], [20.0], [10.0], [10.5], [True], lambda_kappa=1.0)
        assert result.portfolio_valid_flag[0]


# ---------------------------------------------------------------------------
# Zero / negative TFA
# ---------------------------------------------------------------------------


class TestNoFinancialAssets:
    def test__zero_opening_tfa_zeroes_all_numeric_outputs(self):
        result = _run([0.0], [5.0], [5.0], [3.0], [True])

        assert result.no_financial_assets_flag[0]
        assert not result.portfolio_valid_flag[0]
        np.testing.assert_allclose(result.delta_tilde, [0.0])
        np.testing.assert_allclose(result.kappa_star_tilde, [0.0])
        np.testing.assert_allclose(result.kappa_tilde, [0.0])
        np.testing.assert_allclose(result.desired_illiquid_adjustment, [0.0])
        np.testing.assert_allclose(result.adjustment_cost, [0.0])
        np.testing.assert_allclose(result.counterfactual_lfa_flow, [0.0])
        np.testing.assert_allclose(result.counterfactual_ifa_flow, [0.0])
        np.testing.assert_allclose(result.actual_illiquid_share, [0.0])
        assert not result.inaction_flag[0]
        assert not result.upper_bound_flag[0]
        assert not result.lower_bound_flag[0]

    def test__negative_opening_tfa_zeroes_all_numeric_outputs(self):
        result = _run([-5.0], [5.0], [5.0], [3.0], [True])

        assert result.no_financial_assets_flag[0]
        assert not result.portfolio_valid_flag[0]
        np.testing.assert_allclose(result.kappa_tilde, [0.0])
        np.testing.assert_allclose(result.desired_illiquid_adjustment, [0.0])

    def test__positive_opening_tfa_is_not_flagged(self):
        result = _run([1.0], [20.0], [10.0], [10.5], [True])

        assert not result.no_financial_assets_flag[0]
        assert result.portfolio_valid_flag[0]


# ---------------------------------------------------------------------------
# actual_illiquid_share denominator (target_tfa_base, not opening_tfa_scale)
# ---------------------------------------------------------------------------


class TestActualIlliquidShareDenominator:
    def test__uses_target_tfa_base_not_opening_tfa_scale(self):
        # opening_tfa_scale=100 (stale, previous-period total), but this
        # period's post-return-and-surplus total (target_tfa_base) is 115
        # (post_return_ifa=60, post_return_lfa=45, plus surplus 10).
        # actual_illiquid_share must reflect "right now" (60/115), not the
        # stale previous-period scale (60/100).
        result = _run(
            opening_tfa_scale=[100.0],
            post_return_lfa=[45.0],
            post_return_ifa=[60.0],
            target_illiquid_assets=[70.0],
            portfolio_participates=[True],
            target_tfa_base=[115.0],
        )

        np.testing.assert_allclose(result.actual_illiquid_share, [60.0 / 115.0])
        assert not np.isclose(result.actual_illiquid_share[0], 60.0 / 100.0)

    def test__zero_target_tfa_base_does_not_divide_by_zero(self):
        result = _run(
            opening_tfa_scale=[1.0],
            post_return_lfa=[0.0],
            post_return_ifa=[0.0],
            target_illiquid_assets=[0.0],
            portfolio_participates=[False],
            target_tfa_base=[0.0],
        )

        assert np.isfinite(result.actual_illiquid_share[0])
        np.testing.assert_allclose(result.actual_illiquid_share, [0.0])

    def test__non_finite_target_tfa_base_flags_invalid(self):
        result = _run(
            opening_tfa_scale=[1.0],
            post_return_lfa=[20.0],
            post_return_ifa=[10.0],
            target_illiquid_assets=[10.5],
            portfolio_participates=[True],
            target_tfa_base=[np.nan],
        )

        assert not result.portfolio_valid_flag[0]
        np.testing.assert_allclose(result.actual_illiquid_share, [0.0])

    def test__negative_target_tfa_base_does_not_blow_up_the_share(self):
        # target_tfa_base < 0 is reachable when post_return_lfa +
        # post_return_ifa is itself negative (e.g. overdraft-style negative
        # deposits). The household-period is otherwise valid (opening_tfa_scale
        # > 0, all inputs finite) -- only this one diagnostic has no
        # meaningful value. A naive max(target_tfa_base, eps) floor would
        # divide by a near-zero epsilon and produce an enormous, meaningless
        # ratio instead of signalling "no measurable share."
        result = _run(
            opening_tfa_scale=[1.0],
            post_return_lfa=[20.0],
            post_return_ifa=[10.0],
            target_illiquid_assets=[10.5],
            portfolio_participates=[True],
            target_tfa_base=[-50.0],
        )

        assert result.portfolio_valid_flag[0]
        np.testing.assert_allclose(result.actual_illiquid_share, [0.0])
        assert result.actual_illiquid_share[0] < 1.0


# ---------------------------------------------------------------------------
# Target below / above actual share (sign of delta_tilde)
# ---------------------------------------------------------------------------


class TestDeltaTildeSign:
    def test__target_above_actual_gives_positive_delta_and_acquisition(self):
        result = _run([1.0], [20.0], [10.0], [10.5], [True])

        assert result.delta_tilde[0] > 0.0
        assert result.kappa_tilde[0] > 0.0
        assert result.counterfactual_ifa_flow[0] > 0.0

    def test__target_below_actual_gives_negative_delta_and_liquidation(self):
        result = _run([1.0], [20.0], [10.0], [9.5], [True])

        assert result.delta_tilde[0] < 0.0
        assert result.kappa_tilde[0] < 0.0
        assert result.counterfactual_ifa_flow[0] < 0.0


# ---------------------------------------------------------------------------
# Interior (unconstrained) adjustment
# ---------------------------------------------------------------------------


class TestInteriorAdjustment:
    def test__interior_solution_matches_lambda_times_delta(self):
        result = _run([1.0], [20.0], [10.0], [10.5], [True])

        expected_delta_tilde = (10.5 - 10.0) / 1.0
        expected_kappa_star = LAMBDA_KAPPA * expected_delta_tilde

        np.testing.assert_allclose(result.delta_tilde, [expected_delta_tilde])
        np.testing.assert_allclose(result.kappa_star_tilde, [expected_kappa_star])
        np.testing.assert_allclose(result.kappa_tilde, [expected_kappa_star])
        assert not result.upper_bound_flag[0]
        assert not result.lower_bound_flag[0]
        assert not result.inaction_flag[0]

    def test__interior_denormalized_outputs(self):
        result = _run([2.0], [40.0], [20.0], [21.0], [True])

        expected_delta_tilde = (21.0 - 20.0) / 2.0
        expected_kappa_tilde = LAMBDA_KAPPA * expected_delta_tilde
        expected_adjustment = expected_kappa_tilde * 2.0
        expected_cost = FIXED_COST_SHARE * 2.0

        np.testing.assert_allclose(result.desired_illiquid_adjustment, [expected_adjustment])
        np.testing.assert_allclose(result.adjustment_cost, [expected_cost])
        np.testing.assert_allclose(result.counterfactual_ifa_flow, [expected_adjustment])
        np.testing.assert_allclose(
            result.counterfactual_lfa_flow,
            [-expected_adjustment - expected_cost],
        )


# ---------------------------------------------------------------------------
# Upper-bound projection
# ---------------------------------------------------------------------------


class TestUpperBoundProjection:
    def test__upper_bound_binding_clips_and_flags(self):
        # post_return_lfa = 0.002 => kappa_upper = 0.002 - 0.001 = 0.001
        # delta_tilde = 1.0 => kappa_star_tilde = 0.1, well above kappa_upper.
        result = _run([1.0], [0.002], [5.0], [6.0], [True])

        kappa_upper = 0.002 - FIXED_COST_SHARE
        assert result.upper_bound_flag[0]
        assert not result.lower_bound_flag[0]
        np.testing.assert_allclose(result.kappa_star_tilde, [0.1])
        if not result.inaction_flag[0]:
            np.testing.assert_allclose(result.kappa_tilde, [kappa_upper], atol=1e-12)


# ---------------------------------------------------------------------------
# Lower-bound projection
# ---------------------------------------------------------------------------


class TestLowerBoundProjection:
    def test__lower_bound_binding_clips_and_flags(self):
        # post_return_ifa = 0.021 => kappa_lower = -(0.021 - 0.001) = -0.02
        result = _run([1.0], [20.0], [0.021], [-0.979], [True])

        kappa_lower = -(0.021 - FIXED_COST_SHARE)
        assert result.lower_bound_flag[0]
        assert not result.upper_bound_flag[0]
        np.testing.assert_allclose(result.delta_tilde, [-1.0])
        if not result.inaction_flag[0]:
            np.testing.assert_allclose(result.kappa_tilde, [kappa_lower], atol=1e-12)


# ---------------------------------------------------------------------------
# Fixed-cost inaction
# ---------------------------------------------------------------------------


class TestFixedCostInaction:
    def test__small_interior_deviation_triggers_inaction(self):
        # |delta_tilde| = 0.01, below the unconstrained threshold
        # sqrt(2*upsilon/(phi_1*lambda_kappa)) ~= 0.0632.
        result = _run([1.0], [20.0], [10.0], [10.01], [True])

        threshold = math.sqrt(2 * FIXED_COST_SHARE / (PHI_1 * LAMBDA_KAPPA))
        assert abs(result.delta_tilde[0]) < threshold
        assert result.inaction_flag[0]
        np.testing.assert_allclose(result.kappa_tilde, [0.0])
        np.testing.assert_allclose(result.adjustment_cost, [0.0])
        np.testing.assert_allclose(result.desired_illiquid_adjustment, [0.0])
        np.testing.assert_allclose(result.counterfactual_lfa_flow, [0.0])
        np.testing.assert_allclose(result.counterfactual_ifa_flow, [0.0])

    def test__zero_fixed_cost_share_never_blocks_a_nonzero_unconstrained_move(self):
        result = _run([1.0], [20.0], [10.0], [10.0001], [True], fixed_cost_share=0.0)

        assert not result.inaction_flag[0]
        assert result.kappa_tilde[0] != 0.0
        np.testing.assert_allclose(result.adjustment_cost, [0.0])


# ---------------------------------------------------------------------------
# Infeasible interval (kappa_lower > kappa_upper)
# ---------------------------------------------------------------------------


class TestInfeasibleInterval:
    def test__inverted_interval_forces_inaction_not_clip_to_upper(self):
        # post_return_lfa = post_return_ifa = 0.0003, fixed_cost_share = 0.001:
        # kappa_upper = 0.0003 - 0.001 = -0.0007
        # kappa_lower = -(0.0003 - 0.001) = +0.0007
        # kappa_lower (0.0007) > kappa_upper (-0.0007): empty feasible interval.
        # np.clip(x, 0.0007, -0.0007) would silently return -0.0007 regardless
        # of x; the household can in fact afford neither direction.
        result = _run(
            opening_tfa_scale=[1.0],
            post_return_lfa=[0.0003],
            post_return_ifa=[0.0003],
            target_illiquid_assets=[-1.0],
            portfolio_participates=[True],
        )

        kappa_upper = 0.0003 - FIXED_COST_SHARE
        kappa_lower = -(0.0003 - FIXED_COST_SHARE)
        assert kappa_lower > kappa_upper, "scenario must exercise an inverted interval"

        assert result.infeasible_interval_flag[0]
        assert result.inaction_flag[0]
        np.testing.assert_allclose(result.kappa_tilde, [0.0])
        np.testing.assert_allclose(result.adjustment_cost, [0.0])
        np.testing.assert_allclose(result.desired_illiquid_adjustment, [0.0])
        np.testing.assert_allclose(result.counterfactual_lfa_flow, [0.0])
        np.testing.assert_allclose(result.counterfactual_ifa_flow, [0.0])

    def test__infeasible_interval_flag_false_when_invalid_household(self):
        # opening_tfa_scale <= 0 takes precedence; infeasible_interval_flag
        # should not be set for an already-invalid household-period.
        result = _run([0.0], [0.0003], [0.0003], [-1.0], [True])

        assert not result.portfolio_valid_flag[0]
        assert not result.infeasible_interval_flag[0]

    def test__feasible_interval_does_not_set_the_flag(self):
        result = _run([1.0], [20.0], [10.0], [10.5], [True])

        assert not result.infeasible_interval_flag[0]


# ---------------------------------------------------------------------------
# Clipping / diagnostic flags consistency
# ---------------------------------------------------------------------------


class TestClippingDiagnosticFlags:
    def test__kappa_breve_within_bounds_when_clipped(self):
        result = _run([1.0], [0.002], [5.0], [6.0], [True])

        kappa_upper = 0.002 - FIXED_COST_SHARE
        kappa_lower = -(5.0 - FIXED_COST_SHARE)
        assert kappa_lower <= result.kappa_tilde[0] <= kappa_upper

    def test__flags_are_false_when_unconstrained(self):
        result = _run([1.0], [20.0], [10.0], [10.5], [True])

        assert not result.upper_bound_flag[0]
        assert not result.lower_bound_flag[0]


# ---------------------------------------------------------------------------
# Non-finite input fallback contract
# ---------------------------------------------------------------------------


class TestNonFiniteInputFallback:
    @pytest.mark.parametrize(
        "field",
        [
            "opening_tfa_scale",
            "post_return_lfa",
            "post_return_ifa",
            "target_illiquid_assets",
            "target_tfa_base",
        ],
    )
    def test__nan_in_any_input_zeroes_outputs_and_flags_invalid(self, field):
        values = {
            "opening_tfa_scale": [1.0],
            "post_return_lfa": [20.0],
            "post_return_ifa": [10.0],
            "target_illiquid_assets": [10.5],
            "target_tfa_base": [30.0],
        }
        values[field] = [np.nan]

        result = compute_portfolio_rebalancing(
            opening_tfa_scale=np.asarray(values["opening_tfa_scale"], dtype=float),
            post_return_lfa=np.asarray(values["post_return_lfa"], dtype=float),
            post_return_ifa=np.asarray(values["post_return_ifa"], dtype=float),
            target_illiquid_assets=np.asarray(values["target_illiquid_assets"], dtype=float),
            target_tfa_base=np.asarray(values["target_tfa_base"], dtype=float),
            portfolio_participates=np.array([True]),
            phi_1=PHI_1,
            lambda_kappa=LAMBDA_KAPPA,
            fixed_cost_share=FIXED_COST_SHARE,
        )

        assert not result.portfolio_valid_flag[0]
        np.testing.assert_allclose(result.delta_tilde, [0.0])
        np.testing.assert_allclose(result.kappa_star_tilde, [0.0])
        np.testing.assert_allclose(result.kappa_tilde, [0.0])
        np.testing.assert_allclose(result.desired_illiquid_adjustment, [0.0])
        np.testing.assert_allclose(result.adjustment_cost, [0.0])
        np.testing.assert_allclose(result.counterfactual_lfa_flow, [0.0])
        np.testing.assert_allclose(result.counterfactual_ifa_flow, [0.0])
        np.testing.assert_allclose(result.actual_illiquid_share, [0.0])
        assert not result.inaction_flag[0]
        assert not result.upper_bound_flag[0]
        assert not result.lower_bound_flag[0]

    def test__inf_in_post_return_ifa_zeroes_outputs_and_flags_invalid(self):
        result = _run([1.0], [20.0], [np.inf], [10.5], [True])

        assert not result.portfolio_valid_flag[0]
        np.testing.assert_allclose(result.kappa_tilde, [0.0])
        np.testing.assert_allclose(result.desired_illiquid_adjustment, [0.0])

    def test__non_finite_does_not_contaminate_other_households(self):
        result = _run(
            opening_tfa_scale=[1.0, 1.0],
            post_return_lfa=[20.0, 20.0],
            post_return_ifa=[10.0, np.nan],
            target_illiquid_assets=[10.5, 10.5],
            portfolio_participates=[True, True],
        )

        assert result.portfolio_valid_flag[0]
        assert not result.portfolio_valid_flag[1]
        assert result.kappa_tilde[0] != 0.0
        assert result.kappa_tilde[1] == 0.0


# ---------------------------------------------------------------------------
# Non-participants
# ---------------------------------------------------------------------------


class TestNonParticipants:
    def test__nonparticipant_with_legacy_ifa_can_only_liquidate(self):
        # Nonparticipant => target_illiquid_assets = 0 upstream (caller's job),
        # but with a legacy IFA balance to sell, delta_tilde is negative.
        result = _run([1.0], [20.0], [10.0], [0.0], [False])

        assert not result.portfolio_participates[0]
        assert result.delta_tilde[0] < 0.0
        assert result.kappa_tilde[0] <= 0.0
        assert result.counterfactual_ifa_flow[0] <= 0.0

    def test__nonparticipant_without_legacy_ifa_has_no_adjustment(self):
        # No IFA to sell, target is 0 too => delta_tilde = 0 => no adjustment.
        result = _run([1.0], [20.0], [0.0], [0.0], [False])

        np.testing.assert_allclose(result.delta_tilde, [0.0])
        np.testing.assert_allclose(result.kappa_star_tilde, [0.0])
        np.testing.assert_allclose(result.kappa_tilde, [0.0])
        assert result.inaction_flag[0]
        np.testing.assert_allclose(result.counterfactual_ifa_flow, [0.0])
        np.testing.assert_allclose(result.counterfactual_lfa_flow, [0.0])


# ---------------------------------------------------------------------------
# Closed-form threshold oracles (independent reference, paper appendix)
# ---------------------------------------------------------------------------


class TestClosedFormThresholdOracle:
    """Cross-check the generic gain > 0 implementation against the four
    explicit closed-form branches given in the wiki page, using concrete
    numeric scenarios with opening_tfa_scale = 1 so normalized quantities
    equal absolute ones, making the closed-form arithmetic easy to verify
    by hand.
    """

    def test__interior_branch_adjust(self):
        # delta_tilde = 0.5; interior threshold ~= 0.0632 => adjust.
        result = _run([1.0], [20.0], [10.0], [10.5], [True])

        threshold = math.sqrt(2 * FIXED_COST_SHARE / (PHI_1 * LAMBDA_KAPPA))
        expect_adjust = abs(result.delta_tilde[0]) > threshold

        assert expect_adjust
        assert not result.inaction_flag[0]
        np.testing.assert_allclose(result.kappa_tilde, [LAMBDA_KAPPA * result.delta_tilde[0]])

    def test__interior_branch_inaction(self):
        # delta_tilde = 0.01; below the interior threshold => inaction.
        result = _run([1.0], [20.0], [10.0], [10.01], [True])

        threshold = math.sqrt(2 * FIXED_COST_SHARE / (PHI_1 * LAMBDA_KAPPA))
        expect_adjust = abs(result.delta_tilde[0]) > threshold

        assert not expect_adjust
        assert result.inaction_flag[0]
        np.testing.assert_allclose(result.kappa_tilde, [0.0])

    def test__upper_bound_branch_adjust(self):
        # kappa_upper = 0.001, delta_tilde = 1.0.
        result = _run([1.0], [0.002], [5.0], [6.0], [True])

        kappa_upper = 0.002 - FIXED_COST_SHARE
        threshold = kappa_upper / (2 * LAMBDA_KAPPA) + FIXED_COST_SHARE / (PHI_1 * kappa_upper)
        expect_adjust = result.delta_tilde[0] > threshold

        assert expect_adjust
        assert not result.inaction_flag[0]
        np.testing.assert_allclose(result.kappa_tilde, [kappa_upper], atol=1e-12)

    def test__upper_bound_branch_inaction(self):
        # kappa_upper = 0.001, delta_tilde = 0.02 (binding but below threshold).
        result = _run([1.0], [0.002], [5.0], [5.02], [True])

        kappa_upper = 0.002 - FIXED_COST_SHARE
        threshold = kappa_upper / (2 * LAMBDA_KAPPA) + FIXED_COST_SHARE / (PHI_1 * kappa_upper)
        expect_adjust = result.delta_tilde[0] > threshold

        assert not expect_adjust
        assert result.inaction_flag[0]
        np.testing.assert_allclose(result.kappa_tilde, [0.0])

    def test__lower_bound_branch_adjust(self):
        # kappa_lower = -0.02, delta_tilde = -1.0.
        result = _run([1.0], [20.0], [0.021], [-0.979], [True])

        kappa_lower = -(0.021 - FIXED_COST_SHARE)
        threshold = kappa_lower / (2 * LAMBDA_KAPPA) + FIXED_COST_SHARE / (PHI_1 * kappa_lower)
        expect_adjust = result.delta_tilde[0] < threshold

        assert expect_adjust
        assert not result.inaction_flag[0]
        np.testing.assert_allclose(result.kappa_tilde, [kappa_lower], atol=1e-12)

    def test__lower_bound_branch_inaction(self):
        # kappa_lower = -0.001, delta_tilde = -0.02 (binding but below threshold magnitude).
        result = _run([1.0], [20.0], [0.002], [-0.018], [True])

        kappa_lower = -(0.002 - FIXED_COST_SHARE)
        threshold = kappa_lower / (2 * LAMBDA_KAPPA) + FIXED_COST_SHARE / (PHI_1 * kappa_lower)
        expect_adjust = result.delta_tilde[0] < threshold

        assert not expect_adjust
        assert result.inaction_flag[0]
        np.testing.assert_allclose(result.kappa_tilde, [0.0])


# ---------------------------------------------------------------------------
# Independent bounds oracle (derives kappa_lower/kappa_upper from raw stocks,
# not by repeating the implementation's expression)
# ---------------------------------------------------------------------------


def _independent_kappa_upper(post_return_lfa, opening_tfa_scale, fixed_cost_share):
    """Max feasible purchase share: liquid buffer net of the fixed cost.

    Written independently of the implementation's `kappa_upper` line so this
    test exercises the bounds formula itself, not just internal consistency
    of the gain-test logic downstream of it.
    """
    liquid_share = post_return_lfa / opening_tfa_scale
    return liquid_share - fixed_cost_share


def _independent_kappa_lower(post_return_ifa, opening_tfa_scale, fixed_cost_share):
    """Max feasible sale share (negative): illiquid buffer net of the fixed cost."""
    illiquid_share = post_return_ifa / opening_tfa_scale
    return -(illiquid_share - fixed_cost_share)


class TestIndependentBoundsOracle:
    @pytest.mark.parametrize(
        "opening_tfa_scale,post_return_lfa,post_return_ifa",
        [
            (1.0, 20.0, 10.0),
            (2.0, 0.5, 8.0),
            (5.0, 100.0, 0.02),
            (1.0, 0.0003, 0.0003),
        ],
    )
    def test__bounds_match_independent_formula(self, opening_tfa_scale, post_return_lfa, post_return_ifa):
        expected_upper = _independent_kappa_upper(post_return_lfa, opening_tfa_scale, FIXED_COST_SHARE)
        expected_lower = _independent_kappa_lower(post_return_ifa, opening_tfa_scale, FIXED_COST_SHARE)

        # Probe kappa_upper/kappa_lower indirectly: drive delta_tilde far
        # enough in each direction that kappa_star_tilde = lambda_kappa *
        # delta_tilde is well outside the true bound, so a binding result
        # reveals the bound's location regardless of the gain test (use
        # fixed_cost_share=0 here to remove the inaction region and isolate
        # the projection step). The offset must be scaled by 1/lambda_kappa
        # since the unconstrained solution shrinks delta_tilde by that factor.
        huge_offset = (1000.0 * opening_tfa_scale) / LAMBDA_KAPPA
        big_target = post_return_ifa + huge_offset
        result_high = _run(
            [opening_tfa_scale],
            [post_return_lfa],
            [post_return_ifa],
            [big_target],
            [True],
            fixed_cost_share=0.0,
        )
        small_target = post_return_ifa - huge_offset
        result_low = _run(
            [opening_tfa_scale],
            [post_return_lfa],
            [post_return_ifa],
            [small_target],
            [True],
            fixed_cost_share=0.0,
        )

        if expected_lower <= expected_upper:
            # Feasible interval: the projection should land exactly on the
            # independently computed bound (fixed_cost_share=0 here, so the
            # bound formula above must be evaluated with cost 0 too).
            expected_upper_zero_cost = _independent_kappa_upper(post_return_lfa, opening_tfa_scale, 0.0)
            expected_lower_zero_cost = _independent_kappa_lower(post_return_ifa, opening_tfa_scale, 0.0)
            np.testing.assert_allclose(result_high.kappa_tilde, [expected_upper_zero_cost], atol=1e-10)
            np.testing.assert_allclose(result_low.kappa_tilde, [expected_lower_zero_cost], atol=1e-10)


# ---------------------------------------------------------------------------
# Shape / dtype
# ---------------------------------------------------------------------------


class TestMultiHouseholdBehavioralMatrix:
    """A single batched call where each household independently exercises a
    different branch, to catch cross-household contamination from a
    vectorized mask applied with the wrong shape/broadcasting (e.g. a scalar
    leaking from one household's branch into another's).

    Expected values are hand-derived per household, independently of the
    single-household tests above (not copy-pasted from them), using the same
    closed-form/oracle arithmetic as those tests.
    """

    def test__eight_households_each_in_a_different_branch(self):
        # (a) interior unconstrained adjustment
        # (b) upper-bound-binding adjustment
        # (c) lower-bound-binding adjustment
        # (d) fixed-cost inaction
        # (e) opening_tfa_scale <= 0 (invalid)
        # (f) non-finite input (invalid)
        # (g) infeasible/inverted interval
        # (h) non-participant with a legacy IFA balance to liquidate
        opening_tfa_scale = [1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0]
        post_return_lfa = [20.0, 0.002, 20.0, 20.0, 20.0, np.nan, 0.0003, 20.0]
        post_return_ifa = [10.0, 5.0, 0.021, 10.0, 10.0, 10.0, 0.0003, 10.0]
        target_illiquid_assets = [10.5, 6.0, -0.979, 10.01, 10.5, 10.5, -1.0, 0.0]
        target_tfa_base = [30.0, 5.002, 20.021, 30.0, 30.0, 30.0, 0.0006, 30.0]
        portfolio_participates = [True, True, True, True, True, True, True, False]

        result = _run(
            opening_tfa_scale,
            post_return_lfa,
            post_return_ifa,
            target_illiquid_assets,
            portfolio_participates,
            target_tfa_base=target_tfa_base,
        )

        # --- per-household hand-derived expectations ---
        # (a) interior: delta=0.5, kappa_star=0.05, within bounds, gain>0.
        # (b) upper bound: kappa_upper=0.002-0.001=0.001, kappa_star=0.1 -> clipped.
        # (c) lower bound: kappa_lower=-(0.021-0.001)=-0.02, kappa_star=-0.1 -> clipped.
        # (d) inaction: |delta|=0.01 below sqrt(2*0.001/(5*0.1))~=0.0632.
        # (e) invalid: opening_tfa_scale<=0.
        # (f) invalid: non-finite post_return_lfa.
        # (g) infeasible: kappa_upper=0.0003-0.001=-0.0007, kappa_lower=+0.0007 -> inverted.
        # (h) nonparticipant liquidation: target=0, delta=-10, kappa_star=-1.0, within
        #     bounds [-9.999, 19.999], gain large positive -> full liquidation -1.0.
        expected_valid = [True, True, True, True, False, False, True, True]
        expected_no_assets = [False, False, False, False, True, False, False, False]
        expected_infeasible = [False, False, False, False, False, False, True, False]
        expected_inaction = [False, False, False, True, False, False, True, False]
        expected_upper_bound = [False, True, False, False, False, False, False, False]
        expected_lower_bound = [False, False, True, False, False, False, True, False]
        expected_kappa_tilde = [0.05, 0.001, -0.02, 0.0, 0.0, 0.0, 0.0, -1.0]
        expected_desired_adjustment = [0.05, 0.001, -0.02, 0.0, 0.0, 0.0, 0.0, -1.0]
        expected_adjustment_cost = [0.001, 0.001, 0.001, 0.0, 0.0, 0.0, 0.0, 0.001]
        expected_lfa_flow = [-0.051, -0.002, 0.019, 0.0, 0.0, 0.0, 0.0, 0.999]
        expected_ifa_flow = [0.05, 0.001, -0.02, 0.0, 0.0, 0.0, 0.0, -1.0]
        expected_delta_tilde = [0.5, 1.0, -1.0, 0.01, 0.0, 0.0, -1.0003, -10.0]
        expected_actual_share = [
            10.0 / 30.0,
            5.0 / 5.002,
            0.021 / 20.021,
            10.0 / 30.0,
            0.0,
            0.0,
            0.0003 / 0.0006,
            10.0 / 30.0,
        ]

        np.testing.assert_array_equal(result.portfolio_valid_flag, expected_valid)
        np.testing.assert_array_equal(result.no_financial_assets_flag, expected_no_assets)
        np.testing.assert_array_equal(result.infeasible_interval_flag, expected_infeasible)
        np.testing.assert_array_equal(result.inaction_flag, expected_inaction)
        np.testing.assert_array_equal(result.upper_bound_flag, expected_upper_bound)
        np.testing.assert_array_equal(result.lower_bound_flag, expected_lower_bound)
        np.testing.assert_allclose(result.kappa_tilde, expected_kappa_tilde, atol=1e-9)
        np.testing.assert_allclose(result.desired_illiquid_adjustment, expected_desired_adjustment, atol=1e-9)
        np.testing.assert_allclose(result.adjustment_cost, expected_adjustment_cost, atol=1e-9)
        np.testing.assert_allclose(result.counterfactual_lfa_flow, expected_lfa_flow, atol=1e-9)
        np.testing.assert_allclose(result.counterfactual_ifa_flow, expected_ifa_flow, atol=1e-9)
        np.testing.assert_allclose(result.delta_tilde, expected_delta_tilde, atol=1e-9)
        np.testing.assert_allclose(result.actual_illiquid_share, expected_actual_share, atol=1e-9)
        np.testing.assert_array_equal(
            result.portfolio_participates,
            [True, True, True, True, True, True, True, False],
        )


# ---------------------------------------------------------------------------
# Extreme magnitudes: very large and very small-but-nonzero values
# ---------------------------------------------------------------------------


class TestExtremeMagnitudes:
    def test__very_large_values_stay_finite_and_consistent(self):
        # opening_tfa_scale ~1e9, target_illiquid_assets ~1e12: a household
        # in a "large economy" with a sizeable target deviation. kappa_upper
        # binds because the household cannot acquire 1e12 worth of IFA in
        # one period without borrowing.
        result = _run(
            opening_tfa_scale=[1e9],
            post_return_lfa=[1.5e9],
            post_return_ifa=[3e8],
            target_illiquid_assets=[1e12],
            portfolio_participates=[True],
            target_tfa_base=[1.8e9],
        )

        for field in (
            "actual_illiquid_share",
            "delta_tilde",
            "kappa_star_tilde",
            "kappa_tilde",
            "desired_illiquid_adjustment",
            "adjustment_cost",
            "counterfactual_lfa_flow",
            "counterfactual_ifa_flow",
        ):
            array = getattr(result, field)
            assert np.all(np.isfinite(array)), f"{field} is not finite: {array}"

        assert result.portfolio_valid_flag[0]
        assert result.upper_bound_flag[0]
        assert not result.inaction_flag[0]
        # kappa_upper = post_return_lfa/scale - fixed_cost_share = 1.5 - 0.001
        expected_kappa_upper = 1.5e9 / 1e9 - FIXED_COST_SHARE
        np.testing.assert_allclose(result.kappa_tilde, [expected_kappa_upper], atol=1e-9)
        expected_adjustment = expected_kappa_upper * 1e9
        np.testing.assert_allclose(result.desired_illiquid_adjustment, [expected_adjustment], rtol=1e-9)
        np.testing.assert_allclose(result.adjustment_cost, [FIXED_COST_SHARE * 1e9], rtol=1e-9)

    def test__very_small_nonzero_values_stay_finite_and_consistent(self):
        # opening_tfa_scale ~1e-8, stocks ~1e-10/1e-8: a household with a
        # tiny but nonzero balance sheet. Normalized quantities (divided by
        # opening_tfa_scale) should remain O(1) and the de-normalized
        # currency outputs should remain finite and non-negative-cost.
        result = _run(
            opening_tfa_scale=[1e-8],
            post_return_lfa=[1.5e-8],
            post_return_ifa=[1e-10],
            target_illiquid_assets=[5e-9],
            portfolio_participates=[True],
            target_tfa_base=[1.6e-8],
        )

        for field in (
            "actual_illiquid_share",
            "delta_tilde",
            "kappa_star_tilde",
            "kappa_tilde",
            "desired_illiquid_adjustment",
            "adjustment_cost",
            "counterfactual_lfa_flow",
            "counterfactual_ifa_flow",
        ):
            array = getattr(result, field)
            assert np.all(np.isfinite(array)), f"{field} is not finite: {array}"

        assert result.portfolio_valid_flag[0]
        assert not result.no_financial_assets_flag[0]

        expected_delta_tilde = (5e-9 - 1e-10) / 1e-8
        expected_kappa_star = LAMBDA_KAPPA * expected_delta_tilde
        np.testing.assert_allclose(result.delta_tilde, [expected_delta_tilde], rtol=1e-9)
        np.testing.assert_allclose(result.kappa_star_tilde, [expected_kappa_star], rtol=1e-9)
        # Not bound-binding at this magnitude (kappa_upper/kappa_lower are
        # both large in normalized terms relative to kappa_star here).
        assert not result.upper_bound_flag[0]
        assert not result.lower_bound_flag[0]
        np.testing.assert_allclose(result.kappa_tilde, [expected_kappa_star], rtol=1e-9)

    def test__squared_loss_terms_do_not_silently_misfire_bound_flags_at_extreme_scale(self):
        # A pathologically large delta_tilde (driven by an enormous target
        # relative to a small opening_tfa_scale) pushes the quadratic loss
        # terms (delta_tilde - kappa_tilde)**2 and kappa_tilde**2 toward
        # float overflow. The implementation must not let an overflowed/NaN
        # `gain` silently produce a wrong-but-still-"valid" result; at minimum
        # outputs must stay finite, and validity/finite-input flags must not
        # be corrupted by the internal overflow.
        with np.errstate(over="ignore", invalid="ignore"):
            result = _run(
                opening_tfa_scale=[1.0],
                post_return_lfa=[1e9],
                post_return_ifa=[1e-10],
                target_illiquid_assets=[1e308],
                portfolio_participates=[True],
                target_tfa_base=[1e9],
            )

        assert result.portfolio_valid_flag[0]
        for field in (
            "kappa_tilde",
            "desired_illiquid_adjustment",
            "adjustment_cost",
            "counterfactual_lfa_flow",
            "counterfactual_ifa_flow",
        ):
            array = getattr(result, field)
            assert np.all(np.isfinite(array)), f"{field} is not finite: {array}"


class TestShapeAndDtype:
    def test__all_fields_match_input_shape(self):
        n = 4
        result = _run(
            opening_tfa_scale=[1.0, 1.0, 0.0, 1.0],
            post_return_lfa=[20.0, 20.0, 5.0, 20.0],
            post_return_ifa=[10.0, 10.0, 5.0, 10.0],
            target_illiquid_assets=[10.5, 10.01, 3.0, 10.5],
            portfolio_participates=[True, True, True, False],
        )

        for field in (
            "portfolio_participates",
            "actual_illiquid_share",
            "target_illiquid_assets",
            "delta_tilde",
            "kappa_star_tilde",
            "kappa_tilde",
            "desired_illiquid_adjustment",
            "adjustment_cost",
            "counterfactual_lfa_flow",
            "counterfactual_ifa_flow",
            "inaction_flag",
            "upper_bound_flag",
            "lower_bound_flag",
            "infeasible_interval_flag",
            "no_financial_assets_flag",
            "portfolio_valid_flag",
        ):
            array = getattr(result, field)
            assert array.shape == (n,), f"{field} has shape {array.shape}"

        assert result.kappa_tilde.dtype.kind == "f"
        assert result.inaction_flag.dtype == bool
        assert result.infeasible_interval_flag.dtype == bool
        assert result.portfolio_valid_flag.dtype == bool
