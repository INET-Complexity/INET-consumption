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
    phi_1=PHI_1,
    lambda_kappa=LAMBDA_KAPPA,
    fixed_cost_share=FIXED_COST_SHARE,
) -> PortfolioRebalancingResult:
    return compute_portfolio_rebalancing(
        opening_tfa_scale=np.asarray(opening_tfa_scale, dtype=float),
        post_return_lfa=np.asarray(post_return_lfa, dtype=float),
        post_return_ifa=np.asarray(post_return_ifa, dtype=float),
        target_illiquid_assets=np.asarray(target_illiquid_assets, dtype=float),
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
        ["opening_tfa_scale", "post_return_lfa", "post_return_ifa", "target_illiquid_assets"],
    )
    def test__nan_in_any_input_zeroes_outputs_and_flags_invalid(self, field):
        values = {
            "opening_tfa_scale": [1.0],
            "post_return_lfa": [20.0],
            "post_return_ifa": [10.0],
            "target_illiquid_assets": [10.5],
        }
        values[field] = [np.nan]

        result = compute_portfolio_rebalancing(
            opening_tfa_scale=np.asarray(values["opening_tfa_scale"], dtype=float),
            post_return_lfa=np.asarray(values["post_return_lfa"], dtype=float),
            post_return_ifa=np.asarray(values["post_return_ifa"], dtype=float),
            target_illiquid_assets=np.asarray(values["target_illiquid_assets"], dtype=float),
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
# Shape / dtype
# ---------------------------------------------------------------------------


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
            "no_financial_assets_flag",
            "portfolio_valid_flag",
        ):
            array = getattr(result, field)
            assert array.shape == (n,), f"{field} has shape {array.shape}"

        assert result.kappa_tilde.dtype.kind == "f"
        assert result.inaction_flag.dtype == bool
        assert result.portfolio_valid_flag.dtype == bool
