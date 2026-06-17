import numpy as np
import pytest
from scipy.stats import norm

from macromodel.agents.households.func.portfolio_target_share import (
    compute_frm_magnitude_target_share,
    compute_target_illiquid_share,
)

# ---------------------------------------------------------------------------
# compute_target_illiquid_share (scalar path)
# ---------------------------------------------------------------------------


def test__scalar_path_participants_get_exact_default_share():
    participates = np.array([True, True, False, True])

    target_share, clipped_flag = compute_target_illiquid_share(
        portfolio_participates=participates,
        target_share_source="scalar",
        default_target_illiquid_share=0.65,
    )

    np.testing.assert_allclose(target_share[participates], 0.65)
    assert not clipped_flag.any()


def test__scalar_path_nonparticipants_get_zero():
    participates = np.array([True, False, False])

    target_share, _ = compute_target_illiquid_share(
        portfolio_participates=participates,
        target_share_source="scalar",
        default_target_illiquid_share=0.65,
    )

    np.testing.assert_allclose(target_share[~participates], 0.0)


def test__scalar_path_unsupported_source_raises():
    with pytest.raises(ValueError):
        compute_target_illiquid_share(
            portfolio_participates=np.array([True]),
            target_share_source="precomputed",
        )


def test__scalar_path_shape_and_dtype_match_input():
    participates = np.array([True, False, True, True, False])

    target_share, clipped_flag = compute_target_illiquid_share(participates)

    assert target_share.shape == participates.shape
    assert clipped_flag.shape == participates.shape
    assert target_share.dtype.kind == "f"
    assert clipped_flag.dtype == bool


# ---------------------------------------------------------------------------
# compute_frm_magnitude_target_share (FRM covariate path, inert in this increment)
# ---------------------------------------------------------------------------

_MAGNITUDE_COEFFICIENTS = {
    "constant": 0.0418,
    "age": 0.0101,
    "household_members_in_employment": -0.1126,
    "investment_attitudes": -0.141,
    "mortgagor": -0.1403,
    "owner": -0.1749,
    "net_wealth": 0.0345,
}


def test__frm_path_matches_hand_calculated_phi_value():
    age = np.array([40.0])
    household_members_in_employment = np.array([1.0])
    investment_attitudes = np.array([2.0])
    mortgagor = np.array([0.0])
    owner = np.array([1.0])
    net_wealth = np.array([100_000.0])  # model-scale net wealth
    participates = np.array([True])
    population_scale_factor = 5000.0
    net_wealth_scale_divisor = 100_000.0

    target_share, clipped_flag = compute_frm_magnitude_target_share(
        age=age,
        household_members_in_employment=household_members_in_employment,
        investment_attitudes=investment_attitudes,
        mortgagor=mortgagor,
        owner=owner,
        net_wealth=net_wealth,
        portfolio_participates=participates,
        magnitude_coefficients=_MAGNITUDE_COEFFICIENTS,
        population_scale_factor=population_scale_factor,
        net_wealth_scale_divisor=net_wealth_scale_divisor,
    )

    scaled_net_wealth = 100_000.0 / (5000.0 * 100_000.0)
    expected_linear_index = (
        0.0418
        + 0.0101 * 40.0
        + (-0.1126) * 1.0
        + (-0.141) * 2.0
        + (-0.1403) * 0.0
        + (-0.1749) * 1.0
        + 0.0345 * scaled_net_wealth
    )
    expected_share = norm.cdf(expected_linear_index)

    np.testing.assert_allclose(target_share, [expected_share])
    assert not clipped_flag.any()


def test__frm_path_with_zero_covariates_matches_phi_of_constant():
    n = 3
    zeros = np.zeros(n)
    participates = np.full(n, True)

    target_share, clipped_flag = compute_frm_magnitude_target_share(
        age=zeros,
        household_members_in_employment=zeros,
        investment_attitudes=zeros,
        mortgagor=zeros,
        owner=zeros,
        net_wealth=zeros,
        portfolio_participates=participates,
        magnitude_coefficients=_MAGNITUDE_COEFFICIENTS,
        population_scale_factor=5000.0,
        net_wealth_scale_divisor=100_000.0,
    )

    expected = norm.cdf(_MAGNITUDE_COEFFICIENTS["constant"])
    np.testing.assert_allclose(target_share, np.full(n, expected))
    assert not clipped_flag.any()


def test__frm_path_clipping_cannot_trigger_for_finite_inputs():
    """Phi (the standard normal CDF) is bounded to the open interval (0, 1) by
    construction, so for any finite linear index, np.clip to [0, 1] is a no-op.
    This test documents and confirms that fact explicitly rather than writing a
    vacuous "clipping happened" test: even an extreme linear index (driven by a
    very large net_wealth covariate) produces a Phi value strictly inside [0, 1],
    so target_share_clipped_flag must be False here.
    """
    extreme_net_wealth = np.array([1e15])
    participates = np.array([True])

    target_share, clipped_flag = compute_frm_magnitude_target_share(
        age=np.array([40.0]),
        household_members_in_employment=np.array([1.0]),
        investment_attitudes=np.array([2.0]),
        mortgagor=np.array([0.0]),
        owner=np.array([1.0]),
        net_wealth=extreme_net_wealth,
        portfolio_participates=participates,
        magnitude_coefficients=_MAGNITUDE_COEFFICIENTS,
        population_scale_factor=5000.0,
        net_wealth_scale_divisor=100_000.0,
    )

    assert 0.0 <= target_share[0] <= 1.0
    assert not clipped_flag.any()


def test__frm_path_flags_non_finite_input_instead_of_silently_clipping():
    nan_net_wealth = np.array([np.nan])
    participates = np.array([True])

    target_share, clipped_flag = compute_frm_magnitude_target_share(
        age=np.array([40.0]),
        household_members_in_employment=np.array([1.0]),
        investment_attitudes=np.array([2.0]),
        mortgagor=np.array([0.0]),
        owner=np.array([1.0]),
        net_wealth=nan_net_wealth,
        portfolio_participates=participates,
        magnitude_coefficients=_MAGNITUDE_COEFFICIENTS,
        population_scale_factor=5000.0,
        net_wealth_scale_divisor=100_000.0,
    )

    assert clipped_flag[0]
    assert target_share[0] == 0.0


def test__frm_path_nonparticipant_override_forces_zero_even_with_high_covariate_share():
    # Covariates chosen so the "as-if-participating" Phi value would be well
    # above zero (high age, owner=0, mortgagor=0, low investment_attitudes).
    age = np.array([90.0])
    household_members_in_employment = np.array([0.0])
    investment_attitudes = np.array([0.0])
    mortgagor = np.array([0.0])
    owner = np.array([0.0])
    net_wealth = np.array([0.0])
    participates = np.array([False])

    target_share, _ = compute_frm_magnitude_target_share(
        age=age,
        household_members_in_employment=household_members_in_employment,
        investment_attitudes=investment_attitudes,
        mortgagor=mortgagor,
        owner=owner,
        net_wealth=net_wealth,
        portfolio_participates=participates,
        magnitude_coefficients=_MAGNITUDE_COEFFICIENTS,
        population_scale_factor=5000.0,
        net_wealth_scale_divisor=100_000.0,
    )

    # Sanity check: the same covariates with participation=True would be > 0.
    target_share_if_participating, _ = compute_frm_magnitude_target_share(
        age=age,
        household_members_in_employment=household_members_in_employment,
        investment_attitudes=investment_attitudes,
        mortgagor=mortgagor,
        owner=owner,
        net_wealth=net_wealth,
        portfolio_participates=np.array([True]),
        magnitude_coefficients=_MAGNITUDE_COEFFICIENTS,
        population_scale_factor=5000.0,
        net_wealth_scale_divisor=100_000.0,
    )

    assert target_share_if_participating[0] > 0.0
    assert target_share[0] == 0.0


def test__frm_path_shape_and_dtype_match_input():
    n = 5
    zeros = np.zeros(n)
    participates = np.full(n, True)

    target_share, clipped_flag = compute_frm_magnitude_target_share(
        age=zeros,
        household_members_in_employment=zeros,
        investment_attitudes=zeros,
        mortgagor=zeros,
        owner=zeros,
        net_wealth=zeros,
        portfolio_participates=participates,
        magnitude_coefficients=_MAGNITUDE_COEFFICIENTS,
        population_scale_factor=5000.0,
        net_wealth_scale_divisor=100_000.0,
    )

    assert target_share.shape == (n,)
    assert clipped_flag.shape == (n,)
    assert target_share.dtype.kind == "f"
    assert clipped_flag.dtype == bool
