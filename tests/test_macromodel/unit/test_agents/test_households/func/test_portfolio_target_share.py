import numpy as np
import pytest
from scipy.stats import norm

from macromodel.agents.households.func.portfolio_target_share import (
    compute_frm_magnitude_target_share,
    compute_household_head_covariates,
    compute_target_illiquid_share,
    validate_target_share_source,
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


def test__frm_path_missing_coefficient_key_fails_clearly():
    """A typo'd or incomplete magnitude_coefficients dict must raise a named
    ValueError listing the missing key(s), not an opaque KeyError from deep
    inside the linear-index computation."""
    incomplete_coefficients = dict(_MAGNITUDE_COEFFICIENTS)
    del incomplete_coefficients["net_wealth"]
    n = 2
    participates = np.full(n, True)
    zeros = np.zeros(n)

    with pytest.raises(ValueError, match="net_wealth"):
        compute_frm_magnitude_target_share(
            age=zeros,
            household_members_in_employment=zeros,
            investment_attitudes=zeros,
            mortgagor=zeros,
            owner=zeros,
            net_wealth=zeros,
            portfolio_participates=participates,
            magnitude_coefficients=incomplete_coefficients,
            population_scale_factor=5000.0,
            net_wealth_scale_divisor=100_000.0,
        )


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


# ---------------------------------------------------------------------------
# validate_target_share_source (config-load-time validation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("valid_source", ["scalar", "frm_magnitude"])
def test__validate_target_share_source_accepts_known_sources(valid_source):
    validate_target_share_source(valid_source)  # must not raise


def test__validate_target_share_source_rejects_unknown_source():
    with pytest.raises(ValueError, match="target_share_source"):
        validate_target_share_source("precomputed")


# ---------------------------------------------------------------------------
# compute_target_illiquid_share dispatch (Increment 5: frm_magnitude opt-in)
# ---------------------------------------------------------------------------


def _frm_kwargs(n: int, participates: np.ndarray) -> dict:
    return dict(
        portfolio_participates=participates,
        target_share_source="frm_magnitude",
        frm_covariates={
            "age": np.full(n, 40.0),
            "household_members_in_employment": np.full(n, 1.0),
            "investment_attitudes": np.full(n, 2.0),
            "mortgagor": np.zeros(n),
            "owner": np.ones(n),
            "net_wealth": np.full(n, 100_000.0),
        },
        frm_magnitude_coefficients=_MAGNITUDE_COEFFICIENTS,
        population_scale_factor=5000.0,
        net_wealth_scale_divisor=100_000.0,
    )


def test__dispatch_scalar_path_default_is_unchanged_by_frm_addition():
    """The scalar default (this increment's user-mandated invariant) must not
    move: calling with target_share_source defaulted/explicitly "scalar"
    produces exactly the same result as before frm_magnitude existed."""
    participates = np.array([True, True, False, True])

    target_share, clipped_flag = compute_target_illiquid_share(
        portfolio_participates=participates,
        target_share_source="scalar",
        default_target_illiquid_share=0.65,
    )

    np.testing.assert_allclose(target_share[participates], 0.65)
    np.testing.assert_allclose(target_share[~participates], 0.0)
    assert not clipped_flag.any()


def test__dispatch_frm_magnitude_matches_direct_helper_call():
    n = 3
    participates = np.full(n, True)
    kwargs = _frm_kwargs(n, participates)

    dispatched_share, dispatched_flag = compute_target_illiquid_share(**kwargs)
    direct_share, direct_flag = compute_frm_magnitude_target_share(
        age=kwargs["frm_covariates"]["age"],
        household_members_in_employment=kwargs["frm_covariates"]["household_members_in_employment"],
        investment_attitudes=kwargs["frm_covariates"]["investment_attitudes"],
        mortgagor=kwargs["frm_covariates"]["mortgagor"],
        owner=kwargs["frm_covariates"]["owner"],
        net_wealth=kwargs["frm_covariates"]["net_wealth"],
        portfolio_participates=participates,
        magnitude_coefficients=_MAGNITUDE_COEFFICIENTS,
        population_scale_factor=5000.0,
        net_wealth_scale_divisor=100_000.0,
    )

    np.testing.assert_allclose(dispatched_share, direct_share)
    np.testing.assert_array_equal(dispatched_flag, direct_flag)


def test__dispatch_frm_magnitude_nonparticipants_forced_to_zero():
    n = 2
    participates = np.array([True, False])
    kwargs = _frm_kwargs(n, participates)

    target_share, _ = compute_target_illiquid_share(**kwargs)

    assert target_share[1] == 0.0
    assert target_share[0] > 0.0


@pytest.mark.parametrize(
    "missing_key",
    ["frm_covariates", "frm_magnitude_coefficients", "population_scale_factor", "net_wealth_scale_divisor"],
)
def test__dispatch_frm_magnitude_missing_required_argument_fails_clearly(missing_key):
    n = 2
    participates = np.full(n, True)
    kwargs = _frm_kwargs(n, participates)
    kwargs[missing_key] = None

    with pytest.raises(ValueError, match="frm_magnitude"):
        compute_target_illiquid_share(**kwargs)


def test__dispatch_frm_magnitude_missing_covariate_key_fails_clearly():
    n = 2
    participates = np.full(n, True)
    kwargs = _frm_kwargs(n, participates)
    del kwargs["frm_covariates"]["net_wealth"]

    with pytest.raises(ValueError, match="net_wealth"):
        compute_target_illiquid_share(**kwargs)


def test__dispatch_unsupported_source_still_raises():
    """frm_magnitude must extend, not loosen, the closed set of valid sources."""
    with pytest.raises(ValueError, match="target_share_source"):
        compute_target_illiquid_share(
            portfolio_participates=np.array([True]),
            target_share_source="grouped",
        )


# ---------------------------------------------------------------------------
# compute_household_head_covariates (Increment 5: household-head aggregation)
#
# Hand-built individual population (5 individuals, 3 households) with
# expected values computed by inspection, independent of the production
# selection algorithm — this is the genuinely independent oracle that
# test_households.py's real-FRA-data integration check intentionally does
# not re-derive.
#
# Individual:        0     1     2     3     4
# Age:               70    45    8     30    60
# Employed:          F     T     F     T     T
# Reference person:  F     T     F     F     T
#
# Household 0 -> members [0, 1, 2]: individual 1 is the flagged reference
#   person -> head_age = 45 (not 70, the oldest); employed count = 1 (only
#   individual 1).
# Household 1 -> members [3]: single member, not flagged reference person
#   -> no flag in this household -> oldest-member fallback -> head_age = 30
#   (the only member); employed count = 1.
# Household 2 -> members [2, 4]: individual 4 is the flagged reference
#   person -> head_age = 60; employed count = 1 (only individual 4).
# ---------------------------------------------------------------------------

_HEAD_TEST_AGES = np.array([70.0, 45.0, 8.0, 30.0, 60.0])
_HEAD_TEST_IS_EMPLOYED = np.array([False, True, False, True, True])
_HEAD_TEST_IS_REFERENCE_PERSON = np.array([False, True, False, False, True])
_HEAD_TEST_CORR_INDIVIDUALS = [
    np.array([0, 1, 2]),
    np.array([3]),
    np.array([2, 4]),
]


def test__household_head_covariates_uses_flagged_reference_person_not_oldest():
    head_age, employed_count = compute_household_head_covariates(
        corr_individuals=_HEAD_TEST_CORR_INDIVIDUALS,
        individual_ages=_HEAD_TEST_AGES,
        individual_is_employed=_HEAD_TEST_IS_EMPLOYED,
        individual_is_reference_person=_HEAD_TEST_IS_REFERENCE_PERSON,
    )

    # Household 0: head is individual 1 (age 45), the flagged reference
    # person, not individual 0 (age 70, the oldest) — this is the case that
    # distinguishes this rule from a pure oldest-member proxy.
    assert head_age[0] == 45.0
    assert employed_count[0] == 1.0


def test__household_head_covariates_falls_back_to_oldest_when_no_reference_person_flagged():
    head_age, employed_count = compute_household_head_covariates(
        corr_individuals=_HEAD_TEST_CORR_INDIVIDUALS,
        individual_ages=_HEAD_TEST_AGES,
        individual_is_employed=_HEAD_TEST_IS_EMPLOYED,
        individual_is_reference_person=_HEAD_TEST_IS_REFERENCE_PERSON,
    )

    # Household 1: single member (individual 3), not flagged as reference
    # person -> falls back to oldest-member rule, which trivially selects
    # the only member.
    assert head_age[1] == 30.0
    assert employed_count[1] == 1.0


def test__household_head_covariates_handles_multi_member_household_with_flag():
    head_age, employed_count = compute_household_head_covariates(
        corr_individuals=_HEAD_TEST_CORR_INDIVIDUALS,
        individual_ages=_HEAD_TEST_AGES,
        individual_is_employed=_HEAD_TEST_IS_EMPLOYED,
        individual_is_reference_person=_HEAD_TEST_IS_REFERENCE_PERSON,
    )

    # Household 2: head is individual 4 (age 60, flagged reference person),
    # not individual 2 (age 8, the other member).
    assert head_age[2] == 60.0
    assert employed_count[2] == 1.0


def test__household_head_covariates_fallback_with_no_flag_among_multiple_members():
    # A genuinely independent case beyond the three fixture households above:
    # two members, neither flagged as reference person (e.g. both NaN on
    # RA0100) -> must fall back to the oldest of the two, not the first.
    corr_individuals = [np.array([0, 4])]  # ages 70 and 60; neither flagged
    head_age, employed_count = compute_household_head_covariates(
        corr_individuals=corr_individuals,
        individual_ages=_HEAD_TEST_AGES,
        individual_is_employed=_HEAD_TEST_IS_EMPLOYED,
        individual_is_reference_person=np.array([False, True, False, False, False]),
    )

    assert head_age[0] == 70.0  # oldest of {individual 0 (70), individual 4 (60)}
    assert employed_count[0] == 1.0  # only individual 4 is employed


def test__household_head_covariates_shape_and_dtype():
    head_age, employed_count = compute_household_head_covariates(
        corr_individuals=_HEAD_TEST_CORR_INDIVIDUALS,
        individual_ages=_HEAD_TEST_AGES,
        individual_is_employed=_HEAD_TEST_IS_EMPLOYED,
        individual_is_reference_person=_HEAD_TEST_IS_REFERENCE_PERSON,
    )

    assert head_age.shape == (3,)
    assert employed_count.shape == (3,)
    assert head_age.dtype.kind == "f"
    assert employed_count.dtype.kind == "f"
