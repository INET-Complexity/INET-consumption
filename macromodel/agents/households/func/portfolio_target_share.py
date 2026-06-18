"""Stage 4 (portfolio choice), Increment 1: target-share contract.

Pure, vectorized helpers that compute, for each household, a
``target_illiquid_share`` and a ``portfolio_participates`` flag. These
functions do not mutate any household object and perform no I/O.

Two computation paths are provided per the architecture doc
(``knowledge-vault/wiki/architecture/consumption-stage-4-portfolio-choice.md``):

- ``compute_target_illiquid_share`` — the first-implementation contract used
  at runtime. It honours ``portfolio_composition.target_share_source`` from
  ``run_model/config/consumption_paper_parameters.yaml``: only ``"scalar"``
  is implemented in this increment, which assigns the fixed
  ``default_target_illiquid_share`` to all participating households.
- ``compute_frm_magnitude_target_share`` — the FRM covariate-driven magnitude
  equation, ``omega_star_it = Phi(x_i @ beta + q_it @ theta)``, with
  ``q_it @ theta = 0`` in this static increment (no dynamic shifters yet).
  This function is fully correct and unit-tested but is deliberately NOT
  called from ``compute_target_illiquid_share`` yet; it exists so that later
  increments (``target_share_source: precomputed`` / ``grouped``) can swap it
  in without new plumbing.

Both paths clip output shares to ``[0, 1]`` and report an explicit
``target_share_clipped_flag`` (clipping must be diagnostic, not silent, per
the doc's invariants). Nonparticipants always get ``target_illiquid_share =
0.0`` regardless of source.

This module implements only the "target-share contract" piece of
Increment 1. It does not implement adjustment costs, inaction bands, or
``Delta_tilde`` — those belong to Increment 2's rebalancing helper.
"""

import numpy as np
from scipy.stats import norm

_VALID_TARGET_SHARE_SOURCES = ("scalar", "frm_magnitude")


def validate_target_share_source(target_share_source: str) -> None:
    """Validate ``target_share_source`` at config-load time.

    Per the Stage 4 architecture doc (Increment 5), ``target_share_source``
    is a closed set of named runtime paths. Validating once at config load
    (here, called from ``PaperAssetReturnWealthSetter.__init__``) avoids
    re-checking the string every period inside the pure dispatch function
    below, mirroring the existing ``lambda_kappa``/``fixed_cost_share``
    config-time validation pattern.

    Args:
        target_share_source (str): The configured target-share source.

    Raises:
        ValueError: If ``target_share_source`` is not one of
            ``_VALID_TARGET_SHARE_SOURCES``.
    """
    if target_share_source not in _VALID_TARGET_SHARE_SOURCES:
        raise ValueError(
            f"target_share_source must be one of {_VALID_TARGET_SHARE_SOURCES!r}; got {target_share_source!r}."
        )


def compute_target_illiquid_share(
    portfolio_participates: np.ndarray,
    target_share_source: str = "scalar",
    default_target_illiquid_share: float = 0.65,
    frm_covariates: dict[str, np.ndarray] | None = None,
    frm_magnitude_coefficients: dict[str, float] | None = None,
    population_scale_factor: float | None = None,
    net_wealth_scale_divisor: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the runtime target illiquid share per household.

    Implements the ``target_share_source`` contract from
    ``portfolio_composition`` in ``consumption_paper_parameters.yaml``.
    ``target_share_source: scalar`` remains the default for every country:
    every participating household is assigned the fixed
    ``default_target_illiquid_share``, and every nonparticipant is assigned
    ``0.0``. ``target_share_source: frm_magnitude`` (Increment 5) is an
    opt-in alternative, switched on deliberately per country, that instead
    calls ``compute_frm_magnitude_target_share`` with household covariates
    and the loaded FRM coefficients. This increment does not change the
    default: a country config that does not set ``target_share_source``
    explicitly continues to run the ``"scalar"`` path unchanged.

    Args:
        portfolio_participates (np.ndarray): Boolean array, one entry per
            household, ``True`` for households with ``initial_IFA > 0``.
        target_share_source (str): Selects the target-share rule. One of
            ``"scalar"`` (default) or ``"frm_magnitude"``.
        default_target_illiquid_share (float): Fixed scalar target share
            assigned to all participating households under the ``"scalar"``
            source. Ignored under ``"frm_magnitude"``.
        frm_covariates (dict[str, np.ndarray] | None): Required only when
            ``target_share_source="frm_magnitude"``. Keys: ``"age"``,
            ``"household_members_in_employment"``,
            ``"investment_attitudes"``, ``"mortgagor"``, ``"owner"``,
            ``"net_wealth"`` — forwarded to
            ``compute_frm_magnitude_target_share``.
        frm_magnitude_coefficients (dict[str, float] | None): Required only
            when ``target_share_source="frm_magnitude"``. The loaded
            ``FRMCoefficients.magnitude_coefficients``.
        population_scale_factor (float | None): Required only when
            ``target_share_source="frm_magnitude"``.
        net_wealth_scale_divisor (float | None): Required only when
            ``target_share_source="frm_magnitude"``.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(target_illiquid_share,
        target_share_clipped_flag)``, both shape ``(n_households,)``.
        ``target_illiquid_share`` is clipped to ``[0, 1]``;
        ``target_share_clipped_flag`` is ``True`` wherever clipping changed
        the value.

    Raises:
        ValueError: If ``target_share_source`` is not one of
            ``_VALID_TARGET_SHARE_SOURCES``, or if ``"frm_magnitude"`` is
            selected but required covariates/coefficients/scale arguments
            are missing.
    """
    validate_target_share_source(target_share_source)

    portfolio_participates = np.asarray(portfolio_participates, dtype=bool)

    if target_share_source == "scalar":
        raw_share = np.where(portfolio_participates, default_target_illiquid_share, 0.0)
        target_illiquid_share = np.clip(raw_share, 0.0, 1.0)
        target_share_clipped_flag = ~np.isclose(raw_share, target_illiquid_share)
        return target_illiquid_share, target_share_clipped_flag

    # target_share_source == "frm_magnitude"
    missing = [
        name
        for name, value in (
            ("frm_covariates", frm_covariates),
            ("frm_magnitude_coefficients", frm_magnitude_coefficients),
            ("population_scale_factor", population_scale_factor),
            ("net_wealth_scale_divisor", net_wealth_scale_divisor),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "target_share_source='frm_magnitude' requires "
            f"{', '.join(missing)} to be provided; got None for each of those."
        )

    required_covariate_keys = (
        "age",
        "household_members_in_employment",
        "investment_attitudes",
        "mortgagor",
        "owner",
        "net_wealth",
    )
    missing_covariates = [key for key in required_covariate_keys if key not in frm_covariates]
    if missing_covariates:
        raise ValueError(f"frm_covariates is missing required keys: {missing_covariates!r}.")

    return compute_frm_magnitude_target_share(
        age=frm_covariates["age"],
        household_members_in_employment=frm_covariates["household_members_in_employment"],
        investment_attitudes=frm_covariates["investment_attitudes"],
        mortgagor=frm_covariates["mortgagor"],
        owner=frm_covariates["owner"],
        net_wealth=frm_covariates["net_wealth"],
        portfolio_participates=portfolio_participates,
        magnitude_coefficients=frm_magnitude_coefficients,
        population_scale_factor=population_scale_factor,
        net_wealth_scale_divisor=net_wealth_scale_divisor,
    )


def compute_frm_magnitude_target_share(
    age: np.ndarray,
    household_members_in_employment: np.ndarray,
    investment_attitudes: np.ndarray,
    mortgagor: np.ndarray,
    owner: np.ndarray,
    net_wealth: np.ndarray,
    portfolio_participates: np.ndarray,
    magnitude_coefficients: dict[str, float],
    population_scale_factor: float,
    net_wealth_scale_divisor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the FRM covariate-driven magnitude (intensive-margin) target share.

    Implements ``omega_star_it = Phi(x_i @ beta + q_it @ theta)`` using the
    GTP-FRM magnitude-equation coefficients (see
    ``macro_data/readers/portfolio_frm.py::FRMCoefficients.magnitude_coefficients``)
    and the standard normal CDF for ``Phi``. The dynamic-shifter term
    ``q_it @ theta`` is fixed at ``0`` in this static increment (per the doc's
    "minimal static implementation sets q_it theta = 0").

    This function is deliberately NOT wired into
    ``compute_target_illiquid_share`` yet (``target_share_source: scalar`` in
    this increment); it is built and tested in isolation so a later increment
    can switch ``target_share_source`` to ``precomputed``/``grouped`` and call
    this function without new plumbing.

    Per the doc's scale convention, ``net_wealth`` is a model-scale quantity
    (inflated by the model's population scale factor ``S``, e.g. ``S=5000``
    in ``run_model/config/data_config_FRA.yaml``). It must first be divided by
    ``population_scale_factor`` to recover per-household HFCS-comparable
    units, then by ``net_wealth_scale_divisor`` (``100000.0``) to match the
    FRM estimation scale. Non-financial covariates (age, employment count,
    investment attitudes, owner/mortgagor flags) are unaffected by ``S``.

    Args:
        age (np.ndarray): Household head's age, one entry per household.
        household_members_in_employment (np.ndarray): Count of employed
            household members.
        investment_attitudes (np.ndarray): Household financial risk attitude
            (HFCS ``HD1800`` ordinal scale).
        mortgagor (np.ndarray): Boolean/0-1 array, ``True``/``1`` where
            ``Tenure Status of the Main Residence == 2``.
        owner (np.ndarray): Boolean/0-1 array, ``True``/``1`` where
            ``Tenure Status of the Main Residence == 1``.
        net_wealth (np.ndarray): Model-scale net wealth (``wealth - debt``,
            already inflated by ``population_scale_factor``).
        portfolio_participates (np.ndarray): Boolean array; nonparticipants
            are forced to ``omega_star = 0`` regardless of covariates, per
            the doc ("Households with initial IFA <= 0 ... map to omega_star
            = 0").
        magnitude_coefficients (dict[str, float]): FRM magnitude-equation
            coefficients, keyed by covariate name plus ``"constant"`` (see
            ``FRMCoefficients.magnitude_coefficients``).
        population_scale_factor (float): The model's population scale factor
            ``S`` used to inflate financial-asset state arrays (e.g.
            ``data_config_FRA.yaml``'s ``scale: 5000``). Passed explicitly
            rather than hardcoded or looked up, since no single clean
            accessor exists in this codebase for it yet.
        net_wealth_scale_divisor (float): FRM estimation-scale divisor for
            net wealth (``FRMCoefficients.net_wealth_scale_divisor``,
            ``100000.0``).

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(target_illiquid_share,
        target_share_clipped_flag)``, both shape ``(n_households,)``. Because
        ``Phi`` (the standard normal CDF) is bounded to the open interval
        ``(0, 1)`` by construction, the clip to ``[0, 1]`` can never actually
        change a finite ``Phi`` output; ``target_share_clipped_flag`` is
        included for interface symmetry with ``compute_target_illiquid_share``
        and to catch non-finite inputs (e.g. NaN covariates), which
        ``np.clip`` would otherwise pass through silently.
    """
    age = np.asarray(age, dtype=float)
    household_members_in_employment = np.asarray(household_members_in_employment, dtype=float)
    investment_attitudes = np.asarray(investment_attitudes, dtype=float)
    mortgagor = np.asarray(mortgagor, dtype=float)
    owner = np.asarray(owner, dtype=float)
    net_wealth = np.asarray(net_wealth, dtype=float)
    portfolio_participates = np.asarray(portfolio_participates, dtype=bool)

    scaled_net_wealth = net_wealth / (population_scale_factor * net_wealth_scale_divisor)

    linear_index = (
        magnitude_coefficients["constant"]
        + magnitude_coefficients["age"] * age
        + magnitude_coefficients["household_members_in_employment"] * household_members_in_employment
        + magnitude_coefficients["investment_attitudes"] * investment_attitudes
        + magnitude_coefficients["mortgagor"] * mortgagor
        + magnitude_coefficients["owner"] * owner
        + magnitude_coefficients["net_wealth"] * scaled_net_wealth
    )

    omega_star = norm.cdf(linear_index)
    omega_star = np.where(portfolio_participates, omega_star, 0.0)

    target_illiquid_share = np.clip(omega_star, 0.0, 1.0)
    target_share_clipped_flag = ~np.isclose(omega_star, target_illiquid_share, equal_nan=False) | ~np.isfinite(
        omega_star
    )
    # Non-finite inputs (e.g. NaN covariates) propagate to a NaN omega_star, which
    # np.clip would pass through unchanged (NaN is not < 0 or > 1) and silently.
    # Flag those explicitly per the "clipping should be diagnostic, not silent" invariant.
    target_illiquid_share = np.where(np.isfinite(omega_star), target_illiquid_share, 0.0)

    return target_illiquid_share, target_share_clipped_flag
