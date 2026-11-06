"""Pure response arithmetic on explicitly aligned nominal household outcome paths."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .two_pass_diagnostics import _freeze


def _finite(value, name, shape=None):
    array = np.asarray(value, dtype=float)
    if (shape is not None and array.shape != shape) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have the declared dimensions and finite values.")
    return array


def _ids(values):
    result = tuple(values)
    if not result or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (str, int, np.integer)) for x in result):
        raise ValueError("Household IDs must be nonempty strings/integers.")
    if len(set(result)) != len(result):
        raise ValueError("Household IDs must be unique.")
    return result


@dataclass(frozen=True)
class PairedResponse:
    outcome: str
    response_kind: str
    household_ids: tuple
    periods: tuple
    nominal_difference: np.ndarray
    real_difference: np.ndarray
    household_denominator_real: np.ndarray
    summaries: Mapping


def measure_paired_response(
    *,
    outcome,
    intervention_kind,
    value_units,
    control_nominal,
    shock_nominal,
    control_ids,
    shock_ids,
    control_periods,
    shock_periods,
    control_deflators,
    shock_deflators,
    intervention_nominal,
    pre_shock_control_deflator,
    horizons=(1, 4, 8),
    groups=None,
):
    """Measure impact/cumulative spending per one-time unit injected, without filtering.

    Inputs are T x H native nominal levels for ONE named outcome and T-vectors
    of path-specific C/C0 (not raw base-100 CPI). Rows start at the impact quarter;
    no initial-state row is included. Amounts already carry model population scale.
    No VAT/rent/financing inference, survey reweighting, ratio clipping or .14
    enforcement occurs. Supply each native outcome separately; real-valued inputs
    are rejected by their required units tag to avoid accidental double deflation.

    Groups are predeclared name -> boolean H-mask from common pre-shock inputs.
    They can overlap; they never replace the primary 'all' cohort. Empty groups
    and zero injections return explicit NA/availability metadata. Household
    quantiles cover defined individual ratios only, with counts disclosed; the
    aggregate always includes every group's household numerator exactly once.
    """
    if not isinstance(outcome, str) or not outcome.strip():
        raise ValueError("outcome must explicitly name the measured native series.")
    if intervention_kind not in ("income_transfer", "lfa_endowment") or value_units != "nominal":
        raise ValueError("Declare income_transfer/lfa_endowment and nominal value_units.")
    ids = _ids(control_ids)
    if _ids(shock_ids) != ids:
        raise ValueError("Control/shock household IDs and ordering must match exactly.")
    periods = np.asarray(control_periods)
    other_periods = np.asarray(shock_periods)
    if (
        periods.ndim != 1
        or periods.size == 0
        or periods.dtype.kind not in "iu"
        or np.any(periods < 0)
        or np.any(np.diff(periods) != 1)
        or not np.array_equal(periods, other_periods)
        or other_periods.dtype.kind not in "iu"
    ):
        raise ValueError("Control/shock periods must be identical consecutive integer rows.")
    shape = (periods.size, len(ids))
    control = _finite(control_nominal, "control_nominal", shape)
    shock = _finite(shock_nominal, "shock_nominal", shape)
    cd = _finite(control_deflators, "control_deflators", (shape[0],))
    sd = _finite(shock_deflators, "shock_deflators", (shape[0],))
    pre = _finite(pre_shock_control_deflator, "pre_shock_control_deflator", ())
    if np.any(cd <= 0) or np.any(sd <= 0) or pre <= 0:
        raise ValueError("All declared deflators must be positive.")
    amounts = _finite(intervention_nominal, "intervention_nominal", (shape[1],))
    if np.any(amounts < 0):
        raise ValueError("Intervention amounts must be nonnegative.")
    horizons = tuple(horizons)
    if (
        not horizons
        or len(set(horizons)) != len(horizons)
        or any(
            isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, np.integer)) or h < 1 or h > shape[0]
            for h in horizons
        )
    ):
        raise ValueError("Horizons must be unique positive integers within the complete supplied path.")
    if groups is not None and not isinstance(groups, Mapping):
        raise ValueError("groups must map names to boolean household masks.")
    masks = {"all": np.ones(len(ids), dtype=bool)}
    for name, mask in ({} if groups is None else groups).items():
        mask = np.asarray(mask)
        if (
            not isinstance(name, str)
            or not name
            or name == "all"
            or mask.dtype.kind != "b"
            or mask.shape != (len(ids),)
        ):
            raise ValueError("Groups require distinct names and boolean household masks; 'all' is reserved.")
        masks[name] = mask
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        nominal_difference = shock - control
        real_difference = shock / sd[:, None] - control / cd[:, None]
        denominator_real = amounts / pre
    for name, array in (
        ("nominal_difference", nominal_difference),
        ("real_difference", real_difference),
        ("household_denominator_real", denominator_real),
    ):
        _finite(array, name)
    if np.any((amounts > 0) & (denominator_real == 0)):
        raise ValueError("Positive intervention underflows in pre-shock real units.")
    summaries = {}
    for horizon in horizons:
        by_group = {}
        for name, mask in masks.items():
            with np.errstate(over="ignore", invalid="ignore"):
                nominal_numerator = nominal_difference[:horizon, mask].sum()
                real_numerator = real_difference[:horizon, mask].sum()
                nominal_denominator = amounts[mask].sum()
                real_denominator = denominator_real[mask].sum()
                household_numerators = real_difference[:horizon, mask].sum(axis=0)
            _finite([nominal_numerator, real_numerator, nominal_denominator, real_denominator], "aggregate arithmetic")
            _finite(household_numerators, "household cumulative arithmetic")
            defined = denominator_real[mask] > 0
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                ratios = household_numerators[defined] / denominator_real[mask][defined]
            _finite(ratios, "household response ratios")
            count = int(mask.sum())
            valid = real_denominator > 0
            nominal_response = float(nominal_numerator / nominal_denominator) if valid else None
            real_response = float(real_numerator / real_denominator) if valid else None
            if valid:
                _finite([nominal_response, real_response], "aggregate response ratios")
            by_group[name] = {
                "status": "empty_group" if count == 0 else "ok" if valid else "zero_intervention",
                "household_count": count,
                "defined_household_ratios": int(defined.sum()),
                "zero_injection_households": int(count - defined.sum()),
                "nominal_numerator": float(nominal_numerator),
                "real_numerator": float(real_numerator),
                "nominal_denominator": float(nominal_denominator),
                "real_denominator": float(real_denominator),
                "nominal_response": nominal_response,
                "real_response": real_response,
                "household_median": float(np.median(ratios)) if ratios.size else None,
                "household_p10": float(np.quantile(ratios, 0.1, method="linear")) if ratios.size else None,
                "household_p90": float(np.quantile(ratios, 0.9, method="linear")) if ratios.size else None,
                "household_negative_share": float(np.mean(ratios < 0)) if ratios.size else None,
            }
        summaries[int(horizon)] = by_group
    return PairedResponse(
        outcome,
        "income_mpc" if intervention_kind == "income_transfer" else "lfa_spending_response",
        ids,
        tuple(int(p) for p in periods),
        _freeze(nominal_difference),
        _freeze(real_difference),
        _freeze(denominator_real),
        _freeze(summaries),
    )


def summarize_seed_responses(responses, *, registered_seeds):
    """Descriptive seed mean/SD/min/max; a missing seed invalidates pooled stats."""
    seeds = tuple(registered_seeds)
    if not seeds or any(
        isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) for seed in seeds
    ):
        raise ValueError("Registered seed IDs must be nonempty integers.")
    if len(set(seeds)) != len(seeds) or not isinstance(responses, Mapping) or set(responses) - set(seeds):
        raise ValueError("Responses must use unique registered seeds only.")
    responses = {seed: responses.get(seed) for seed in seeds}
    values = [value for value in responses.values() if value is not None]
    _finite(values, "seed responses")
    complete = len(values) == len(responses)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(np.mean(values)) if complete else None
        sd = float(np.std(values, ddof=1)) if complete and len(values) > 1 else None
    if complete:
        _finite([mean] + ([] if sd is None else [sd]), "seed summary arithmetic")
    return _freeze(
        {
            "status": "ok" if complete else "incomplete",
            "per_seed": responses,
            "registered_seed_count": len(responses),
            "available_seed_count": len(values),
            "mean": mean,
            "sd": sd,
            "min": float(min(values)) if complete else None,
            "max": float(max(values)) if complete else None,
        }
    )
