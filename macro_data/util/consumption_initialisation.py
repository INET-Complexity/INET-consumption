"""Opening consumption allocations, before the CACF residual is anchored."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InitialConsumptionAllocation:
    """Candidate and reconciled opening expenditure in model-period units."""

    candidate_total: np.ndarray
    candidate_goods: np.ndarray
    goods: np.ndarray
    total: np.ndarray
    goods_scale: float


@dataclass(frozen=True)
class WinsorisedConsumptionRatios:
    """Source-row-aligned ratios; invalid observations remain NaN."""

    ratios: np.ndarray
    valid: np.ndarray
    capped: np.ndarray
    lower: float
    upper: float


def winsorise_hfcs_consumption_ratios(
    *,
    c_y: np.ndarray,
    survey_weights: np.ndarray,
    hard_valid: np.ndarray,
    threshold_pool: np.ndarray | None = None,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> WinsorisedConsumptionRatios:
    """Cap source C/Y using the calibration's weighted percentile convention.

    Call on source records BEFORE weighted model sampling. ``hard_valid`` must
    be the calibration's pre-percentile validity mask, including its hard bounds
    and checks on other variables. Do not pass the final trimmed-sample mask.
    ``threshold_pool`` optionally selects the registered estimation wave(s).
    Other-variable percentile tails do not remove otherwise valid households.

    Quantiles interpolate against cumulative survey-weight midpoints, exactly
    as consumption_calibration.weighted_quantile does. Invalid source records
    stay in place as NaN with valid=False; no donor, floor or legacy fallback
    is silently substituted. This helper does not reconstruct calibration
    validity rules or infer source-income definitions from model income.
    """
    ratios = np.asarray(c_y, dtype=float)
    weights = np.asarray(survey_weights, dtype=float)
    valid = np.asarray(hard_valid)
    if ratios.ndim != 1 or weights.shape != ratios.shape or valid.shape != ratios.shape:
        raise ValueError("Source ratios, survey weights and hard_valid must be aligned household vectors.")
    if valid.dtype.kind != "b":
        raise ValueError("hard_valid must be a boolean pre-percentile validity mask.")
    valid = valid.copy()
    if threshold_pool is None:
        pool = valid.copy()
    else:
        pool = np.asarray(threshold_pool)
        if pool.shape != ratios.shape or pool.dtype.kind != "b":
            raise ValueError("threshold_pool must be an aligned boolean mask.")
        pool = pool & valid
    for name, value in (("lower_quantile", lower_quantile), ("upper_quantile", upper_quantile)):
        a = np.asarray(value, dtype=float)
        if a.shape != () or not np.isfinite(a) or not 0 <= a <= 1:
            raise ValueError(f"{name} must be a scalar probability.")
    if lower_quantile >= upper_quantile:
        raise ValueError("lower_quantile must be below upper_quantile.")
    if np.any(valid & (~np.isfinite(ratios) | (ratios < 0) | ~np.isfinite(weights) | (weights <= 0))):
        raise ValueError("hard_valid includes invalid C/Y or survey weights; apply calibration validity first.")
    if not np.any(pool):
        raise ValueError("No hard-valid source households remain in the threshold pool.")
    order = np.argsort(ratios[pool])
    x = ratios[pool][order]
    w = weights[pool][order]
    # Common rescaling leaves quantiles unchanged and prevents weight overflow.
    w = w / w.max()
    mass = (np.cumsum(w) - 0.5 * w) / w.sum()
    lower, upper = np.interp([lower_quantile, upper_quantile], mass, x)
    result = np.full(ratios.shape, np.nan)
    result[valid] = np.clip(ratios[valid], lower, upper)
    capped = valid & ((ratios < lower) | (ratios > upper))
    return WinsorisedConsumptionRatios(result, valid, capped, float(lower), float(upper))


def allocate_hfcs_ratio_consumption(
    *,
    hfcs_c_y: np.ndarray,
    model_income: np.ndarray,
    cash_rent: np.ndarray,
    imputed_rent: np.ndarray,
    vat: float,
    aggregate_net_goods: float,
) -> InitialConsumptionAllocation:
    """Apply source total C/Y to comparable model income and reconcile goods.

    All flows must use the same currency, period and population scale. Income
    must match the source ratio's concept, before any planning-income floor.
    Rents and total consumption are at purchase prices; goods are VAT exclusive.
    The aggregate goods anchor includes any goods used to supply housing
    services. Callers must not add those goods a second time to this allocation.

    Reconciliation scales goods proportionally, leaving rent unchanged. Thus
    source ratios hold before reconciliation; they generally change afterwards.
    Infeasible household budgets are rejected rather than silently floored.
    This calculation books no transaction and neither fits nor uses saving rates.
    """
    values = {
        "hfcs_c_y": np.asarray(hfcs_c_y, dtype=float),
        "model_income": np.asarray(model_income, dtype=float),
        "cash_rent": np.asarray(cash_rent, dtype=float),
        "imputed_rent": np.asarray(imputed_rent, dtype=float),
    }
    shape = values["hfcs_c_y"].shape
    if len(shape) != 1 or any(a.shape != shape for a in values.values()):
        raise ValueError("Initial consumption inputs must be household vectors with identical shapes.")
    for name, array in values.items():
        if not np.all(np.isfinite(array)) or np.any(array < 0):
            raise ValueError(f"{name} must contain finite nonnegative values.")
    if np.any(values["model_income"] <= 0):
        raise ValueError("Comparable model income must be strictly positive; no income floor is applied here.")
    for name, value in (("vat", vat), ("aggregate_net_goods", aggregate_net_goods)):
        scalar = np.asarray(value, dtype=float)
        if scalar.shape != () or not np.isfinite(scalar) or scalar < 0:
            raise ValueError(f"{name} must be a finite nonnegative scalar.")

    candidate_total = values["hfcs_c_y"] * values["model_income"]
    housing = values["cash_rent"] + values["imputed_rent"]
    if not np.all(np.isfinite(candidate_total)) or not np.all(np.isfinite(housing)):
        raise ValueError("Initial expenditure arithmetic must remain finite.")
    cash_goods = candidate_total - housing
    below_housing = cash_goods < 0
    if np.any(below_housing):
        raise ValueError(
            f"HFCS-ratio total is below housing expenditure for {np.count_nonzero(below_housing)} households; "
            "choose an explicit treatment before initialising."
        )
    candidate_goods = cash_goods / (1.0 + float(vat))
    goods_sum = candidate_goods.sum()
    if not np.isfinite(goods_sum):
        raise ValueError("Aggregate candidate goods must be finite.")
    if goods_sum == 0:
        if aggregate_net_goods > 0:
            raise ValueError("Cannot allocate positive aggregate goods without positive household goods shares.")
        goods_scale = 1.0
    else:
        goods_scale = float(aggregate_net_goods) / goods_sum
    goods = candidate_goods * goods_scale
    total = (1.0 + float(vat)) * goods + housing
    if not np.isfinite(goods_scale) or not np.all(np.isfinite(total)):
        raise ValueError("Reconciled initial expenditure must remain finite.")
    return InitialConsumptionAllocation(candidate_total, candidate_goods, goods, total, goods_scale)
