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
    legacy_fallback: np.ndarray | None = None
    # DNA allocator only: opening income quintile (0..4) of each feasible household, -1 for legacy fallback.
    quintile: np.ndarray | None = None


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


def allocate_hfcs_with_legacy_reserve(
    *, hfcs_c_y, model_income, cash_rent, imputed_rent, vat, aggregate_net_goods, legacy_goods
) -> InitialConsumptionAllocation:
    """Retain legacy goods for finite infeasible states; allocate the remainder.

    This changes opening consumption only. It neither repairs income nor books
    financing/transfers. Missing or malformed inputs remain errors. The candidate
    total remains the raw HFCS target; candidate goods use legacy amounts on the
    fallback subset. The scale factor applies only to feasible households.
    """
    values = {
        name: np.asarray(value, dtype=float)
        for name, value in {
            "hfcs_c_y": hfcs_c_y,
            "model_income": model_income,
            "cash_rent": cash_rent,
            "imputed_rent": imputed_rent,
            "legacy_goods": legacy_goods,
        }.items()
    }
    shape = values["model_income"].shape
    if len(shape) != 1 or any(a.shape != shape for a in values.values()):
        raise ValueError("Initial consumption inputs must be aligned household vectors.")
    for name, array in values.items():
        if not np.isfinite(array).all() or (name != "model_income" and (array < 0).any()):
            raise ValueError(f"{name} must be finite and nonnegative, except finite negative model income.")
    for name, value in (("vat", vat), ("aggregate_net_goods", aggregate_net_goods)):
        scalar = np.asarray(value, dtype=float)
        if scalar.shape != () or not np.isfinite(scalar) or scalar < 0:
            raise ValueError(f"{name} must be a finite nonnegative scalar.")
    candidate = values["hfcs_c_y"] * values["model_income"]
    housing = values["cash_rent"] + values["imputed_rent"]
    if not np.isfinite(candidate).all() or not np.isfinite(housing).all():
        raise ValueError("Initial expenditure arithmetic must remain finite.")
    fallback = (values["model_income"] <= 0) | (candidate < housing)
    reserved = values["legacy_goods"][fallback].sum()
    remaining = float(aggregate_net_goods) - reserved
    if not np.isfinite(reserved) or (
        remaining < 0 and not np.isclose(remaining, 0, rtol=0, atol=1e-12 * float(aggregate_net_goods))
    ):
        raise ValueError("Reserved legacy goods exceed the aggregate consumption budget.")
    remaining = max(0.0, remaining)
    feasible = ~fallback
    goods = values["legacy_goods"].copy()
    candidate_goods = goods.copy()
    goods_scale = 1.0
    if feasible.any():
        allocation = allocate_hfcs_ratio_consumption(
            hfcs_c_y=values["hfcs_c_y"][feasible],
            model_income=values["model_income"][feasible],
            cash_rent=values["cash_rent"][feasible],
            imputed_rent=values["imputed_rent"][feasible],
            vat=vat,
            aggregate_net_goods=remaining,
        )
        goods[feasible] = allocation.goods
        candidate_goods[feasible] = allocation.candidate_goods
        goods_scale = allocation.goods_scale
    elif not np.isclose(remaining, 0, rtol=0, atol=1e-12 * float(aggregate_net_goods)):
        raise ValueError("No feasible households can absorb the remaining aggregate goods budget.")
    total = (1 + float(vat)) * goods + housing
    if not np.isfinite(total).all():
        raise ValueError("Reconciled initial expenditure must remain finite.")
    return InitialConsumptionAllocation(candidate, candidate_goods, goods, total, goods_scale, fallback)


def allocate_dna_quintile_consumption(
    *,
    hfcs_c_y,
    model_income,
    household_size,
    cash_rent,
    imputed_rent,
    vat,
    aggregate_net_goods,
    legacy_goods,
    target_c_y,
) -> InitialConsumptionAllocation:
    """Opening budget whose quintile C/Y equals lambda * target C/Y_q, with goods summing to the anchor.

    Quintiles rank income / sqrt(household_size) among positive-income households. Over feasible households
    (F) the common level is the closed form
        lambda = ((1 + vat) * G_F + sum_F H) / sum_q target_q * Y_{q,F},
    G_F = anchor - legacy goods of fallback households, H = cash rent + imputed rent. Quintile total
    C_q = lambda * target_q * Y_{q,F} is split within the quintile in proportion to hfcs_c_y * income, and
    goods = (C - H) / (1 + vat). Total C (housing included) is scaled, so the shape holds exactly.
    Fallback households keep legacy goods: income <= 0, raw HFCS total below housing, or allocated total
    below housing (iterated until none remain). Their number is ``legacy_fallback.sum()``.
    ``quintile`` returns the quintile index (0..4) of every non-fallback household and -1 for fallback households.
    ``goods_scale`` holds lambda.
    """
    values = {
        name: np.asarray(value, dtype=float)
        for name, value in {
            "hfcs_c_y": hfcs_c_y,
            "model_income": model_income,
            "household_size": household_size,
            "cash_rent": cash_rent,
            "imputed_rent": imputed_rent,
            "legacy_goods": legacy_goods,
        }.items()
    }
    shape = values["model_income"].shape
    if len(shape) != 1 or any(a.shape != shape for a in values.values()):
        raise ValueError("Initial consumption inputs must be aligned household vectors.")
    for name, array in values.items():
        if not np.isfinite(array).all() or (name != "model_income" and (array < 0).any()):
            raise ValueError(f"{name} must be finite and nonnegative, except finite negative model income.")
    if (values["household_size"] <= 0).any():
        raise ValueError("household_size must be positive.")
    target = np.asarray(target_c_y, dtype=float)
    if target.shape != (5,) or not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError("target_c_y must contain five positive ratios.")
    for name, value in (("vat", vat), ("aggregate_net_goods", aggregate_net_goods)):
        scalar = np.asarray(value, dtype=float)
        if scalar.shape != () or not np.isfinite(scalar) or scalar < 0:
            raise ValueError(f"{name} must be a finite nonnegative scalar.")
    y, c_y = values["model_income"], values["hfcs_c_y"]
    housing = values["cash_rent"] + values["imputed_rent"]
    candidate = c_y * y
    positive = y > 0
    quintile = np.full(shape, -1)
    rank = np.argsort(np.argsort(y[positive] / np.sqrt(values["household_size"][positive]), kind="stable"))
    quintile[positive] = rank * 5 // max(positive.sum(), 1)
    fallback = ~positive | (candidate < housing)
    vat1 = 1.0 + float(vat)
    tol = 1e-12 * max(float(aggregate_net_goods), 1.0)
    while True:
        feasible = ~fallback
        remaining = float(aggregate_net_goods) - values["legacy_goods"][fallback].sum()
        if remaining < -tol:
            raise ValueError("Reserved legacy goods exceed the aggregate consumption budget.")
        if not feasible.any():
            if abs(remaining) > tol:
                raise ValueError("No feasible households can absorb the remaining aggregate goods budget.")
            total = np.zeros(shape)
            lam = 1.0
            break
        weighted_target = np.zeros(shape)
        total_q = np.zeros(5)
        for q in range(5):
            m = feasible & (quintile == q)
            if not m.any():
                continue
            denominator = candidate[m].sum()
            if not denominator > 0:
                raise ValueError(f"HFCS-ratio consumption of quintile {q + 1} must be positive.")
            total_q[q] = target[q] * y[m].sum()
            weighted_target[m] = candidate[m] / denominator
        numerator = vat1 * max(remaining, 0.0) + housing[feasible].sum()
        lam = numerator / total_q.sum()
        total = np.zeros(shape)
        for q in range(5):
            m = feasible & (quintile == q)
            total[m] = lam * total_q[q] * weighted_target[m]
        below = feasible & (total < housing)
        if not below.any():
            break
        fallback = fallback | below
    goods = values["legacy_goods"].copy()
    goods[feasible] = (total[feasible] - housing[feasible]) / vat1
    total = vat1 * goods + housing
    if not np.isfinite(total).all():
        raise ValueError("Reconciled initial expenditure must remain finite.")
    candidate_goods = np.where(fallback, values["legacy_goods"], np.maximum(candidate - housing, 0.0) / vat1)
    return InitialConsumptionAllocation(
        candidate, candidate_goods, goods, total, float(lam), fallback, np.where(fallback, -1, quintile)
    )
