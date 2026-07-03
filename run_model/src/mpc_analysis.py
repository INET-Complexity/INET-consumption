"""Analysis helpers for paired household MPC experiments.

The experiment runner saves a baseline HDF5 and a shocked HDF5 for each seed.
This module reads those files, computes household-level realised and target MPCs,
adds pre-shock heterogeneity bins, and writes distribution plots.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from macromodel.agents.individuals.individual_properties import ActivityStatus

TENURE_LABELS = {
    -1: "unknown",
    1: "owner_outright",
    2: "owner_mortgage",
    3: "renter",
    4: "owner_other",
}
BINNING_VARIABLES = (
    "income",
    "net_wealth",
    "net_liquid_financial_assets",
    "illiquid_financial_assets",
    "housing_wealth",
    "housing_tenure",
)
POSITIVE_ONLY_BINNING_VARIABLES = {"illiquid_financial_assets", "housing_wealth"}
PLOT_KINDS = {"violin", "box", "box_only"}
MPC_PLOT_MEASURES = {"nominal", "real"}
WINSOR_MODES = {"clip", "trim"}
PLOT_OUTPUT_FORMATS = {"html", "png", "both"}
SAVED_PLOT_BASE_HEIGHT = 600
SAVED_PLOT_HEIGHT_SCALE = 0.8


def _validate_quantile_pair(values: tuple[float, float] | None, *, name: str) -> tuple[float, float] | None:
    """Validate optional lower/upper quantile arguments."""
    if values is None:
        return None
    lower, upper = values
    lower = float(lower)
    upper = float(upper)
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError(f"{name} must satisfy 0 <= lower < upper <= 1.")
    return lower, upper


def _validate_range_pair(values: tuple[float, float] | None, *, name: str) -> tuple[float, float] | None:
    """Validate optional lower/upper numeric range arguments."""
    if values is None:
        return None
    lower, upper = values
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(f"{name} must contain finite values with lower < upper.")
    return lower, upper


@dataclass(frozen=True)
class MPCFilterConfig:
    """Configurable sample filters for robust MPC summaries and plots.

    The raw household MPC panel should be kept unchanged. These filters define
    the analysis sample used for distribution plots and summary tables.
    """

    enabled: bool = True
    min_income: float | None = 100.0
    income_top_quantile: float | None = 0.99
    consumption_income_ratio_quantiles: tuple[float, float] | None = (0.01, 0.99)
    income_growth_min: float | None = None
    income_growth_max: float | None = None
    max_debt_asset_ratio: float | None = None
    gross_wealth_top_quantile: float | None = 0.999
    head_age_min: float | None = 25.0
    head_age_max: float | None = 75.0
    min_abs_income_change: float | None = None
    min_abs_income_change_income_share: float | None = None
    min_abs_shock_amount: float | None = None


def period_to_year_month(initial_year: int, time_unit: int, period: int) -> tuple[int, int]:
    """Convert a zero-based simulation period to the calendar date used by prehooks.

    HDF5 time-series row ``0`` is the initial state. A shock staged during
    simulation iteration ``period`` therefore affects realised HDF5 row
    ``period + 1``.
    """
    if period < 0:
        raise ValueError("period must be non-negative.")
    month_index = 1 + int(period) * int(time_unit)
    year = int(initial_year) + (month_index - 1) // 12
    month = ((month_index - 1) % 12) + 1
    return year, month


def read_household_consumption(h5_path: str | Path, country_code: str) -> np.ndarray:
    """Read realised household nominal consumption from a saved simulation."""
    with h5py.File(h5_path, "r") as handle:
        return np.asarray(handle[f"{country_code}/households/consumption"], dtype=float)


def read_household_income(h5_path: str | Path, country_code: str) -> np.ndarray | None:
    """Read household income if it was saved in the HDF5 output."""
    with h5py.File(h5_path, "r") as handle:
        path = f"{country_code}/households/income"
        if path not in handle:
            return None
        return np.asarray(handle[path], dtype=float)


def read_cpi(h5_path: str | Path, country_code: str, *, cpi_source: str = "cpi_fixed_basket") -> np.ndarray:
    """Read a saved CPI series as a one-dimensional array."""
    with h5py.File(h5_path, "r") as handle:
        cpi = np.asarray(handle[f"{country_code}/economy/{cpi_source}"], dtype=float).reshape(-1)
    if np.any(~np.isfinite(cpi)) or np.any(cpi <= 0.0):
        raise ValueError(f"{cpi_source} must be finite and strictly positive for real MPC analysis.")
    return cpi


def read_household_target_consumption_sum(h5_path: str | Path, country_code: str) -> np.ndarray:
    """Read target consumption and sum industry budgets for each household.

    ``TimeSeries.write_3d_field_to_h5`` flattens a 3D array
    ``(time, household, industry)`` into a 2D dataset and stores the original
    household/industry column labels in ``target_consumption_columns``. MPC
    analysis needs the household total, so this helper reconstructs the
    household dimension before differencing baseline and shocked runs.
    """
    with h5py.File(h5_path, "r") as handle:
        group = handle[f"{country_code}/households"]
        target = np.asarray(group["target_consumption"], dtype=float)
        columns = np.asarray(group["target_consumption_columns"], dtype=int)

    if target.ndim != 2 or columns.ndim != 2 or columns.shape[1] != 2:
        raise ValueError("Unexpected target_consumption HDF5 shape.")

    n_households = int(columns[:, 0].max()) + 1
    target_sum = np.zeros((target.shape[0], n_households), dtype=float)
    for column_index, (household_id, _industry_id) in enumerate(columns):
        target_sum[:, int(household_id)] += target[:, column_index]
    return target_sum


def compute_shock_amount(metadata: pd.DataFrame, shock_fraction: float) -> float:
    """Return the equal nominal shock amount implied by pre-shock metadata."""
    positive_income = metadata.loc[metadata["income"] > 0.0, "income"].dropna()
    if positive_income.empty:
        raise ValueError("Cannot compute shock amount: no positive pre-shock household incomes.")
    return float(shock_fraction * positive_income.median())


def _delta_at_and_cumulative(
    baseline: np.ndarray,
    shock: np.ndarray,
    *,
    shock_row: int,
    horizon_periods: int,
) -> tuple[np.ndarray, np.ndarray]:
    if horizon_periods <= 0:
        raise ValueError("horizon_periods must be positive.")
    if shock_row < 0:
        raise ValueError("shock_row must be non-negative.")
    if baseline.shape != shock.shape:
        raise ValueError(f"Baseline and shock arrays must have the same shape: {baseline.shape} != {shock.shape}.")
    if shock_row + horizon_periods > baseline.shape[0]:
        raise ValueError("Requested MPC horizon extends beyond available HDF5 rows.")

    delta = shock - baseline
    return delta[shock_row], delta[shock_row : shock_row + horizon_periods].sum(axis=0)


def _deflate_by_own_cpi(values: np.ndarray, cpi: np.ndarray) -> np.ndarray:
    """Deflate each path by its own period CPI."""
    if values.shape[0] != cpi.shape[0]:
        raise ValueError(f"CPI length must match values rows: {cpi.shape[0]} != {values.shape[0]}.")
    return np.divide(values, cpi[:, None])


def _series_or_nan(values: np.ndarray | None, *, row: int, n_households: int) -> np.ndarray:
    """Return one HDF5 row or an all-NaN vector when a saved series is absent."""
    if values is None:
        return np.full(n_households, np.nan)
    if row >= values.shape[0]:
        raise ValueError(f"Requested row {row} extends beyond saved series with {values.shape[0]} rows.")
    return np.asarray(values[row], dtype=float)


def build_household_mpc_panel(
    *,
    baseline_h5: str | Path,
    shock_h5: str | Path,
    metadata: pd.DataFrame,
    country_code: str,
    shock_row: int,
    horizon_periods: int = 4,
    shock_fraction: float = 0.01,
    cpi_source: str = "cpi_fixed_basket",
    cumulative_mpc_column: str = "cmpc_4q",
    target_cumulative_mpc_column: str = "target_cmpc_4q",
    real_cumulative_mpc_column: str = "real_cmpc_4q",
    target_real_cumulative_mpc_column: str = "target_real_cmpc_4q",
) -> pd.DataFrame:
    """Build a household-level MPC panel from paired baseline and shock HDF5s.

    The returned panel preserves pre-shock household characteristics and adds:
    impact realised MPC, cumulative realised MPC, and target-consumption
    counterparts. All MPC denominators use the common equal household shock.
    """
    panel = metadata.sort_values("household_id").reset_index(drop=True).copy()
    household_ids = panel["household_id"].to_numpy(dtype=int)
    if not np.array_equal(household_ids, np.arange(household_ids.size)):
        raise ValueError("Metadata household_id values must cover 0..n_households-1 exactly once.")

    shock_amount = compute_shock_amount(panel, shock_fraction)

    baseline_consumption = read_household_consumption(baseline_h5, country_code)
    shock_consumption = read_household_consumption(shock_h5, country_code)
    baseline_income = read_household_income(baseline_h5, country_code)
    shock_income = read_household_income(shock_h5, country_code)
    baseline_cpi = read_cpi(baseline_h5, country_code, cpi_source=cpi_source)
    shock_cpi = read_cpi(shock_h5, country_code, cpi_source=cpi_source)
    impact, cumulative = _delta_at_and_cumulative(
        baseline_consumption,
        shock_consumption,
        shock_row=shock_row,
        horizon_periods=horizon_periods,
    )
    real_impact, real_cumulative = _delta_at_and_cumulative(
        _deflate_by_own_cpi(baseline_consumption, baseline_cpi),
        _deflate_by_own_cpi(shock_consumption, shock_cpi),
        shock_row=shock_row,
        horizon_periods=horizon_periods,
    )

    baseline_target = read_household_target_consumption_sum(baseline_h5, country_code)
    shock_target = read_household_target_consumption_sum(shock_h5, country_code)
    target_impact, target_cumulative = _delta_at_and_cumulative(
        baseline_target,
        shock_target,
        shock_row=shock_row,
        horizon_periods=horizon_periods,
    )
    target_real_impact, target_real_cumulative = _delta_at_and_cumulative(
        _deflate_by_own_cpi(baseline_target, baseline_cpi),
        _deflate_by_own_cpi(shock_target, shock_cpi),
        shock_row=shock_row,
        horizon_periods=horizon_periods,
    )

    real_shock_amount = shock_amount / shock_cpi[shock_row]
    baseline_income_impact = _series_or_nan(baseline_income, row=shock_row, n_households=household_ids.size)
    shock_income_impact = _series_or_nan(shock_income, row=shock_row, n_households=household_ids.size)
    panel["shock_amount"] = shock_amount
    panel["real_shock_amount"] = real_shock_amount
    panel["baseline_consumption_impact"] = baseline_consumption[shock_row]
    panel["shock_consumption_impact"] = shock_consumption[shock_row]
    panel["baseline_consumption_cumulative"] = baseline_consumption[shock_row : shock_row + horizon_periods].sum(axis=0)
    panel["baseline_consumption_to_income"] = np.divide(
        panel["baseline_consumption_impact"],
        panel["income"],
        out=np.full(household_ids.size, np.nan),
        where=np.asarray(panel["income"], dtype=float) > 0.0,
    )
    panel["baseline_income_impact"] = baseline_income_impact
    panel["shock_income_impact"] = shock_income_impact
    panel["income_change"] = shock_income_impact - baseline_income_impact
    panel["abs_income_change"] = np.abs(panel["income_change"])
    panel["income_growth"] = np.divide(
        panel["income_change"],
        baseline_income_impact,
        out=np.full(household_ids.size, np.nan),
        where=baseline_income_impact > 0.0,
    )
    panel["abs_income_change_income_share"] = np.divide(
        panel["abs_income_change"],
        panel["income"],
        out=np.full(household_ids.size, np.nan),
        where=np.asarray(panel["income"], dtype=float) > 0.0,
    )
    panel["mpc_impact"] = impact / shock_amount
    panel[cumulative_mpc_column] = cumulative / shock_amount
    panel["target_mpc_impact"] = target_impact / shock_amount
    panel[target_cumulative_mpc_column] = target_cumulative / shock_amount
    panel["real_mpc_impact"] = real_impact / real_shock_amount
    panel[real_cumulative_mpc_column] = real_cumulative / real_shock_amount
    panel["target_real_mpc_impact"] = target_real_impact / real_shock_amount
    panel[target_real_cumulative_mpc_column] = target_real_cumulative / real_shock_amount
    for periods in range(2, horizon_periods):
        suffix = "4q" if periods == 4 else f"{periods}p"
        nominal_horizon = _delta_at_and_cumulative(
            baseline_consumption,
            shock_consumption,
            shock_row=shock_row,
            horizon_periods=periods,
        )[1]
        target_horizon = _delta_at_and_cumulative(
            baseline_target,
            shock_target,
            shock_row=shock_row,
            horizon_periods=periods,
        )[1]
        real_horizon = _delta_at_and_cumulative(
            _deflate_by_own_cpi(baseline_consumption, baseline_cpi),
            _deflate_by_own_cpi(shock_consumption, shock_cpi),
            shock_row=shock_row,
            horizon_periods=periods,
        )[1]
        target_real_horizon = _delta_at_and_cumulative(
            _deflate_by_own_cpi(baseline_target, baseline_cpi),
            _deflate_by_own_cpi(shock_target, shock_cpi),
            shock_row=shock_row,
            horizon_periods=periods,
        )[1]
        panel[f"cmpc_{suffix}"] = nominal_horizon / shock_amount
        panel[f"target_cmpc_{suffix}"] = target_horizon / shock_amount
        panel[f"real_cmpc_{suffix}"] = real_horizon / real_shock_amount
        panel[f"target_real_cmpc_{suffix}"] = target_real_horizon / real_shock_amount
    return panel


def _current_or_zero(households, field: str, *, n_households: int) -> np.ndarray:
    try:
        return np.asarray(households.ts.current(field), dtype=float)
    except (KeyError, AttributeError):
        return np.zeros(n_households, dtype=float)


def _household_head_age_proxy(country, *, n_households: int) -> np.ndarray:
    """Return the oldest linked individual age per household when available.

    NOTE — coexisting "household head" definition: this proxy uses oldest-by-age
    only and does not consult "Relation to Reference Person" (RA0100). Stage 4's
    FRM covariate path (Households.from_pickled_agent's household_head_age state,
    macromodel/agents/households/households.py) instead uses RA0100 == 1 (HFCS
    reference person), falling back to oldest-by-age only when no individual is
    flagged. The two definitions can disagree whenever the reference person is
    not the oldest household member. Do not assume this MPC proxy and Stage 4's
    household_head_age agree without checking, if ever combining their outputs.
    """
    ages = getattr(getattr(country, "individuals", None), "states", {}).get("Age")
    corr_individuals = getattr(country.households, "states", {}).get("corr_individuals")
    if ages is None or corr_individuals is None:
        return np.full(n_households, np.nan)

    age_array = np.asarray(ages, dtype=float)
    result = np.full(n_households, np.nan)
    for household_id, individual_ids in enumerate(corr_individuals):
        ids = np.asarray(individual_ids, dtype=int)
        valid_ids = ids[(ids >= 0) & (ids < age_array.size)]
        if valid_ids.size:
            result[household_id] = np.nanmax(age_array[valid_ids])
    return result


ACTIVITY_BRACKET_CODES = {
    "employed": 1,
    "unemployed": 2,
    "investor": 3,
    "not_economically_active": 4,
}
HOUSEHOLD_LABOR_STATE_COLUMNS = (
    "activity_bracket",
    "worked_hours",
    "labor_income",
)


def household_activity_bracket(country) -> np.ndarray:
    """Return each household's labor-market activity bracket as an integer code.

    A household's bracket is resolved from its linked individuals' Activity
    Status by priority: employed > unemployed > investor (firm or bank) > not
    economically active. Codes are listed in ``ACTIVITY_BRACKET_CODES``.

    Activity Status is a live model state, not a saved HDF5 time series (unlike
    income/consumption), so callers must capture this via a posthook during the
    simulation run rather than reading it back from disk afterwards.
    """
    households = country.households
    n_households = int(households.ts.current("n_households"))
    activity = getattr(getattr(country, "individuals", None), "states", {}).get("Activity Status")
    corr_individuals = getattr(households, "states", {}).get("corr_individuals")
    if activity is None or corr_individuals is None:
        return np.full(n_households, np.nan)

    activity = np.asarray(activity)
    employed_mask = activity == ActivityStatus.EMPLOYED
    unemployed_mask = activity == ActivityStatus.UNEMPLOYED
    investor_mask = (activity == ActivityStatus.FIRM_INVESTOR) | (activity == ActivityStatus.BANK_INVESTOR)

    result = np.full(n_households, np.nan)
    for household_id, individual_ids in enumerate(corr_individuals):
        ids = np.asarray(individual_ids, dtype=int)
        valid_ids = ids[(ids >= 0) & (ids < activity.size)]
        if not valid_ids.size:
            continue
        if employed_mask[valid_ids].any():
            result[household_id] = ACTIVITY_BRACKET_CODES["employed"]
        elif unemployed_mask[valid_ids].any():
            result[household_id] = ACTIVITY_BRACKET_CODES["unemployed"]
        elif investor_mask[valid_ids].any():
            result[household_id] = ACTIVITY_BRACKET_CODES["investor"]
        else:
            result[household_id] = ACTIVITY_BRACKET_CODES["not_economically_active"]
    return result


def household_worked_hours(country) -> np.ndarray:
    """Return each household's total worked-hours proxy from linked individuals.

    The model stores labour supply intensity as ``individuals.ts.current(
    "labour_inputs")``. For MPC identification we aggregate that live state to
    the household level so households can be dropped when their effective labour
    supply changes within the measurement window.
    """
    households = country.households
    n_households = int(households.ts.current("n_households"))
    corr_individuals = getattr(households, "states", {}).get("corr_individuals")
    if corr_individuals is None:
        return np.full(n_households, np.nan)

    individuals = getattr(country, "individuals", None)
    try:
        labour_inputs = np.asarray(individuals.ts.current("labour_inputs"), dtype=float)
    except (AttributeError, KeyError):
        labour_inputs = None
    if labour_inputs is None:
        return np.full(n_households, np.nan)

    result = np.full(n_households, np.nan)
    for household_id, individual_ids in enumerate(corr_individuals):
        ids = np.asarray(individual_ids, dtype=int)
        valid_ids = ids[(ids >= 0) & (ids < labour_inputs.size)]
        if not valid_ids.size:
            continue
        household_hours = labour_inputs[valid_ids]
        if np.any(np.isfinite(household_hours)):
            result[household_id] = np.nansum(household_hours)
    return result


def household_labor_income(country) -> np.ndarray:
    """Return each household's current labour income from employment."""
    households = country.households
    n_households = int(households.ts.current("n_households"))
    try:
        return np.asarray(households.ts.current("income_employee"), dtype=float)
    except KeyError:
        return np.full(n_households, np.nan)


def make_household_labor_state_snapshot(country) -> dict[str, np.ndarray]:
    """Capture the household labour-state vectors used for MPC identification."""
    snapshot = {
        "activity_bracket": household_activity_bracket(country),
        "worked_hours": household_worked_hours(country),
        "labor_income": household_labor_income(country),
    }
    lengths = {name: values.shape[0] for name, values in snapshot.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Household labour-state snapshots must share a common length, got {lengths}.")
    return snapshot


def make_household_metadata(country, *, seed: int, pre_shock_row: int) -> pd.DataFrame:
    """Capture pre-shock household characteristics used for binning MPCs.

    Housing tenure is a household state, not a standard saved time series, so the
    runner captures it from the live model before the shock period.
    """
    households = country.households
    n_households = int(households.ts.current("n_households"))
    wealth_main = np.asarray(households.ts.current("wealth_main_residence"), dtype=float)
    wealth_other_property = np.asarray(households.ts.current("wealth_other_properties"), dtype=float)
    wealth_deposits = np.asarray(households.ts.current("wealth_deposits"), dtype=float)
    wealth_other_financial_assets = np.asarray(households.ts.current("wealth_other_financial_assets"), dtype=float)
    wealth_other_real_assets = _current_or_zero(households, "wealth_other_real_assets", n_households=n_households)
    consumption_loan_debt = np.asarray(households.ts.current("consumption_loan_debt"), dtype=float)
    mortgage_debt = _current_or_zero(households, "mortgage_debt", n_households=n_households)
    debt = consumption_loan_debt + mortgage_debt
    gross_wealth = wealth_deposits + wealth_other_financial_assets + wealth_main + wealth_other_property
    gross_wealth = gross_wealth + wealth_other_real_assets
    debt_asset_ratio = np.divide(
        debt,
        gross_wealth,
        out=np.where(debt <= 0.0, 0.0, np.inf),
        where=gross_wealth > 0.0,
    )
    tenure = np.asarray(households.states["Tenure Status of the Main Residence"], dtype=int)

    return pd.DataFrame(
        {
            "seed": int(seed),
            "pre_shock_row": int(pre_shock_row),
            "household_id": np.arange(n_households, dtype=int),
            "income": np.asarray(households.ts.current("income"), dtype=float),
            "net_wealth": np.asarray(households.ts.current("net_wealth"), dtype=float),
            "net_liquid_financial_assets": wealth_deposits - consumption_loan_debt,
            "illiquid_financial_assets": wealth_other_financial_assets,
            "housing_wealth": wealth_main + wealth_other_property,
            "debt": debt,
            "gross_wealth": gross_wealth,
            "debt_asset_ratio": debt_asset_ratio,
            "head_age": _household_head_age_proxy(country, n_households=n_households),
            "housing_tenure": pd.Series(tenure).map(TENURE_LABELS).fillna("other").to_numpy(),
            "tenure_code": tenure,
        }
    )


def assign_quantile_bin(values: pd.Series, *, n_bins: int = 5) -> pd.Series:
    """Assign robust quantile labels while tolerating repeated values.

    Wealth variables often have many ties or zero mass points. Ranking with
    ``method='first'`` avoids ``pd.qcut`` duplicate-edge failures while preserving
    approximately equal bin counts.
    """
    valid = values.replace([np.inf, -np.inf], np.nan)
    labels = pd.Series(pd.NA, index=values.index, dtype="object")
    non_null = valid.dropna()
    if non_null.empty:
        return labels.fillna("missing")
    if non_null.nunique() == 1:
        labels.loc[non_null.index] = "all"
        return labels.fillna("missing")
    ranked = non_null.rank(method="first")
    binned = pd.qcut(ranked, q=min(n_bins, non_null.size), duplicates="drop")
    categories = list(binned.cat.categories)
    label_map = {category: f"Q{index + 1}" for index, category in enumerate(categories)}
    labels.loc[non_null.index] = binned.map(label_map).astype(object)
    return labels.fillna("missing")


def add_mpc_bins(panel: pd.DataFrame, *, n_bins: int = 5) -> pd.DataFrame:
    """Add quintile-style bins for continuous variables and tenure categories."""
    result = panel.copy()
    for variable in BINNING_VARIABLES[:-1]:
        result[f"{variable}_bin"] = assign_quantile_bin(result[variable], n_bins=n_bins)
    result["housing_tenure_bin"] = result["housing_tenure"].astype(str)
    return result


def _filter_report_row(
    *,
    name: str,
    before: int,
    after: int,
    skipped: bool = False,
    reason: str = "",
) -> dict[str, object]:
    return {
        "filter": name,
        "before": int(before),
        "dropped": int(before - after),
        "remaining": int(after),
        "skipped": bool(skipped),
        "reason": reason,
    }


def _finite_series(panel: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(panel[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def filter_mpc_panel(
    panel: pd.DataFrame,
    config: MPCFilterConfig | None = None,
    *,
    required_mpc_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply robust analysis-sample filters and return filtered data plus report.

    Filters are applied sequentially, so the report shows the incremental number
    of households dropped by each rule. Optional rules are skipped when the
    corresponding diagnostic column is unavailable or entirely missing.
    """
    cfg = config or MPCFilterConfig()
    current = panel.copy()
    report: list[dict[str, object]] = []
    if not cfg.enabled:
        report.append(
            _filter_report_row(
                name="filters_enabled",
                before=len(current),
                after=len(current),
                skipped=True,
                reason="MPC filters disabled.",
            )
        )
        return current, pd.DataFrame(report)

    def apply_mask(name: str, mask: pd.Series | np.ndarray, *, reason: str = "") -> None:
        nonlocal current
        before = len(current)
        keep = pd.Series(mask, index=current.index).fillna(False).astype(bool)
        current = current.loc[keep].copy()
        report.append(_filter_report_row(name=name, before=before, after=len(current), reason=reason))

    def skip(name: str, reason: str) -> None:
        report.append(
            _filter_report_row(name=name, before=len(current), after=len(current), skipped=True, reason=reason)
        )

    def apply_column_mask(name: str, column: str, predicate) -> None:
        if column not in current.columns:
            skip(name, f"Missing column: {column}.")
            return
        values = _finite_series(current, column)
        if values.notna().sum() == 0:
            skip(name, f"No finite values in column: {column}.")
            return
        apply_mask(name, predicate(values))

    if required_mpc_columns:
        for column in required_mpc_columns:
            apply_column_mask(f"finite_{column}", column, lambda values: values.notna())

    apply_column_mask("positive_income", "income", lambda values: values > 0.0)
    if cfg.min_income is not None:
        apply_column_mask("minimum_income", "income", lambda values: values >= float(cfg.min_income))

    if cfg.income_top_quantile is not None:
        if "income" not in current.columns:
            skip("top_income_trim", "Missing column: income.")
        else:
            income = _finite_series(current, "income")
            threshold = income.dropna().quantile(float(cfg.income_top_quantile)) if income.notna().any() else np.nan
            if np.isfinite(threshold):
                apply_mask("top_income_trim", income <= threshold, reason=f"threshold={threshold:.6g}")
            else:
                skip("top_income_trim", "No finite income threshold.")

    apply_column_mask("positive_baseline_consumption", "baseline_consumption_impact", lambda values: values > 0.0)
    apply_column_mask("positive_shock_consumption", "shock_consumption_impact", lambda values: values > 0.0)

    if cfg.consumption_income_ratio_quantiles is not None:
        if "baseline_consumption_to_income" not in current.columns:
            skip("consumption_income_ratio_trim", "Missing column: baseline_consumption_to_income.")
        else:
            ratio = _finite_series(current, "baseline_consumption_to_income")
            valid = ratio.dropna()
            if valid.empty:
                skip("consumption_income_ratio_trim", "No finite consumption-to-income ratios.")
            else:
                lower_q, upper_q = cfg.consumption_income_ratio_quantiles
                lower, upper = valid.quantile([float(lower_q), float(upper_q)])
                apply_mask(
                    "consumption_income_ratio_trim",
                    (ratio >= lower) & (ratio <= upper),
                    reason=f"thresholds=[{lower:.6g}, {upper:.6g}]",
                )

    if cfg.income_growth_min is not None:
        apply_column_mask("income_growth_floor", "income_growth", lambda values: values >= float(cfg.income_growth_min))
    if cfg.income_growth_max is not None:
        apply_column_mask(
            "income_growth_ceiling", "income_growth", lambda values: values <= float(cfg.income_growth_max)
        )
    if cfg.max_debt_asset_ratio is not None:
        apply_column_mask(
            "debt_asset_ratio_ceiling",
            "debt_asset_ratio",
            lambda values: values <= float(cfg.max_debt_asset_ratio),
        )

    if cfg.gross_wealth_top_quantile is not None:
        if "gross_wealth" not in current.columns:
            skip("top_gross_wealth_trim", "Missing column: gross_wealth.")
        else:
            gross_wealth = _finite_series(current, "gross_wealth")
            threshold = (
                gross_wealth.dropna().quantile(float(cfg.gross_wealth_top_quantile))
                if gross_wealth.notna().any()
                else np.nan
            )
            if np.isfinite(threshold):
                apply_mask("top_gross_wealth_trim", gross_wealth <= threshold, reason=f"threshold={threshold:.6g}")
            else:
                skip("top_gross_wealth_trim", "No finite gross-wealth threshold.")

    if cfg.head_age_min is not None:
        apply_column_mask("head_age_floor", "head_age", lambda values: values >= float(cfg.head_age_min))
    if cfg.head_age_max is not None:
        apply_column_mask("head_age_ceiling", "head_age", lambda values: values <= float(cfg.head_age_max))
    if cfg.min_abs_income_change is not None:
        apply_column_mask(
            "minimum_abs_income_change",
            "abs_income_change",
            lambda values: values >= float(cfg.min_abs_income_change),
        )
    if cfg.min_abs_income_change_income_share is not None:
        apply_column_mask(
            "minimum_abs_income_change_income_share",
            "abs_income_change_income_share",
            lambda values: values >= float(cfg.min_abs_income_change_income_share),
        )
    if cfg.min_abs_shock_amount is not None:
        apply_column_mask(
            "minimum_abs_shock_amount",
            "shock_amount",
            lambda values: values >= float(cfg.min_abs_shock_amount),
        )

    return current, pd.DataFrame(report)


def _variable_panel(panel: pd.DataFrame, variable: str, *, n_bins: int = 5) -> pd.DataFrame:
    """Return the panel slice and bins used for a variable-level summary or plot."""
    result = panel.copy()
    if variable in POSITIVE_ONLY_BINNING_VARIABLES:
        result = result[result[variable] > 0.0].copy()
        result[f"{variable}_bin"] = assign_quantile_bin(result[variable], n_bins=n_bins)
    elif f"{variable}_bin" not in result.columns:
        result[f"{variable}_bin"] = (
            result["housing_tenure"].astype(str)
            if variable == "housing_tenure"
            else assign_quantile_bin(result[variable])
        )
    return result


def summarize_mpc_bins(panel: pd.DataFrame, *, mpc_column: str = "cmpc_4q") -> pd.DataFrame:
    """Summarise the MPC distribution within each variable/bin pair."""
    rows = []
    weight_column = "real_shock_amount" if mpc_column.startswith(("real_", "target_real_")) else "shock_amount"
    for variable in BINNING_VARIABLES:
        variable_panel = _variable_panel(panel, variable)
        bin_column = f"{variable}_bin"
        for bin_value, group in variable_panel.groupby(bin_column, dropna=False, sort=False):
            values = group[mpc_column].replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "variable": variable,
                    "bin": bin_value,
                    "n": int(values.size),
                    "mean": float(values.mean()),
                    "weighted_mean": float(np.average(values, weights=group.loc[values.index, weight_column])),
                    "median": float(values.median()),
                    "p10": float(values.quantile(0.10)),
                    "p25": float(values.quantile(0.25)),
                    "p75": float(values.quantile(0.75)),
                    "p90": float(values.quantile(0.90)),
                }
            )
    return pd.DataFrame(rows)


def _bin_category_order(values: pd.Series) -> list[str]:
    """Return stable x-axis ordering for quantile and categorical bins."""
    labels = [str(value) for value in values.dropna().unique()]
    quantile_labels = sorted(
        (label for label in labels if label.startswith("Q") and label[1:].isdigit()),
        key=lambda label: int(label[1:]),
    )
    ordered = [*quantile_labels]
    for label in (
        "all",
        "owner_outright",
        "owner_mortgage",
        "renter",
        "owner_other",
        "unknown",
        "missing",
        "other",
    ):
        if label in labels and label not in ordered:
            ordered.append(label)
    ordered.extend(sorted(label for label in labels if label not in ordered))
    return ordered


def make_distribution_plot(
    panel: pd.DataFrame,
    *,
    variable: str,
    mpc_column: str = "cmpc_4q",
    plot_kind: str = "violin",
    title: str | None = None,
    y_quantiles: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    winsor_quantiles: tuple[float, float] | None = None,
    winsor_mode: str = "clip",
) -> go.Figure:
    """Create a distribution plot for one MPC binning variable.

    Display filters are deliberately plot-only. ``winsor_quantiles`` adjusts the
    plotted values for readability, either clipping them to the bounds
    (``winsor_mode="clip"``, the default) or dropping values outside the
    bounds entirely (``winsor_mode="trim"``). ``y_quantiles`` or ``y_range``
    only control the visible y-axis window.
    """
    if plot_kind not in PLOT_KINDS:
        raise ValueError(f"plot_kind must be one of {sorted(PLOT_KINDS)}.")
    if winsor_mode not in WINSOR_MODES:
        raise ValueError(f"winsor_mode must be one of {sorted(WINSOR_MODES)}.")
    y_quantiles = _validate_quantile_pair(y_quantiles, name="y_quantiles")
    winsor_quantiles = _validate_quantile_pair(winsor_quantiles, name="winsor_quantiles")
    y_range = _validate_range_pair(y_range, name="y_range")
    if y_quantiles is not None and y_range is not None:
        raise ValueError("Specify either y_quantiles or y_range, not both.")

    bin_column = f"{variable}_bin"
    plot_panel = _variable_panel(panel, variable)
    plot_values = plot_panel[mpc_column].replace([np.inf, -np.inf], np.nan).dropna()
    winsor_bounds = (
        tuple(plot_values.quantile(list(winsor_quantiles)).astype(float)) if winsor_quantiles is not None else None
    )
    y_quantile_range = tuple(plot_values.quantile(list(y_quantiles)).astype(float)) if y_quantiles is not None else None

    if winsor_bounds is not None and winsor_mode == "trim":
        in_bounds = plot_panel[mpc_column].between(winsor_bounds[0], winsor_bounds[1])
        plot_panel = plot_panel.loc[in_bounds]

    fig = go.Figure()
    category_order = _bin_category_order(plot_panel[bin_column])
    for bin_value in category_order:
        group = plot_panel[plot_panel[bin_column].astype(str) == bin_value]
        values = group[mpc_column].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        if winsor_bounds is not None and winsor_mode == "clip":
            values = values.clip(lower=winsor_bounds[0], upper=winsor_bounds[1])
        if plot_kind == "violin":
            fig.add_trace(
                go.Violin(
                    y=values,
                    name=str(bin_value),
                    box_visible=True,
                    meanline_visible=False,
                    points=False,
                )
            )
        else:
            fig.add_trace(
                go.Box(
                    y=values,
                    name=str(bin_value),
                    boxpoints=False,
                )
            )
        if plot_kind in ("violin", "box_only"):
            fig.add_trace(
                go.Scatter(
                    x=[str(bin_value)],
                    y=[values.mean()],
                    mode="markers",
                    marker={"color": "black", "size": 7},
                    name=f"{bin_value} mean",
                    showlegend=False,
                )
            )

    fig.update_layout(
        title_text=title or f"{mpc_column} by {variable}",
        template="plotly_white",
        xaxis_title=variable,
        yaxis_title=mpc_column,
        xaxis={"categoryorder": "array", "categoryarray": category_order},
        showlegend=False,
    )
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))
    elif y_quantile_range is not None:
        fig.update_yaxes(range=list(y_quantile_range))
    return fig


def write_distribution_plots(
    panel: pd.DataFrame,
    output_dir: str | Path,
    *,
    mpc_column: str = "cmpc_4q",
    plot_kind: str = "violin",
    y_quantiles: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    winsor_quantiles: tuple[float, float] | None = None,
    winsor_mode: str = "clip",
    output_format: str = "html",
) -> None:
    """Write one standalone Plotly plot per binning variable."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for variable in BINNING_VARIABLES:
        fig = make_distribution_plot(
            panel,
            variable=variable,
            mpc_column=mpc_column,
            plot_kind=plot_kind,
            y_quantiles=y_quantiles,
            y_range=y_range,
            winsor_quantiles=winsor_quantiles,
            winsor_mode=winsor_mode,
        )
        _write_plot_outputs(
            fig,
            output_dir=output_path,
            stem=f"{mpc_column}_by_{variable}",
            output_format=output_format,
        )


def plot_mpc_panel(
    panel: pd.DataFrame,
    *,
    variables: Iterable[str] = BINNING_VARIABLES,
    mpc_columns: Iterable[str] = ("cmpc_4q",),
    plot_kind: str = "violin",
    y_quantiles: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    winsor_quantiles: tuple[float, float] | None = None,
    winsor_mode: str = "clip",
    output_dir: str | Path | None = None,
    output_format: str = "html",
    show: bool = True,
) -> list[go.Figure]:
    """Build (and optionally display) distribution plots for selected variables/columns.

    This is the notebook-facing counterpart to ``write_distribution_plots``: it
    returns the Plotly figures directly, while optionally writing HTML or PNG
    files when ``output_dir`` is provided, so the caller can customise the
    variable/column selection per call.
    """
    resolved_output_dir = None if output_dir is None else Path(output_dir)
    if resolved_output_dir is not None:
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for mpc_column in mpc_columns:
        for variable in variables:
            fig = make_distribution_plot(
                panel,
                variable=variable,
                mpc_column=mpc_column,
                plot_kind=plot_kind,
                y_quantiles=y_quantiles,
                y_range=y_range,
                winsor_quantiles=winsor_quantiles,
                winsor_mode=winsor_mode,
            )
            if resolved_output_dir is not None:
                _write_plot_outputs(
                    fig,
                    output_dir=resolved_output_dir,
                    stem=f"{mpc_column}_by_{variable}",
                    output_format=output_format,
                )
            if show:
                fig.show()
            figures.append(fig)
    return figures


def _write_plot_outputs(
    fig: go.Figure,
    *,
    output_dir: Path,
    stem: str,
    output_format: str,
) -> None:
    """Persist Plotly figures in the requested output format."""
    if output_format not in PLOT_OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {sorted(PLOT_OUTPUT_FORMATS)}.")
    saved_fig = go.Figure(fig)
    base_height = saved_fig.layout.height if saved_fig.layout.height is not None else SAVED_PLOT_BASE_HEIGHT
    saved_fig.update_layout(height=int(round(base_height * SAVED_PLOT_HEIGHT_SCALE)))
    if output_format in {"html", "both"}:
        saved_fig.write_html(output_dir / f"{stem}.html")
    if output_format in {"png", "both"}:
        if importlib.util.find_spec("kaleido") is None:
            raise RuntimeError(
                "PNG export requires the optional 'kaleido' package. Install it, or use output_format='html'."
            )
        saved_fig.write_image(output_dir / f"{stem}.png")
