from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import h5py
import numpy as np
import pandas as pd

PathLike = str | Path


@dataclass(frozen=True)
class ThresholdRule:
    """Threshold definition used to flag potentially problematic transitions."""

    dataset_path: str
    condition: str
    threshold: float
    label: str | None = None


DEFAULT_CORE_DATASETS = {
    "gdp_output": "/FRA/economy/gdp_output",
    "gdp_expenditure": "/FRA/economy/gdp_expenditure",
    "gdp_income": "/FRA/economy/gdp_income",
    "total_output": "/FRA/economy/total_output",
    "household_fce": "/FRA/economy/total_household_fce",
    "government_fce": "/FRA/economy/total_government_fce",
    "gfcf": "/FRA/economy/total_gross_fixed_capital_formation",
    "exports": "/FRA/economy/total_exports",
    "imports": "/FRA/economy/total_imports",
    "cpi_transaction": "/FRA/economy/cpi_transaction",
    "ppi": "/FRA/economy/ppi",
    "estimated_cpi_inflation": "/FRA/economy/estimated_cpi_inflation",
    "estimated_ppi_inflation": "/FRA/economy/estimated_ppi_inflation",
    "unemployment_rate": "/FRA/economy/unemployment_rate",
    "vacancy_rate": "/FRA/economy/vacancy_rate",
    "government_revenue": "/FRA/central_government/revenue",
    "government_deficit": "/FRA/central_government/deficit",
    "government_debt": "/FRA/central_government/debt",
    "policy_rate": "/FRA/central_Bank/policy_rate",
    "firm_insolvency_rate": "/FRA/economy/firm_insolvency_rate",
}

DEFAULT_FIRM_DATASETS = {
    "profits": "/FRA/firms/profits",
    "deposits": "/FRA/firms/deposits",
    "debt": "/FRA/firms/debt",
    "debt_installments": "/FRA/firms/debt_installments",
    "demand": "/FRA/firms/demand",
    "estimated_demand": "/FRA/firms/estimated_demand",
    "desired_labour_inputs": "/FRA/firms/desired_labour_inputs",
    "equity": "/FRA/firms/equity",
    "target_production": "/FRA/firms/target_production",
    "limiting_intermediate_inputs": "/FRA/firms/limiting_intermediate_inputs",
    "limiting_capital_inputs": "/FRA/firms/limiting_capital_inputs",
    "production": "/FRA/firms/production",
    "price": "/FRA/firms/price",
    "wage_tightness_markup": "/FRA/firms/wage_tightness_markup",
    "target_short_term_credit": "/FRA/firms/target_short_term_credit",
    "target_long_term_credit": "/FRA/firms/target_long_term_credit",
}

DEFAULT_CREDIT_DATASETS = {
    "new_short_term_loans": "/FRA/CM/total_newly_loans_granted_firms_short_term",
    "new_long_term_loans": "/FRA/CM/total_newly_loans_granted_firms_long_term",
    "outstanding_short_term_loans": "/FRA/CM/total_outstanding_loans_granted_firms_short_term",
    "outstanding_long_term_loans": "/FRA/CM/total_outstanding_loans_granted_firms_long_term",
    "bank_profits": "/FRA/banks/profits",
    "bank_equity": "/FRA/banks/equity",
    "bank_total_outstanding_loans": "/FRA/banks/total_outstanding_loans",
}

DEFAULT_THRESHOLD_RULES = (
    ThresholdRule("/FRA/economy/total_exports", "equals", 0.0, "exports_reach_zero"),
    ThresholdRule("/FRA/economy/total_gross_fixed_capital_formation", "below", 1e10, "gfcf_below_10bn"),
    ThresholdRule("/FRA/firms/deposits", "any_below", -1e12, "firm_deposits_below_minus_1tn"),
    ThresholdRule("/FRA/firms/deposits", "all_below", 0.0, "all_firm_deposits_negative"),
    ThresholdRule("/FRA/economy/gdp_output", "below", 1e12, "gdp_output_below_1tn"),
    ThresholdRule("/FRA/economy/unemployment_rate", "constant", 0.0, "unemployment_flat_window"),
)


def _ensure_path(h5_path: PathLike) -> Path:
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _slice_bounds(length: int, start: int | None, end: int | None) -> tuple[int, int]:
    start_idx = 0 if start is None else max(0, int(start))
    end_idx = length if end is None else min(length, int(end))
    if end_idx < start_idx:
        raise ValueError(f"Invalid slice bounds: start={start_idx}, end={end_idx}")
    return start_idx, end_idx


def _read_dataset_window(
    handle: h5py.File,
    dataset_path: str,
    start: int | None = None,
    end: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = handle[dataset_path]
    start_idx, end_idx = _slice_bounds(dataset.shape[0], start, end)
    values = dataset[start_idx:end_idx]
    time_index = np.arange(start_idx, end_idx)
    return time_index, np.asarray(values)


def load_series_window(
    h5_path: PathLike,
    dataset_path: str,
    start: int | None = None,
    end: int | None = None,
    reducer: str | Callable[[np.ndarray], np.ndarray] | None = "auto",
) -> pd.DataFrame:
    """Load one HDF5 dataset into a time-indexed DataFrame.

    For 2D datasets, ``reducer="auto"`` preserves columns. Pass ``"sum"``,
    ``"mean"``, ``"min"``, or ``"max"`` to aggregate by row.
    """
    path = _ensure_path(h5_path)
    with h5py.File(path, "r") as handle:
        time_index, values = _read_dataset_window(handle, dataset_path, start=start, end=end)

    if values.ndim == 1:
        return pd.DataFrame({"value": values}, index=pd.Index(time_index, name="time"))

    if values.ndim == 2 and values.shape[1] == 1:
        return pd.DataFrame({"value": values[:, 0]}, index=pd.Index(time_index, name="time"))

    if values.ndim != 2:
        raise ValueError(f"Unsupported dataset shape for {dataset_path}: {values.shape}")

    index = pd.Index(time_index, name="time")
    if reducer == "auto" or reducer is None:
        columns = [f"col_{i}" for i in range(values.shape[1])]
        return pd.DataFrame(values, index=index, columns=columns)

    if callable(reducer):
        reduced = reducer(values)
    else:
        reducers = {
            "sum": lambda x: x.sum(axis=1),
            "mean": lambda x: x.mean(axis=1),
            "min": lambda x: x.min(axis=1),
            "max": lambda x: x.max(axis=1),
        }
        if reducer not in reducers:
            raise ValueError(f"Unknown reducer: {reducer}")
        reduced = reducers[reducer](values)

    return pd.DataFrame({"value": reduced}, index=index)


def load_named_series_window(
    h5_path: PathLike,
    dataset_map: Mapping[str, str],
    start: int | None = None,
    end: int | None = None,
) -> pd.DataFrame:
    """Load multiple 1D datasets into one time-indexed DataFrame."""
    series = {}
    index = None
    for name, dataset_path in dataset_map.items():
        df = load_series_window(h5_path, dataset_path, start=start, end=end)
        if df.shape[1] != 1:
            raise ValueError(f"Dataset '{dataset_path}' is not 1D after loading.")
        if index is None:
            index = df.index
        series[name] = df.iloc[:, 0]
    return pd.DataFrame(series, index=index)


def load_entity_panel(
    h5_path: PathLike,
    dataset_map: Mapping[str, str],
    start: int | None = None,
    end: int | None = None,
    top_n: int | None = None,
    entity_indices: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Load multiple 2D datasets into a long panel indexed by time and entity."""
    path = _ensure_path(h5_path)
    with h5py.File(path, "r") as handle:
        frames = []
        for metric, dataset_path in dataset_map.items():
            time_index, values = _read_dataset_window(handle, dataset_path, start=start, end=end)
            values = np.asarray(values)
            if values.ndim == 1:
                values = values[:, None]
            n_entities = values.shape[1]
            if entity_indices is not None:
                chosen_entities = [idx for idx in entity_indices if 0 <= idx < n_entities]
            else:
                chosen_entities = list(range(n_entities))
            if top_n is not None:
                chosen_entities = chosen_entities[:top_n]
            metric_df = pd.DataFrame(values[:, chosen_entities], index=time_index, columns=chosen_entities)
            metric_df.index.name = "time"
            metric_df = metric_df.stack().rename(metric).to_frame()
            metric_df.index.set_names(["time", "entity"], inplace=True)
            frames.append(metric_df)

    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, axis=1).sort_index()
    return panel


def detect_threshold_crossings(
    h5_path: PathLike,
    rules: Iterable[ThresholdRule] = DEFAULT_THRESHOLD_RULES,
    start: int | None = None,
    end: int | None = None,
) -> pd.DataFrame:
    """Flag the first period where each threshold rule is triggered."""
    results = []
    path = _ensure_path(h5_path)
    with h5py.File(path, "r") as handle:
        for rule in rules:
            time_index, values = _read_dataset_window(handle, rule.dataset_path, start=start, end=end)
            values = np.asarray(values)
            label = rule.label or rule.dataset_path

            if rule.condition == "equals":
                mask = np.isclose(values.squeeze(), rule.threshold)
            elif rule.condition == "below":
                mask = values.squeeze() < rule.threshold
            elif rule.condition == "any_below":
                mask = (values < rule.threshold).any(axis=1)
            elif rule.condition == "all_below":
                mask = (values < rule.threshold).all(axis=1)
            elif rule.condition == "constant":
                flat = np.isclose(np.ptp(values.squeeze()), rule.threshold)
                mask = np.array([flat] * len(time_index))
            else:
                raise ValueError(f"Unsupported threshold condition: {rule.condition}")

            triggered = np.where(mask)[0]
            first_time = None if len(triggered) == 0 else int(time_index[triggered[0]])
            results.append(
                {
                    "label": label,
                    "dataset_path": rule.dataset_path,
                    "condition": rule.condition,
                    "threshold": rule.threshold,
                    "first_trigger_time": first_time,
                    "triggered": first_time is not None,
                }
            )

    return pd.DataFrame(results).sort_values(["triggered", "first_trigger_time"], ascending=[False, True])


def build_collapse_core_view(
    h5_path: PathLike,
    country_code: str = "FRA",
    start: int = 24,
    end: int = 33,
) -> pd.DataFrame:
    """Load the main economy and fiscal series used to inspect collapse dynamics."""
    dataset_map = {
        name: dataset_path.replace("/FRA/", f"/{country_code}/") for name, dataset_path in DEFAULT_CORE_DATASETS.items()
    }
    return load_named_series_window(h5_path, dataset_map, start=start, end=end + 1)


def build_firm_collapse_panel(
    h5_path: PathLike,
    country_code: str = "FRA",
    start: int = 24,
    end: int = 33,
    top_n: int = 10,
) -> pd.DataFrame:
    """Load firm-level metrics around the collapse window for side-by-side comparison."""
    dataset_map = {
        name: dataset_path.replace("/FRA/", f"/{country_code}/") for name, dataset_path in DEFAULT_FIRM_DATASETS.items()
    }
    return load_entity_panel(h5_path, dataset_map, start=start, end=end + 1, top_n=top_n)


def build_credit_collapse_view(
    h5_path: PathLike,
    country_code: str = "FRA",
    start: int = 24,
    end: int = 33,
) -> pd.DataFrame:
    """Load aggregate credit and bank series around the collapse window."""
    dataset_map = {
        name: dataset_path.replace("/FRA/", f"/{country_code}/")
        for name, dataset_path in DEFAULT_CREDIT_DATASETS.items()
    }
    return load_named_series_window(h5_path, dataset_map, start=start, end=end + 1)


def summarize_collapse_diagnostics(
    h5_path: PathLike,
    country_code: str = "FRA",
    start: int = 24,
    end: int = 33,
) -> dict[str, pd.DataFrame]:
    """Return a notebook-friendly bundle of the main collapse inspection tables."""
    resolved_rules = tuple(
        ThresholdRule(
            dataset_path=rule.dataset_path.replace("/FRA/", f"/{country_code}/"),
            condition=rule.condition,
            threshold=rule.threshold,
            label=rule.label,
        )
        for rule in DEFAULT_THRESHOLD_RULES
    )
    return {
        "core": build_collapse_core_view(h5_path, country_code=country_code, start=start, end=end),
        "firms": build_firm_collapse_panel(h5_path, country_code=country_code, start=start, end=end),
        "credit": build_credit_collapse_view(h5_path, country_code=country_code, start=start, end=end),
        "thresholds": detect_threshold_crossings(h5_path, rules=resolved_rules, start=start, end=end + 1),
    }


def _first_below(values: np.ndarray, threshold: float) -> float:
    hits = np.where(np.isfinite(values) & (values < threshold))[0]
    return float(hits[0]) if len(hits) else np.nan


def _first_zero(values: np.ndarray) -> float:
    hits = np.where(values == 0.0)[0]
    return float(hits[0]) if len(hits) else np.nan


def _first_relative_change(values: np.ndarray, threshold: float, direction: str) -> float:
    previous = np.asarray(values[:-1], dtype=float)
    current = np.asarray(values[1:], dtype=float)
    relative_change = _safe_divide(current - previous, previous)
    if direction == "drop":
        hits = np.where(np.isfinite(relative_change) & (relative_change < -threshold))[0]
    elif direction == "jump":
        hits = np.where(np.isfinite(relative_change) & (relative_change > threshold))[0]
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    return float(hits[0] + 1) if len(hits) else np.nan


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray, fill: float = np.nan) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full(np.broadcast_shapes(numerator.shape, denominator.shape), fill, dtype=float),
        where=denominator != 0.0,
    )


def _sector_value(values: np.ndarray, time: int, sector: int) -> float:
    if time >= values.shape[0] or sector >= values.shape[1]:
        return np.nan
    return float(values[time, sector])


def summarize_government_bridge_run(
    h5_path: PathLike,
    country_code: str = "FRA",
    sector: int = 15,
) -> dict[str, float]:
    """Summarize desired-vs-realised government bridge diagnostics for one HDF5 run."""
    path = _ensure_path(h5_path)
    with h5py.File(path, "r") as handle:
        base = f"/{country_code}"
        gdp = np.asarray(handle[f"{base}/economy/gdp_output"], dtype=float).squeeze()
        government_fce = np.asarray(handle[f"{base}/economy/total_government_fce"], dtype=float).squeeze()
        desired = np.asarray(handle[f"{base}/government_entities/desired_consumption_in_lcu"], dtype=float)
        realised = np.asarray(handle[f"{base}/government_entities/consumption_in_lcu"], dtype=float)
        profits = np.asarray(handle[f"{base}/firms/profits"], dtype=float).sum(axis=1)
        unemployment = np.asarray(handle[f"{base}/economy/unemployment_rate"], dtype=float).squeeze()
        vacancy = np.asarray(handle[f"{base}/economy/vacancy_rate"], dtype=float).squeeze()
        debt = np.asarray(handle[f"{base}/central_government/debt"], dtype=float).squeeze()
        prices = np.asarray(handle[f"{base}/economy/good_prices"], dtype=float)
        production = np.asarray(handle[f"{base}/firms/production"], dtype=float)
        inventory = np.asarray(handle[f"{base}/firms/inventory"], dtype=float)

    desired_total = desired.sum(axis=1)
    realised_total = realised.sum(axis=1)
    realised_desired_ratio = _safe_divide(realised_total, desired_total)
    desired_share = _safe_divide(desired, desired_total[:, None])
    government_real_demand = _safe_divide(desired, prices, fill=0.0)
    supply_cover = _safe_divide(production + inventory, government_real_demand)
    debt_gdp = _safe_divide(debt, 4.0 * gdp)
    desired_window = desired_total[1:]
    desired_cv = float(np.nanstd(desired_window) / np.nanmean(desired_window)) if np.nanmean(desired_window) else np.nan

    return {
        "desired_government_consumption_t0": float(desired_total[0]) if len(desired_total) > 0 else np.nan,
        "desired_government_consumption_t20": float(desired_total[20]) if len(desired_total) > 20 else np.nan,
        "desired_government_consumption_t21": float(desired_total[21]) if len(desired_total) > 21 else np.nan,
        "desired_government_consumption_t22": float(desired_total[22]) if len(desired_total) > 22 else np.nan,
        "desired_government_consumption_t50": float(desired_total[50]) if len(desired_total) > 50 else np.nan,
        "gdp_t50_change": float(100.0 * (gdp[-1] / gdp[0] - 1.0)),
        "government_fce_min": float(np.nanmin(government_fce)),
        "government_fce_first_below_1m": _first_below(government_fce, 1_000_000.0),
        "government_fce_first_zero": _first_zero(government_fce),
        "desired_government_consumption_first_below_1bn": _first_below(desired_total, 1_000_000_000.0),
        "desired_government_consumption_first_zero": _first_zero(desired_total),
        "desired_government_consumption_first_drop_gt_50pct": _first_relative_change(desired_total, 0.5, "drop"),
        "desired_government_consumption_first_jump_gt_50pct": _first_relative_change(desired_total, 0.5, "jump"),
        "desired_government_consumption_cv_t1_t50": desired_cv,
        "realised_desired_first_below_0_9": _first_below(realised_desired_ratio, 0.9),
        "realised_desired_first_below_0_5": _first_below(realised_desired_ratio, 0.5),
        "sector15_desired_share_t20": _sector_value(desired_share, 20, sector),
        "sector15_desired_share_t21": _sector_value(desired_share, 21, sector),
        "sector15_supply_cover_t20": _sector_value(supply_cover, 20, sector),
        "sector15_supply_cover_t21": _sector_value(supply_cover, 21, sector),
        "firm_profits_t50": float(profits[-1]),
        "unemployment_t50": float(unemployment[-1]),
        "vacancy_t50": float(vacancy[-1]),
        "debt_gdp_t50": float(debt_gdp[-1]),
    }


def compare_unemployment_to_exogenous(
    h5_path: PathLike,
    country_code: str = "FRA",
    start: int = 0,
    end: int | None = None,
) -> pd.DataFrame:
    """Compare endogenous unemployment to the stored exogenous path when available."""
    dataset_map = {
        "economy_unemployment_rate": f"/{country_code}/economy/unemployment_rate",
        "economy_vacancy_rate": f"/{country_code}/economy/vacancy_rate",
    }
    df = load_named_series_window(h5_path, dataset_map, start=start, end=end)

    path = _ensure_path(h5_path)
    with h5py.File(path, "r") as handle:
        exog_ur_path = f"/{country_code}/exogenous/unemployment_rate"
        exog_vr_path = f"/{country_code}/exogenous/vacancy_rate"
        if exog_ur_path in handle:
            exog_ur = load_series_window(h5_path, exog_ur_path, start=start, end=end)
            df["exogenous_unemployment_rate"] = exog_ur.iloc[:, 0].reindex(df.index)
        if exog_vr_path in handle:
            exog_vr = load_series_window(h5_path, exog_vr_path, start=start, end=end)
            df["exogenous_vacancy_rate"] = exog_vr.iloc[:, 0].reindex(df.index)

    return df


def first_entity_drop_below_threshold(
    h5_path: PathLike,
    dataset_path: str,
    threshold: float,
    start: int | None = None,
    end: int | None = None,
    country_code: str = "FRA",
) -> pd.DataFrame:
    """Return the first time each entity drops below a threshold."""
    resolved_path = dataset_path.replace("/FRA/", f"/{country_code}/")
    panel = load_series_window(h5_path, resolved_path, start=start, end=end, reducer="auto")
    records = []
    for column in panel.columns:
        below = np.where(panel[column].to_numpy() < threshold)[0]
        first_time = None if len(below) == 0 else int(panel.index[below[0]])
        records.append({"entity": column, "first_time_below_threshold": first_time, "threshold": threshold})
    return pd.DataFrame(records).sort_values("first_time_below_threshold", na_position="last")
