"""Household support mode and minimum-income validation metrics from saved runs.

Two readers over a saved HDF5 file, with no model loaded:

- ``resolve_household_support`` returns the run's support mode (``legacy_stage5`` or ``minimum_income``) and
  whether the compulsory goods floor was on. It reads the configuration saved with the run and only infers
  from the series when that configuration is absent.
- ``minimum_income_metrics`` computes the PLAN section 8 incidence metrics by income quintile.

Money is nominal, in model monetary units, per model period. Shares are unweighted household-periods.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SUPPORT_LEGACY = "legacy_stage5"
SUPPORT_MINIMUM_INCOME = "minimum_income"
QUINTILES = ("Q1", "Q2", "Q3", "Q4", "Q5")
RE_ENTRY_WINDOW = 4


def _parse_household_support(configured: Mapping) -> dict:
    """Apply the configuration defaults: legacy mode with the goods floor on."""
    mode = configured.get("mode", SUPPORT_LEGACY)
    if mode not in (SUPPORT_LEGACY, SUPPORT_MINIMUM_INCOME):
        raise ValueError(f"Unknown household_support.mode: {mode!r}.")
    return {"mode": mode, "goods_floor": bool(configured.get("goods_floor", True))}


def resolve_household_support(
    h5_path: str | Path, country_code: str, configured: Mapping | None = None
) -> dict[str, object]:
    """Return ``{"mode", "goods_floor", "source"}`` for one saved run.

    Order of precedence: the ``household_support`` mapping passed by the caller (``source`` "argument"), the
    ``configuration`` attribute saved in the HDF5 file ("saved_configuration"), then inference ("inferred").
    Inference calls a run ``minimum_income`` only when it booked a positive top-up and never paid Stage 5
    support. It cannot see the goods-floor switch, so it assumes the floor is on.
    """
    if configured is not None:
        return {**_parse_household_support(configured), "source": "argument"}
    with h5py.File(h5_path, "r") as handle:
        saved = handle.attrs.get("configuration")
        if isinstance(saved, bytes):
            saved = saved.decode()
        try:
            countries = ast.literal_eval(saved)["country_configurations"] if saved else {}
            block = countries[country_code]["household_support"]
        except (ValueError, SyntaxError, KeyError, TypeError):
            block = None
        if block is not None:
            return {**_parse_household_support(block), "source": "saved_configuration"}
        group = handle.get(f"{country_code}/households")
        mode = SUPPORT_LEGACY
        if group is not None and "income_minimum_income" in group:
            booked = np.asarray(group["income_minimum_income"], dtype=float)
            stage5 = (
                np.asarray(group["stage5_subsistence_support"], dtype=float)
                if "stage5_subsistence_support" in group
                else 0.0
            )
            if np.any(booked > 0.0) and not np.any(stage5 != 0.0):
                mode = SUPPORT_MINIMUM_INCOME
    return {"mode": mode, "goods_floor": True, "source": "inferred"}


def _income_quintiles(income: np.ndarray) -> np.ndarray:
    """Quintile index 0..4 of each household within each period (rows)."""
    quintile = np.zeros(income.shape, dtype=int)
    for row, values in enumerate(income):
        quintile[row] = np.searchsorted(np.quantile(values, [0.2, 0.4, 0.6, 0.8]), values, side="right")
    return quintile


def _re_entries(social: np.ndarray, first_period: int) -> tuple[np.ndarray, np.ndarray]:
    """Social-housing entries and re-entries within RE_ENTRY_WINDOW periods of leaving, by period and household.

    Row ``t`` of the result describes the move between periods ``t-1`` and ``t``, so it starts at ``first_period``.
    The exit history is read from row 0, so a household that left just before ``first_period`` still counts.
    """
    entries = np.zeros(social.shape, dtype=bool)
    re_entries = np.zeros(social.shape, dtype=bool)
    last_exit = np.full(social.shape[1], -(10**9))
    for t in range(1, social.shape[0]):
        entered = social[t] & ~social[t - 1]
        if t >= first_period:
            entries[t] = entered
            re_entries[t] = entered & (t - last_exit <= RE_ENTRY_WINDOW)
        last_exit[~social[t] & social[t - 1]] = t
    return entries, re_entries


def minimum_income_metrics(h5_path: str | Path, country_code: str, *, first_period: int = 1) -> pd.DataFrame:
    """PLAN section 8 metrics by income quintile for one saved run (rows Q1..Q5 and "all").

    Households are ranked in each period by cash income before any support
    (``income - income_minimum_income - stage5_subsistence_support``), so that recipients are not moved up by the
    transfer. Quintile cuts use the quintile of the period in which the event is counted. Periods
    ``first_period..end`` are used; row 0 is the initialisation row. A metric whose series is not in the file is NaN.

    Columns:
    - ``share_receiving_T``: share of household-periods with a booked top-up T > 0;
    - ``mean_T_over_M_recipients``: mean of T / M over those household-periods;
    - ``fiscal_cost_pct_gdp``: sum of T over the group, per period, as % of nominal GDP (``gdp_output``),
      averaged over periods; the quintile values add up to the "all" row;
    - ``guard_binding_share``: share with ``income_guard_bound`` set (in legacy mode the series is the legacy
      subsistence floor, so it measures how often that floor binds);
    - ``unpaid_rent_events``: household-periods with unpaid rent > 0 (count);
    - ``social_housing_entries``: moves into social housing (count);
    - ``re_entries_within_4q``: entries within 4 periods of a previous exit (count; periods are quarters);
    - ``unmet_floor_share``: share with unmet goods floor > 0.
    """
    if first_period < 1:
        raise ValueError("first_period must be at least 1; row 0 is the initialisation row.")
    with h5py.File(h5_path, "r") as handle:
        households = handle[f"{country_code}/households"]

        def household(name):
            return np.asarray(households[name], dtype=float) if name in households else None

        income = household("income")
        top_up = household("income_minimum_income")
        stage5 = household("stage5_subsistence_support")
        level = household("minimum_income_level")
        guard = household("income_guard_bound")
        unpaid = household("unpaid_rent")
        social = household("social_housing_tenant")
        unmet = household("unmet_floor")
        gdp_path = f"{country_code}/economy/gdp_output"
        gdp = np.asarray(handle[gdp_path], dtype=float).reshape(-1) if gdp_path in handle else None
    if income is None or top_up is None or level is None:
        raise ValueError("income, income_minimum_income and minimum_income_level are required.")
    n_periods = income.shape[0]
    if first_period >= n_periods:
        raise ValueError("first_period is beyond the saved rows.")
    rows = slice(first_period, None)
    quintile = _income_quintiles(income - top_up - (0.0 if stage5 is None else stage5))[rows]
    top_up_rows, level_rows = top_up[rows], level[rows]
    gdp_rows = None if gdp is None else gdp[rows]
    recipient = top_up_rows > 0.0
    ratio = np.divide(
        top_up_rows, level_rows, out=np.full(top_up_rows.shape, np.nan), where=recipient & (level_rows > 0.0)
    )
    if social is not None:
        entries, re_entries = _re_entries(social.astype(bool), first_period)
        entries, re_entries = entries[rows], re_entries[rows]
    flags = {
        "guard_binding_share": None if guard is None else guard[rows] != 0.0,
        "unpaid_rent_events": None if unpaid is None else unpaid[rows] > 0.0,
        "social_housing_entries": None if social is None else entries,
        "re_entries_within_4q": None if social is None else re_entries,
        "unmet_floor_share": None if unmet is None else unmet[rows] > 0.0,
    }
    counts = {"unpaid_rent_events", "social_housing_entries", "re_entries_within_4q"}

    def summarise(mask: np.ndarray) -> dict[str, float]:
        out = {"n_household_periods": float(mask.sum()), "share_receiving_T": recipient[mask].mean()}
        in_group = mask & recipient
        out["mean_T_over_M_recipients"] = ratio[in_group].mean() if in_group.any() else np.nan
        if gdp_rows is None:
            out["fiscal_cost_pct_gdp"] = np.nan
        else:
            out["fiscal_cost_pct_gdp"] = 100 * np.mean((top_up_rows * mask).sum(axis=1) / gdp_rows)
        for name, flag in flags.items():
            if flag is None:
                out[name] = np.nan
            else:
                out[name] = float(flag[mask].sum()) if name in counts else flag[mask].mean()
        return out

    table = {label: summarise(quintile == i) for i, label in enumerate(QUINTILES)}
    table["all"] = summarise(np.ones(quintile.shape, dtype=bool))
    return pd.DataFrame(table).T
