"""Map simulation state into common permanent-income forecast regressors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from macro_data.readers.permanent_income_forecast import (
    RAW_VARIABLE_ORDER,
    SIMULATION_VARIABLE_NAME_MAP,
)
from macro_data.util.rates import fisher_real_rate

NOTEBOOK_EXPORT_TO_FORECAST_READER_NAME = SIMULATION_VARIABLE_NAME_MAP
FORECAST_READER_TO_NOTEBOOK_EXPORT_NAME = {
    forecast_reader_name: notebook_name
    for notebook_name, forecast_reader_name in NOTEBOOK_EXPORT_TO_FORECAST_READER_NAME.items()
}
FORECAST_READER_VARIABLE_ORDER = [
    NOTEBOOK_EXPORT_TO_FORECAST_READER_NAME[notebook_name] for notebook_name in RAW_VARIABLE_ORDER
]

FORECAST_READER_TO_SIMULATION_SOURCE_NAME = {
    "constant": "frozen_design_matrix_initial_period",
    "time_trend": "simulation_period_index",
    "split_trend_from_2009q4_discounted_present_value": "frozen_design_matrix_initial_period",
    "covid19": "excluded_stage_3_dummy",
    "log_real_pc_income": "real_pc_income",
    "d4_log_real_pc_income": "real_pc_income",
    "log_working_age_population_share": "frozen_design_matrix_initial_period",
    "survey_expectations_ma4_l1": "frozen_design_matrix_initial_period",
    "real_interest_rate_ma4_l1": "policy_rate_and_cpi_fixed_basket",
    "real_interest_rate_ma4_l5": "policy_rate_and_cpi_fixed_basket",
    "real_interest_rate_ma4_l9": "policy_rate_and_cpi_fixed_basket",
    "d4_t_bill_rate": "frozen_design_matrix_initial_period",
    "d4_t_bill_rate_l1": "frozen_design_matrix_initial_period",
    "log_stock_market_index_ma4_l1": "frozen_design_matrix_initial_period",
    "unemp_rate_ma4_l1": "unemployment_rate",
    "unemp_rate_ma4_l5": "unemployment_rate",
    "log_real_oil_price_ma4_l1": "frozen_design_matrix_initial_period",
    "log_real_oil_price_ma4_l5": "frozen_design_matrix_initial_period",
    "log_real_fx_rate_ma4_l1": "frozen_design_matrix_initial_period",
    "log_real_fx_rate_ma4_l5": "frozen_design_matrix_initial_period",
}

# d4_t_bill_rate/d4_t_bill_rate_l1 are frozen here too, unlike the other entries
# which lack a model concept entirely. If an endogenous t-bill proxy is added
# later, it will need a dedicated lookup rather than this static initial-period read.
FROZEN_EXOGENOUS_FORECAST_READER_NAMES = tuple(
    name
    for name, source in FORECAST_READER_TO_SIMULATION_SOURCE_NAME.items()
    if source == "frozen_design_matrix_initial_period"
)


@dataclass(frozen=True)
class PermanentIncomeSimulationSources:
    """Simulation-source inputs used by the Stage 3 X-mapping helper.

    The history arrays must be quarterly and ordered from the simulation's first
    quarter through ``current_period``. ``real_pc_income`` must already be the
    notebook/export real per-capita income index scale. Policy rates are model
    decimal rates; unemployment rates are shares on ``[0, 1]``, matching
    ``Economy.ts.unemployment_rate``, and are converted to percentage points
    internally for the ``unemp_rate_ma4_*`` regressors.
    """

    current_period: str | pd.Period | pd.Timestamp
    real_pc_income: Sequence[float]
    policy_rate: Sequence[float]
    cpi_fixed_basket: Sequence[float]
    unemployment_rate: Sequence[float]


def _quarter_period(value: str | int | pd.Period | pd.Timestamp) -> pd.Period:
    if isinstance(value, pd.Period):
        return value.asfreq("Q")
    if isinstance(value, int):
        return pd.Period(f"{value}Q1", freq="Q")
    return pd.Period(value, freq="Q")


def _quarter_index(period: pd.Period, start_period: str | int | pd.Period | pd.Timestamp) -> int:
    start = _quarter_period(start_period)
    return (period.year - start.year) * 4 + (period.quarter - start.quarter)


def _periods_for_history(current_period: pd.Period, history_length: int) -> pd.PeriodIndex:
    if history_length <= 0:
        raise ValueError("Simulation source histories must contain at least one quarter.")
    return pd.period_range(end=current_period, periods=history_length, freq="Q")


def design_matrix_to_forecast_reader_names(design_matrix: pd.DataFrame) -> pd.DataFrame:
    """Return a design matrix whose columns use canonical forecast-reader names."""
    out = design_matrix.copy()
    rename_map = {
        notebook_name: forecast_reader_name
        for notebook_name, forecast_reader_name in NOTEBOOK_EXPORT_TO_FORECAST_READER_NAME.items()
        if notebook_name in out.columns
    }
    out = out.rename(columns=rename_map)

    if "date" in out.columns:
        periods = pd.to_datetime(out["date"]).dt.to_period("Q")
        out = out.drop(columns=["date"])
        out.index = periods
    elif not isinstance(out.index, pd.PeriodIndex):
        out.index = pd.to_datetime(out.index).to_period("Q")
    else:
        out.index = out.index.asfreq("Q")

    return out.sort_index()


def _design_row(design: pd.DataFrame, period: pd.Period) -> pd.Series:
    if period not in design.index:
        raise ValueError(f"Design matrix does not contain required quarter {period}.")
    return design.loc[period]


def _design_row_or_nearest(design: pd.DataFrame, period: pd.Period) -> pd.Series:
    """Look up a design-matrix row, clamping to the nearest available period.

    The design matrix is a fixed historical export with finite coverage, while
    ``period`` (derived from the simulation clock) grows without bound. Periods
    past either end of ``design.index`` are clamped to the first/last available
    row so design-matrix-only regressors freeze at their boundary values instead
    of raising once the simulation runs longer than the export's coverage.
    """
    if period in design.index:
        return design.loc[period]
    if period > design.index.max():
        return design.loc[design.index.max()]
    return design.loc[design.index.min()]


def _as_1d_float_history(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} history must contain at least one quarter.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} history contains non-finite values.")
    return arr


def _log_income_index_history(real_pc_income: np.ndarray) -> np.ndarray:
    if np.any(real_pc_income <= 0.0):
        raise ValueError("real_pc_income index history must be strictly positive.")
    return np.log(real_pc_income)


def _fisher_real_short_rate_percent(
    policy_rate: np.ndarray,
    cpi_fixed_basket: np.ndarray,
    *,
    policy_rate_periods_per_year: int,
) -> np.ndarray:
    if policy_rate_periods_per_year <= 0:
        raise ValueError("policy_rate_periods_per_year must be positive.")
    if policy_rate.size != cpi_fixed_basket.size:
        raise ValueError("policy_rate and cpi_fixed_basket histories must have matching lengths.")
    if np.any(cpi_fixed_basket <= 0.0):
        raise ValueError("cpi_fixed_basket history must be strictly positive.")
    if cpi_fixed_basket.size < 5:
        return np.array([], dtype=float)

    annual_nominal_rate = policy_rate[4:] * float(policy_rate_periods_per_year)
    yoy_inflation = cpi_fixed_basket[4:] / cpi_fixed_basket[:-4] - 1.0
    return 100.0 * fisher_real_rate(annual_nominal_rate, yoy_inflation)


def _ma4_lag_from_history(values: np.ndarray, lag: int) -> float | None:
    end_exclusive = values.size - lag
    start = end_exclusive - 4
    if start < 0 or end_exclusive > values.size:
        return None
    return float(values[start:end_exclusive].mean())


def _unemployment_percent_history_from_share(unemployment_rate: np.ndarray) -> np.ndarray:
    if np.any((unemployment_rate < 0.0) | (unemployment_rate > 1.0)):
        raise ValueError("unemployment_rate history must be shares on [0, 1].")
    return unemployment_rate * 100.0


def build_permanent_income_forecast_regressors(
    *,
    sources: PermanentIncomeSimulationSources,
    design_matrix: pd.DataFrame,
    start_period: str | int | pd.Period | pd.Timestamp,
    estimation_epoch: str | pd.Period | pd.Timestamp | None = None,
    policy_rate_periods_per_year: int = 4,
) -> pd.Series:
    """Build canonical forecast-reader regressors from simulation-source values.

    Endogenous lagged terms use the design matrix during warm-up and switch to
    simulation history only once the required lag window is fully available.
    ``start_period`` anchors the X timeline and trend only; income histories are
    expected to already use the notebook/export index convention.

    ``estimation_epoch`` sets the origin for the ``time_trend`` counter.  When
    provided it overrides ``start_period`` for that purpose only, allowing the
    trend to be anchored to the estimation sample's epoch (e.g. 1980Q1) rather
    than the simulation's start.  Defaults to ``None``, in which case
    ``start_period`` is used as the anchor (preserving the prior behaviour for
    all existing call sites and tests).
    """
    current_period = _quarter_period(sources.current_period)
    real_pc_income = _as_1d_float_history(sources.real_pc_income, "real_pc_income")
    policy_rate = _as_1d_float_history(sources.policy_rate, "policy_rate")
    cpi_fixed_basket = _as_1d_float_history(sources.cpi_fixed_basket, "cpi_fixed_basket")
    unemployment_rate = _as_1d_float_history(sources.unemployment_rate, "unemployment_rate")

    history_length = real_pc_income.size
    for name, arr in {
        "policy_rate": policy_rate,
        "cpi_fixed_basket": cpi_fixed_basket,
        "unemployment_rate": unemployment_rate,
    }.items():
        if arr.size != history_length:
            raise ValueError(f"{name} history length {arr.size} does not match real_pc_income length {history_length}.")

    periods = _periods_for_history(current_period, history_length)
    initial_period = periods[0]
    design = design_matrix_to_forecast_reader_names(design_matrix)
    current_design = _design_row_or_nearest(design, current_period)
    initial_design = _design_row_or_nearest(design, initial_period)
    log_income_history = _log_income_index_history(real_pc_income)

    epoch = _quarter_period(estimation_epoch) if estimation_epoch is not None else _quarter_period(start_period)

    values: dict[str, float] = {name: float(current_design[name]) for name in FORECAST_READER_VARIABLE_ORDER}
    values["time_trend"] = float(_quarter_index(current_period, epoch))
    values["covid19"] = 0.0

    values["log_real_pc_income"] = float(log_income_history[-1])
    if history_length >= 5:
        values["d4_log_real_pc_income"] = float(log_income_history[-1] - log_income_history[-5])

    real_short_rate = _fisher_real_short_rate_percent(
        policy_rate,
        cpi_fixed_basket,
        policy_rate_periods_per_year=policy_rate_periods_per_year,
    )
    for lag, name in [
        (1, "real_interest_rate_ma4_l1"),
        (5, "real_interest_rate_ma4_l5"),
        (9, "real_interest_rate_ma4_l9"),
    ]:
        endogenous_value = _ma4_lag_from_history(real_short_rate, lag)
        if endogenous_value is not None:
            values[name] = endogenous_value

    unemployment_percent = _unemployment_percent_history_from_share(unemployment_rate)
    for lag, name in [(1, "unemp_rate_ma4_l1"), (5, "unemp_rate_ma4_l5")]:
        endogenous_value = _ma4_lag_from_history(unemployment_percent, lag)
        if endogenous_value is not None:
            values[name] = endogenous_value

    for name in FROZEN_EXOGENOUS_FORECAST_READER_NAMES:
        values[name] = float(initial_design[name])

    return pd.Series(values, index=FORECAST_READER_VARIABLE_ORDER, dtype=float, name=current_period)


def load_permanent_income_design_matrix(path: str | Path) -> pd.DataFrame:
    """Load the raw permanent-income design matrix export.

    Exported from ``consumption-notebooks/data/pi_est/FR/design_matrix.csv``
    (``permanent_income_estimation.ipynb``). Returns the design matrix with
    notebook/export column names and a ``date`` column intact, ready to be
    passed into :func:`design_matrix_to_forecast_reader_names`.
    """
    df = pd.read_csv(path)
    unnamed_columns = [col for col in df.columns if str(col).startswith("Unnamed")]
    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)
    if "const" not in df.columns:
        df["const"] = 1.0
    return df


def rebase_real_pc_income_index(
    real_pc_income_levels: np.ndarray,
    base_period_index: int,
    index_base_value: float = 100.0,
) -> np.ndarray:
    """Rebase a real per-capita income level history to the notebook's index convention.

    Implements ``index_base_value * real_pc / real_pc[base_period_index]``.
    """
    levels = np.asarray(real_pc_income_levels, dtype=float)
    if levels.ndim != 1 or levels.size == 0:
        raise ValueError("real_pc_income_levels must be a non-empty 1D array.")
    base_value = levels[base_period_index]
    if not np.isfinite(base_value) or base_value <= 0.0:
        raise ValueError("real_pc_income_levels base period value must be finite and positive.")
    return index_base_value * levels / base_value
