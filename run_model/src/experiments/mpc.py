"""Run paired baseline-vs-shock household MPC experiments.

This workflow is intentionally separate from Monte Carlo because the
current Monte Carlo helper does not expose a hook-registration point. The runner
instantiates each simulation directly, registers the household income prehook
only on the shock run, saves paired HDF5 files, and then delegates MPC
extraction/plotting to ``src.mpc_analysis``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

RUN_MODEL_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = RUN_MODEL_DIR.parent
for path in (str(REPO_ROOT), str(RUN_MODEL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.mpc_analysis import (  # noqa: E402
    HOUSEHOLD_LABOR_STATE_COLUMNS,
    MPC_PLOT_MEASURES,
    MPCFilterConfig,
    add_mpc_bins,
    build_household_mpc_panel,
    filter_mpc_panel,
    make_household_labor_state_snapshot,
    make_household_metadata,
    period_to_year_month,
    summarize_mpc_bins,
)
from src.notebook_workflow import NotebookRunConfig, build_country_config, prepare_data  # noqa: E402

from macro_data import DataWrapper  # noqa: E402
from macro_data.readers.default_readers import DataPaths  # noqa: E402
from macromodel.configurations import CountryConfiguration, SimulationConfiguration  # noqa: E402
from macromodel.simulation import Simulation  # noqa: E402
from macromodel.utils.prehooks.household_income_shock import create_household_income_shock_hook  # noqa: E402

DEFAULT_SEEDS = [12, 13, 14]
DEFAULT_T_MAX = 50
DEFAULT_SHOCK_PERIOD = 20
DEFAULT_HORIZON_PERIODS = 4
DEFAULT_SHOCK_FRACTION = 0.01
STATE_BIN_ROUND_DECIMALS = 8


def _resolve_run_model_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return RUN_MODEL_DIR / resolved


def _validate_unique_seeds(seeds: list[int]) -> list[int]:
    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise argparse.ArgumentTypeError("At least one seed must be provided.")
    if len(set(seed_list)) != len(seed_list):
        raise argparse.ArgumentTypeError("Seeds must be unique.")
    return seed_list


def _build_run_config(
    *,
    seed: int,
    t_max: int,
    country_iso3: str | None,
    raw_data_path: str | Path | None,
    output_dir: str | Path,
    config_dir: str | Path | None,
    force_rebuild_data: bool,
    single_hfcs_survey: bool,
) -> NotebookRunConfig:
    """Build the notebook workflow config used for data/config preparation."""
    return NotebookRunConfig(
        seed=int(seed),
        t_max=int(t_max),
        country_iso3=country_iso3,
        raw_data_path=raw_data_path,
        output_dir=output_dir,
        config_dir=config_dir,
        force_rebuild_data=force_rebuild_data,
        single_hfcs_survey=single_hfcs_survey,
    )


def _prepare_mpc_inputs(
    *,
    seed: int,
    t_max: int,
    country_iso3: str | None,
    raw_data_path: str | Path | None,
    output_dir: str | Path,
    config_dir: str | Path | None,
    force_rebuild_data: bool,
    single_hfcs_survey: bool,
) -> tuple[NotebookRunConfig, DataWrapper, dict[str, CountryConfiguration], str, Path]:
    """Reuse notebook data preparation and country-config alignment for MPC runs."""
    run_config = _build_run_config(
        seed=seed,
        t_max=t_max,
        country_iso3=country_iso3,
        raw_data_path=raw_data_path,
        output_dir=output_dir,
        config_dir=config_dir,
        force_rebuild_data=force_rebuild_data,
        single_hfcs_survey=single_hfcs_survey,
    )
    prepared = prepare_data(run_config)
    country_configurations = build_country_config(data=prepared.data, config=run_config)
    country_cfg = country_configurations[prepared.cfg.country_iso3]
    if country_cfg.assume_zero_growth:
        raise ValueError("MPC experiments require assume_zero_growth=False because target consumption ignores income.")
    return run_config, prepared.data, country_configurations, prepared.cfg.country_iso3, prepared.raw_data_path


def _run_one(
    *,
    data: DataWrapper,
    country_configurations: dict[str, CountryConfiguration],
    country_code: str,
    seed: int,
    t_max: int,
    output_dir: Path,
    shock: bool,
    shock_period: int,
    shock_fraction: float,
    horizon_periods: int,
    data_paths: DataPaths | None = None,
) -> pd.DataFrame:
    """Run one baseline or shock simulation and return pre-shock metadata.

    ``shock_period`` is a zero-based simulation iteration. Since HDF5 row ``0``
    is the initial state, metadata for binning is captured at HDF5 row
    ``shock_period``: the state immediately before the shock affects realised row
    ``shock_period + 1``.

    Household labour-state snapshots are also captured at every HDF5 row in the
    MPC measurement window (``shock_period`` through ``shock_period +
    horizon_periods``): activity bracket, aggregated worked hours, and
    household labour income. These are live model states, not fully saved HDF5
    series, so they must be captured via posthook while the simulation runs.
    """
    configuration = SimulationConfiguration.model_validate(
        {
            "country_configurations": {code: config.model_dump() for code, config in country_configurations.items()},
            "t_max": int(t_max),
            "seed": int(seed),
        }
    )
    model = Simulation.from_datawrapper(datawrapper=data, simulation_configuration=configuration, data_paths=data_paths)

    pre_shock_row = int(shock_period)
    metadata: list[pd.DataFrame] = []
    window_state_columns: dict[str, dict[str, np.ndarray]] = {
        state_name: {} for state_name in HOUSEHOLD_LABOR_STATE_COLUMNS
    }

    def capture_household_labor_state(country, *, offset: int) -> None:
        snapshot = make_household_labor_state_snapshot(country)
        for state_name, values in snapshot.items():
            window_state_columns[state_name][f"{state_name}_t{offset}"] = values

    def capture_window_metadata(simulation: Simulation, t: int, _year: int, _month: int) -> None:
        # Posthooks see the realised row for iteration ``t``. ``offset`` counts
        # periods from the pre-shock row (offset 0) through the end of the
        # cumulative-MPC horizon (offset horizon_periods).
        offset = t - (shock_period - 1)
        if offset < 0 or offset > horizon_periods:
            return
        country = simulation.countries[country_code]
        if offset == 0:
            metadata.append(make_household_metadata(country, seed=seed, pre_shock_row=pre_shock_row))
        capture_household_labor_state(country, offset=offset)

    if shock_period == 0:
        country = model.countries[country_code]
        metadata.append(make_household_metadata(country, seed=seed, pre_shock_row=0))
        capture_household_labor_state(country, offset=0)
        if horizon_periods > 0:
            model.posthooks.append(capture_window_metadata)
    else:
        model.posthooks.append(capture_window_metadata)

    if shock:
        target_year, target_month = period_to_year_month(data.configuration.year, data.time_unit, shock_period)
        model.prehooks.append(
            create_household_income_shock_hook(
                country_code=country_code,
                target_year=target_year,
                target_month=target_month,
                shock_fraction_of_median_income=shock_fraction,
            )
        )

    model.run()
    if not metadata:
        raise ValueError("Pre-shock metadata was not captured; check shock_period and t_max.")
    for state_name, state_columns in window_state_columns.items():
        if len(state_columns) != horizon_periods + 1:
            raise ValueError(
                f"Expected {horizon_periods + 1} {state_name} snapshots (t0..t{horizon_periods}); "
                f"captured {len(state_columns)}. Check shock_period, horizon_periods, and t_max."
            )

    result = metadata[0]
    for state_columns in window_state_columns.values():
        for column, values in state_columns.items():
            result[column] = values

    seed_dir = output_dir / f"seed-{int(seed)}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    model.save(save_dir=seed_dir, file_name="multi_country_simulation.h5")
    return result


def _run_seed_pair(
    *,
    data: DataWrapper,
    country_configurations: dict[str, CountryConfiguration],
    country_code: str,
    seed: int,
    t_max: int,
    baseline_dir: Path,
    shock_dir: Path,
    shock_period: int,
    shock_fraction: float,
    horizon_periods: int,
    data_paths: DataPaths | None = None,
) -> pd.DataFrame:
    """Run the baseline and shock simulations for one seed.

    Labour-state columns from the shock run are merged in as
    ``*_shock_t*`` alongside the baseline run's ``*_t*`` columns so callers can
    detect households whose activity bracket, worked hours, or labour income
    changes in either the factual or counterfactual path during the MPC window.
    """
    baseline_metadata = _run_one(
        data=data,
        country_configurations=country_configurations,
        country_code=country_code,
        seed=seed,
        t_max=t_max,
        output_dir=baseline_dir,
        shock=False,
        shock_period=shock_period,
        shock_fraction=shock_fraction,
        horizon_periods=horizon_periods,
        data_paths=data_paths,
    )
    shock_metadata = _run_one(
        data=data,
        country_configurations=country_configurations,
        country_code=country_code,
        seed=seed,
        t_max=t_max,
        output_dir=shock_dir,
        shock=True,
        shock_period=shock_period,
        shock_fraction=shock_fraction,
        horizon_periods=horizon_periods,
        data_paths=data_paths,
    )
    shock_state_columns = [
        column
        for column in shock_metadata.columns
        if any(column.startswith(f"{state_name}_t") for state_name in HOUSEHOLD_LABOR_STATE_COLUMNS)
    ]
    shock_state_metadata = shock_metadata[["household_id", *shock_state_columns]].rename(
        columns={
            column: f"{column.split('_t', maxsplit=1)[0]}_shock_t{column.split('_t', maxsplit=1)[1]}"
            for column in shock_state_columns
        }
    )
    return baseline_metadata.merge(shock_state_metadata, on="household_id", how="left")


def _resolve_identification_filter_mode(
    *,
    stay_in_activity_bracket_only: bool,
    stable_effective_labor_state_only: bool,
) -> str:
    """Return the active opt-in MPC identification filter mode."""
    if stay_in_activity_bracket_only and stable_effective_labor_state_only:
        raise ValueError(
            "Choose at most one MPC identification filter: "
            "--stay-in-activity-bracket-only or --stable-effective-labor-state-only."
        )
    if stable_effective_labor_state_only:
        return "effective_labor_state"
    if stay_in_activity_bracket_only:
        return "activity_bracket"
    return "none"


def _snapshot_columns(panel: pd.DataFrame, *, prefix: str, mode_label: str) -> list[str]:
    columns = sorted(column for column in panel.columns if column.startswith(prefix))
    if not columns:
        raise KeyError(
            f"No columns starting with {prefix!r} were found in the MPC panel. "
            f"Add the required {mode_label} state snapshots upstream, then re-run the MPC experiment."
        )
    return columns


def _rounded_global_bin_codes(frame: pd.DataFrame, *, round_decimals: int) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="coerce").round(round_decimals)
    stacked = numeric.stack().dropna()
    if stacked.empty:
        raise ValueError("Cannot derive MPC state bins from entirely missing state snapshots.")
    value_map = {value: code for code, value in enumerate(np.sort(stacked.unique()))}
    return numeric.apply(lambda column: column.map(value_map).astype("Int64"))


def _keep_households_staying_in_activity_bracket(
    panel: pd.DataFrame, activity_bracket_prefix: str = "activity_bracket_"
) -> pd.DataFrame:
    """Keep only households whose activity bracket is constant across all matched columns.

    This expects the panel to carry one or more period-by-period activity
    bracket codes with a shared prefix, for example ``activity_bracket_t0``,
    ``activity_bracket_t1``, ``activity_bracket_shock_t0``, ... covering both
    the baseline and shock runs. A household is "stable" if its bracket
    (employed, unemployed, investor, or not economically active) never
    switches across any of those columns; stable brackets other than employed
    are kept too, since their MPC is still a valid data point. Only switchers
    are dropped, to avoid attributing activity-bracket transition effects to
    the income-shock MPC. This is a stricter filter than employment status
    alone, since wage/hours-relevant transitions between, say, unemployed and
    not-economically-active are also excluded. If no such columns are present,
    raise an error so the caller knows the activity-bracket path needs to be
    added upstream (typically in ``build_household_mpc_panel`` or the metadata
    builder).
    """
    activity_bracket_cols = _snapshot_columns(
        panel,
        prefix=activity_bracket_prefix,
        mode_label="activity-bracket",
    )

    status = panel[activity_bracket_cols].fillna(-1).astype(int)
    stay_mask = status.nunique(axis=1).eq(1) & status.iloc[:, 0].ne(-1)
    return panel.loc[stay_mask].copy()


def _keep_households_in_stable_effective_labor_state(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep households stable in activity, worked hours, and labour income.

    Identification keeps only households whose labour micro-state is unchanged
    across baseline and shock paths over the MPC window, so the estimated
    consumption response is not contaminated by endogenous labour adjustment.

    Worked-hours and labour-income bins are exact rounded-value state bins built
    from the full baseline-plus-shock MPC window. This is intentionally stricter
    than quantile binning: any meaningful within-window labour-state change gets
    a different bin code and the household is excluded.
    """
    activity_columns = _snapshot_columns(
        panel,
        prefix="activity_bracket_",
        mode_label="effective-labor-state",
    )
    worked_hours_columns = _snapshot_columns(
        panel,
        prefix="worked_hours_",
        mode_label="effective-labor-state",
    )
    labor_income_columns = _snapshot_columns(
        panel,
        prefix="labor_income_",
        mode_label="effective-labor-state",
    )

    result = panel.copy()
    worked_hours_bins = _rounded_global_bin_codes(
        result[worked_hours_columns],
        round_decimals=STATE_BIN_ROUND_DECIMALS,
    )
    labor_income_bins = _rounded_global_bin_codes(
        result[labor_income_columns],
        round_decimals=STATE_BIN_ROUND_DECIMALS,
    )
    result[[f"{column}_bin" for column in worked_hours_columns]] = worked_hours_bins.set_axis(
        [f"{column}_bin" for column in worked_hours_columns],
        axis=1,
    )
    result[[f"{column}_bin" for column in labor_income_columns]] = labor_income_bins.set_axis(
        [f"{column}_bin" for column in labor_income_columns],
        axis=1,
    )

    activity_state = result[activity_columns].fillna(-1).astype(int)
    worked_hours_state = result[[f"{column}_bin" for column in worked_hours_columns]]
    labor_income_state = result[[f"{column}_bin" for column in labor_income_columns]]

    activity_stable = activity_state.nunique(axis=1).eq(1) & activity_state.iloc[:, 0].ne(-1)
    worked_hours_stable = worked_hours_state.notna().all(axis=1) & worked_hours_state.nunique(axis=1).eq(1)
    labor_income_stable = labor_income_state.notna().all(axis=1) & labor_income_state.nunique(axis=1).eq(1)
    stable_mask = activity_stable & worked_hours_stable & labor_income_stable
    return result.loc[stable_mask].copy()


def run_mpc_experiment(
    *,
    seeds: list[int],
    t_max: int,
    shock_period: int,
    horizon_periods: int,
    shock_fraction: float,
    output_dir: Path,
    country_iso3: str | None = None,
    raw_data_path: str | Path | None = None,
    config_dir: str | Path | None = None,
    force_rebuild_data: bool = False,
    single_hfcs_survey: bool = False,
    n_jobs: int = 1,
    backend: str = "loky",
    verbose: int = 0,
    batch_size: int = 1,
    mpc_plot_measure: str | None = None,
    cpi_source: str = "cpi_fixed_basket",
    apply_mpc_filters: bool = True,
    mpc_filter_config: MPCFilterConfig | None = None,
    stay_in_activity_bracket_only: bool = False,
    stable_effective_labor_state_only: bool = False,
    activity_bracket_prefix: str = "activity_bracket_",
) -> dict[str, Path]:
    """Run paired MPC simulations, analyse household responses, and write outputs.

    ``n_jobs`` controls seed-level parallelism. Each job runs one baseline and
    one shock simulation for a seed, so ``n_jobs=1`` is serial and values above
    one use joblib workers.

    Setting ``stay_in_activity_bracket_only=True`` drops households whose
    activity bracket (employed, unemployed, investor, or not economically
    active) switches at any point across the MPC measurement window, in either
    the baseline or shock run. Households that are stable in any one bracket
    throughout are kept.

    Setting ``stable_effective_labor_state_only=True`` applies a stricter MPC
    identification rule: keep only households whose activity bracket,
    aggregated worked-hours bin, and labour-income bin are each constant across
    the full baseline-plus-shock MPC window.
    """
    seed_list = _validate_unique_seeds(seeds)
    if n_jobs == 0:
        raise ValueError("n_jobs must be non-zero.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if mpc_plot_measure is not None and mpc_plot_measure not in MPC_PLOT_MEASURES:
        raise ValueError(f"mpc_plot_measure must be one of {sorted(MPC_PLOT_MEASURES)}.")
    if shock_period < 0 or shock_period >= t_max:
        raise ValueError("shock_period must satisfy 0 <= shock_period < t_max.")
    identification_filter_mode = _resolve_identification_filter_mode(
        stay_in_activity_bracket_only=stay_in_activity_bracket_only,
        stable_effective_labor_state_only=stable_effective_labor_state_only,
    )
    output_dir = Path(output_dir)
    shock_row = shock_period + 1
    if shock_row + horizon_periods > t_max + 1:
        raise ValueError("shock_period + horizon_periods extends beyond available HDF5 rows.")

    _run_config, data, country_configurations, country_code, resolved_raw_data_path = _prepare_mpc_inputs(
        seed=seed_list[0],
        t_max=t_max,
        country_iso3=country_iso3,
        raw_data_path=raw_data_path,
        output_dir=output_dir,
        config_dir=config_dir,
        force_rebuild_data=force_rebuild_data,
        single_hfcs_survey=single_hfcs_survey,
    )
    data_paths = DataPaths.default_paths(resolved_raw_data_path, [data.configuration.year])

    baseline_dir = output_dir / "baseline"
    shock_dir = output_dir / "shock"
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose, batch_size=batch_size)(
        delayed(_run_seed_pair)(
            data=data,
            country_configurations=country_configurations,
            country_code=country_code,
            seed=seed,
            t_max=t_max,
            baseline_dir=baseline_dir,
            shock_dir=shock_dir,
            shock_period=shock_period,
            shock_fraction=shock_fraction,
            horizon_periods=horizon_periods,
            data_paths=data_paths,
        )
        for seed in seed_list
    )

    metadata = pd.concat(metadata_rows, ignore_index=True)
    metadata.to_csv(analysis_dir / "household_pre_shock_metadata.csv", index=False)

    cumulative_mpc_column = "cmpc_4q" if horizon_periods == 4 else f"cmpc_{horizon_periods}p"
    target_cumulative_mpc_column = "target_cmpc_4q" if horizon_periods == 4 else f"target_cmpc_{horizon_periods}p"
    real_cumulative_mpc_column = "real_cmpc_4q" if horizon_periods == 4 else f"real_cmpc_{horizon_periods}p"
    target_real_cumulative_mpc_column = (
        "target_real_cmpc_4q" if horizon_periods == 4 else f"target_real_cmpc_{horizon_periods}p"
    )
    plotted_mpc_column = cumulative_mpc_column if mpc_plot_measure == "nominal" else real_cumulative_mpc_column
    plot_mpc_columns = (
        [plotted_mpc_column] if mpc_plot_measure is not None else [real_cumulative_mpc_column, cumulative_mpc_column]
    )

    panels = []
    for seed in seed_list:
        seed_metadata = metadata.loc[metadata["seed"] == int(seed)].copy()
        panel = build_household_mpc_panel(
            baseline_h5=baseline_dir / f"seed-{int(seed)}" / "multi_country_simulation.h5",
            shock_h5=shock_dir / f"seed-{int(seed)}" / "multi_country_simulation.h5",
            metadata=seed_metadata,
            country_code=country_code,
            shock_row=shock_row,
            horizon_periods=horizon_periods,
            shock_fraction=shock_fraction,
            cpi_source=cpi_source,
            cumulative_mpc_column=cumulative_mpc_column,
            target_cumulative_mpc_column=target_cumulative_mpc_column,
            real_cumulative_mpc_column=real_cumulative_mpc_column,
            target_real_cumulative_mpc_column=target_real_cumulative_mpc_column,
        )
        panels.append(panel)

    raw_household_panel = add_mpc_bins(pd.concat(panels, ignore_index=True))
    if identification_filter_mode == "activity_bracket":
        raw_household_panel = _keep_households_staying_in_activity_bracket(
            raw_household_panel,
            activity_bracket_prefix=activity_bracket_prefix,
        )
    elif identification_filter_mode == "effective_labor_state":
        raw_household_panel = _keep_households_in_stable_effective_labor_state(raw_household_panel)
    filter_config = mpc_filter_config or MPCFilterConfig(enabled=apply_mpc_filters)
    if not apply_mpc_filters and filter_config.enabled:
        filter_config = MPCFilterConfig(enabled=False)
    filtered_household_panel, filter_report = filter_mpc_panel(
        raw_household_panel,
        filter_config,
        required_mpc_columns=plot_mpc_columns,
    )
    filtered_household_panel = add_mpc_bins(filtered_household_panel)
    raw_summary = summarize_mpc_bins(raw_household_panel, mpc_column=plotted_mpc_column)
    filtered_summary = summarize_mpc_bins(filtered_household_panel, mpc_column=plotted_mpc_column)
    raw_household_panel.to_csv(analysis_dir / "household_mpc_panel_raw.csv", index=False)
    filtered_household_panel.to_csv(analysis_dir / "household_mpc_panel_filtered.csv", index=False)
    raw_summary.to_csv(analysis_dir / "household_mpc_summary_raw.csv", index=False)
    filtered_summary.to_csv(analysis_dir / "household_mpc_summary_filtered.csv", index=False)
    filter_report.to_csv(analysis_dir / "household_mpc_filter_report.csv", index=False)
    filtered_household_panel.to_csv(analysis_dir / "household_mpc_panel.csv", index=False)
    filtered_summary.to_csv(analysis_dir / "household_mpc_summary.csv", index=False)

    return {
        "baseline_dir": baseline_dir,
        "shock_dir": shock_dir,
        "analysis_dir": analysis_dir,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run household MPC distribution experiment.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--t-max", type=int, default=DEFAULT_T_MAX)
    parser.add_argument("--country", default=None, help="Optional country ISO3 override.")
    parser.add_argument("--raw-data-path", default=None, help="Optional raw data path override.")
    parser.add_argument("--config-dir", default=None, help="Optional config directory override.")
    parser.add_argument("--force-rebuild-data", action="store_true", help="Force rebuilding the prepared data cache.")
    parser.add_argument(
        "--single-hfcs-survey", action="store_true", help="Use one HFCS survey during data preparation."
    )
    parser.add_argument("--shock-period", type=int, default=DEFAULT_SHOCK_PERIOD)
    parser.add_argument("--horizon-periods", type=int, default=DEFAULT_HORIZON_PERIODS)
    parser.add_argument("--shock-fraction", type=float, default=DEFAULT_SHOCK_FRACTION)
    parser.add_argument("--output-dir", type=Path, default=Path("/private/tmp/inet-mpc-experiment"))
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of joblib workers for seed-level parallelism.")
    parser.add_argument("--backend", default="loky", help="Joblib backend. Default: loky.")
    parser.add_argument("--verbose", type=int, default=0, help="Joblib verbosity level.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of seeds batched per worker.")
    parser.add_argument(
        "--mpc-plot-measure",
        choices=("real", "nominal"),
        default=None,
        help=(
            "MPC measure used for filtering and summaries. By default, summaries use real MPC. Real deflates "
            "each path by its own CPI."
        ),
    )
    parser.add_argument("--cpi-source", default="cpi_fixed_basket", help="Saved economy CPI series for real MPCs.")
    parser.add_argument(
        "--no-mpc-filters",
        action="store_true",
        help="Write summaries from the raw MPC panel instead of the filtered analysis sample.",
    )
    parser.add_argument(
        "--stay-in-activity-bracket-only",
        action="store_true",
        help=(
            "Drop households whose activity bracket (employed, unemployed, investor, or not economically "
            "active) switches during the MPC window, in either the baseline or shock run. Households stable "
            "in any one bracket throughout are kept."
        ),
    )
    parser.add_argument(
        "--stable-effective-labor-state-only",
        action="store_true",
        help=(
            "Drop households unless their activity bracket, worked-hours state, and labour-income state are "
            "all stable throughout the MPC window in both baseline and shock paths."
        ),
    )
    parser.add_argument(
        "--activity-bracket-prefix",
        default="activity_bracket_",
        help="Column prefix used to identify period-by-period activity bracket indicators in the MPC panel.",
    )
    args = parser.parse_args()
    args.seeds = _validate_unique_seeds(args.seeds)
    return args


def main() -> None:
    """CLI entry point retained for the compatibility command wrapper."""
    parsed = _parse_args()
    identification_filter_mode = _resolve_identification_filter_mode(
        stay_in_activity_bracket_only=parsed.stay_in_activity_bracket_only,
        stable_effective_labor_state_only=parsed.stable_effective_labor_state_only,
    )
    outputs = run_mpc_experiment(
        seeds=parsed.seeds,
        t_max=parsed.t_max,
        shock_period=parsed.shock_period,
        horizon_periods=parsed.horizon_periods,
        shock_fraction=parsed.shock_fraction,
        output_dir=parsed.output_dir,
        country_iso3=parsed.country,
        raw_data_path=parsed.raw_data_path,
        config_dir=parsed.config_dir,
        force_rebuild_data=parsed.force_rebuild_data,
        single_hfcs_survey=parsed.single_hfcs_survey,
        n_jobs=parsed.n_jobs,
        backend=parsed.backend,
        verbose=parsed.verbose,
        batch_size=parsed.batch_size,
        mpc_plot_measure=parsed.mpc_plot_measure,
        cpi_source=parsed.cpi_source,
        apply_mpc_filters=not parsed.no_mpc_filters,
        stay_in_activity_bracket_only=parsed.stay_in_activity_bracket_only,
        stable_effective_labor_state_only=parsed.stable_effective_labor_state_only,
        activity_bracket_prefix=parsed.activity_bracket_prefix,
    )
    print(
        {
            "identification_filter_mode": identification_filter_mode,
            **{name: str(path) for name, path in outputs.items()},
        }
    )


if __name__ == "__main__":
    main()
