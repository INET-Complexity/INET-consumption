"""Run paired baseline-vs-shock household MPC experiments.

This script is intentionally separate from ``run_model_mc.py`` because the
current Monte Carlo helper does not expose a hook-registration point. The runner
instantiates each simulation directly, registers the household income prehook
only on the shock run, saves paired HDF5 files, and then delegates MPC
extraction/plotting to ``run_model.src.mpc_analysis``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

RUN_MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_MODEL_DIR.parent
for path in (str(REPO_ROOT), str(RUN_MODEL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.mpc_analysis import (  # noqa: E402
    MPC_PLOT_MEASURES,
    PLOT_KINDS,
    MPCFilterConfig,
    add_mpc_bins,
    build_household_mpc_panel,
    filter_mpc_panel,
    make_household_metadata,
    period_to_year_month,
    summarize_mpc_bins,
    write_distribution_plots,
)
from src.notebook_workflow import NotebookRunConfig, build_country_config, prepare_data  # noqa: E402

from macro_data import DataWrapper  # noqa: E402
from macromodel.configurations import CountryConfiguration, SimulationConfiguration  # noqa: E402
from macromodel.simulation import Simulation  # noqa: E402
from macromodel.utils.prehooks.household_income_shock import create_household_income_shock_hook  # noqa: E402

DEFAULT_SEEDS = [12, 13, 14]
DEFAULT_T_MAX = 50
DEFAULT_SHOCK_PERIOD = 20
DEFAULT_HORIZON_PERIODS = 4
DEFAULT_SHOCK_FRACTION = 0.01


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
) -> tuple[NotebookRunConfig, DataWrapper, dict[str, CountryConfiguration], str]:
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
    return run_config, prepared.data, country_configurations, prepared.cfg.country_iso3


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
) -> pd.DataFrame:
    """Run one baseline or shock simulation and return pre-shock metadata.

    ``shock_period`` is a zero-based simulation iteration. Since HDF5 row ``0``
    is the initial state, metadata for binning is captured at HDF5 row
    ``shock_period``: the state immediately before the shock affects realised row
    ``shock_period + 1``.
    """
    configuration = SimulationConfiguration.model_validate(
        {
            "country_configurations": {code: config.model_dump() for code, config in country_configurations.items()},
            "t_max": int(t_max),
            "seed": int(seed),
        }
    )
    model = Simulation.from_datawrapper(datawrapper=data, simulation_configuration=configuration)

    pre_shock_row = int(shock_period)
    metadata: list[pd.DataFrame] = []

    def capture_pre_shock_metadata(simulation: Simulation, t: int, _year: int, _month: int) -> None:
        # Posthooks see the realised row for iteration ``t``. Capturing when
        # ``t == shock_period - 1`` gives the pre-shock HDF5 row.
        if t == shock_period - 1:
            metadata.append(
                make_household_metadata(
                    simulation.countries[country_code],
                    seed=seed,
                    pre_shock_row=pre_shock_row,
                )
            )

    if shock_period == 0:
        metadata.append(make_household_metadata(model.countries[country_code], seed=seed, pre_shock_row=0))
    else:
        model.posthooks.append(capture_pre_shock_metadata)

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

    seed_dir = output_dir / f"seed-{int(seed)}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    model.save(save_dir=seed_dir, file_name="multi_country_simulation.h5")
    return metadata[0]


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
) -> pd.DataFrame:
    """Run the baseline and shock simulations for one seed."""
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
    )
    _run_one(
        data=data,
        country_configurations=country_configurations,
        country_code=country_code,
        seed=seed,
        t_max=t_max,
        output_dir=shock_dir,
        shock=True,
        shock_period=shock_period,
        shock_fraction=shock_fraction,
    )
    return baseline_metadata


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
    distribution_plot_kind: str = "violin",
    mpc_plot_measure: str | None = None,
    cpi_source: str = "cpi_fixed_basket",
    apply_mpc_filters: bool = True,
    mpc_filter_config: MPCFilterConfig | None = None,
    mpc_plot_y_quantiles: tuple[float, float] | None = None,
    mpc_plot_y_range: tuple[float, float] | None = None,
    mpc_winsor_quantiles: tuple[float, float] | None = None,
) -> dict[str, Path]:
    """Run paired MPC simulations, analyse household responses, and write outputs.

    ``n_jobs`` controls seed-level parallelism. Each job runs one baseline and
    one shock simulation for a seed, so ``n_jobs=1`` is serial and values above
    one use joblib workers.
    """
    seed_list = _validate_unique_seeds(seeds)
    if n_jobs == 0:
        raise ValueError("n_jobs must be non-zero.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if distribution_plot_kind not in PLOT_KINDS:
        raise ValueError(f"distribution_plot_kind must be one of {sorted(PLOT_KINDS)}.")
    if mpc_plot_measure is not None and mpc_plot_measure not in MPC_PLOT_MEASURES:
        raise ValueError(f"mpc_plot_measure must be one of {sorted(MPC_PLOT_MEASURES)}.")
    if shock_period < 0 or shock_period >= t_max:
        raise ValueError("shock_period must satisfy 0 <= shock_period < t_max.")
    output_dir = Path(output_dir)
    shock_row = shock_period + 1
    if shock_row + horizon_periods > t_max + 1:
        raise ValueError("shock_period + horizon_periods extends beyond available HDF5 rows.")

    _run_config, data, country_configurations, country_code = _prepare_mpc_inputs(
        seed=seed_list[0],
        t_max=t_max,
        country_iso3=country_iso3,
        raw_data_path=raw_data_path,
        output_dir=output_dir,
        config_dir=config_dir,
        force_rebuild_data=force_rebuild_data,
        single_hfcs_survey=single_hfcs_survey,
    )

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
        [plotted_mpc_column]
        if mpc_plot_measure is not None
        else [real_cumulative_mpc_column, cumulative_mpc_column]
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
    for plot_mpc_column in plot_mpc_columns:
        write_distribution_plots(
            filtered_household_panel,
            analysis_dir / "plots",
            mpc_column=plot_mpc_column,
            plot_kind=distribution_plot_kind,
            y_quantiles=mpc_plot_y_quantiles,
            y_range=mpc_plot_y_range,
            winsor_quantiles=mpc_winsor_quantiles,
        )

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
        "--distribution-plot-kind",
        choices=("violin", "box"),
        default="violin",
        help="Distribution plot style. Use 'box' to omit violin traces.",
    )
    parser.add_argument(
        "--mpc-plot-measure",
        choices=("real", "nominal"),
        default=None,
        help=(
            "MPC measure used in summaries and plots. By default, summaries use real MPC and plots include both "
            "real and nominal MPC. Real deflates each path by its own CPI."
        ),
    )
    parser.add_argument("--cpi-source", default="cpi_fixed_basket", help="Saved economy CPI series for real MPCs.")
    parser.add_argument(
        "--no-mpc-filters",
        action="store_true",
        help="Write summaries and plots from the raw MPC panel instead of the filtered analysis sample.",
    )
    parser.add_argument(
        "--mpc-plot-y-quantiles",
        nargs=2,
        type=float,
        default=None,
        metavar=("LOWER", "UPPER"),
        help="Clip plot y-axis display to these MPC quantiles, e.g. 0.01 0.99. Does not change data.",
    )
    parser.add_argument(
        "--mpc-plot-y-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("LOWER", "UPPER"),
        help="Clip plot y-axis display to this fixed range. Does not change data.",
    )
    parser.add_argument(
        "--mpc-winsor-quantiles",
        nargs=2,
        type=float,
        default=None,
        metavar=("LOWER", "UPPER"),
        help="Winsorise plotted MPC values at these quantiles. Does not change saved panels or summaries.",
    )
    args = parser.parse_args()
    args.seeds = _validate_unique_seeds(args.seeds)
    return args


if __name__ == "__main__":
    parsed = _parse_args()
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
        distribution_plot_kind=parsed.distribution_plot_kind,
        mpc_plot_measure=parsed.mpc_plot_measure,
        cpi_source=parsed.cpi_source,
        apply_mpc_filters=not parsed.no_mpc_filters,
        mpc_plot_y_quantiles=tuple(parsed.mpc_plot_y_quantiles) if parsed.mpc_plot_y_quantiles else None,
        mpc_plot_y_range=tuple(parsed.mpc_plot_y_range) if parsed.mpc_plot_y_range else None,
        mpc_winsor_quantiles=tuple(parsed.mpc_winsor_quantiles) if parsed.mpc_winsor_quantiles else None,
    )
    print({name: str(path) for name, path in outputs.items()})
