"""Run paired baseline-vs-shock impulse-response experiments.

This runner is intentionally separate from ``run_mpc_experiment.py``. MPC
experiments remain household-level income-shock measurements; this module runs
generic macro shock arms and writes aggregate impulse-response outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from joblib import Parallel, delayed

RUN_MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_MODEL_DIR.parent
for path in (str(REPO_ROOT), str(RUN_MODEL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.irf_analysis import build_irf_panel, summarize_irf_panel, write_irf_plots  # noqa: E402
from src.notebook_workflow import NotebookRunConfig, build_country_config, prepare_data  # noqa: E402

from macro_data import DataWrapper  # noqa: E402
from macro_data.readers.default_readers import DataPaths  # noqa: E402
from macromodel.configurations import CountryConfiguration, SimulationConfiguration  # noqa: E402
from macromodel.simulation import Simulation  # noqa: E402
from macromodel.utils.prehooks.irf_shocks import ShockSpec, create_irf_shock_hook  # noqa: E402

DEFAULT_SEEDS = [12, 13, 14]
DEFAULT_T_MAX = 50
DEFAULT_HORIZON_PERIODS = 12


def _validate_unique_seeds(seeds: list[int]) -> list[int]:
    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise argparse.ArgumentTypeError("At least one seed must be provided.")
    if len(set(seed_list)) != len(seed_list):
        raise argparse.ArgumentTypeError("Seeds must be unique.")
    return seed_list


def _prepare_irf_inputs(
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
    run_config = NotebookRunConfig(
        seed=int(seed),
        t_max=int(t_max),
        country_iso3=country_iso3,
        raw_data_path=raw_data_path,
        output_dir=output_dir,
        config_dir=config_dir,
        force_rebuild_data=force_rebuild_data,
        single_hfcs_survey=single_hfcs_survey,
    )
    prepared = prepare_data(run_config)
    country_configurations = build_country_config(data=prepared.data, config=run_config)
    return run_config, prepared.data, country_configurations, prepared.cfg.country_iso3, prepared.raw_data_path


def _build_simulation(
    *,
    data: DataWrapper,
    country_configurations: dict[str, CountryConfiguration],
    seed: int,
    t_max: int,
    data_paths: DataPaths | None = None,
) -> Simulation:
    configuration = SimulationConfiguration.model_validate(
        {
            "country_configurations": {code: config.model_dump() for code, config in country_configurations.items()},
            "t_max": int(t_max),
            "seed": int(seed),
        }
    )
    return Simulation.from_datawrapper(datawrapper=data, simulation_configuration=configuration, data_paths=data_paths)


def _run_one(
    *,
    data: DataWrapper,
    country_configurations: dict[str, CountryConfiguration],
    country_code: str,
    seed: int,
    t_max: int,
    output_dir: Path,
    shock_spec: ShockSpec | None,
    data_paths: DataPaths | None = None,
) -> Path:
    model = _build_simulation(
        data=data,
        country_configurations=country_configurations,
        seed=seed,
        t_max=t_max,
        data_paths=data_paths,
    )
    if shock_spec is not None:
        model.prehooks.append(
            create_irf_shock_hook(
                country_code=country_code,
                initial_year=data.configuration.year,
                time_unit=data.time_unit,
                spec=shock_spec,
            )
        )
    model.run()
    seed_dir = output_dir / f"seed-{int(seed)}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    model.save(save_dir=seed_dir, file_name="multi_country_simulation.h5")
    return seed_dir / "multi_country_simulation.h5"


def _run_seed(
    *,
    data: DataWrapper,
    country_configurations: dict[str, CountryConfiguration],
    country_code: str,
    seed: int,
    t_max: int,
    baseline_dir: Path,
    shocks_dir: Path,
    shock_specs: tuple[ShockSpec, ...],
    data_paths: DataPaths | None = None,
) -> list[dict[str, object]]:
    baseline_h5 = _run_one(
        data=data,
        country_configurations=country_configurations,
        country_code=country_code,
        seed=seed,
        t_max=t_max,
        output_dir=baseline_dir,
        shock_spec=None,
        data_paths=data_paths,
    )
    rows: list[dict[str, object]] = []
    for spec in shock_specs:
        shock_h5 = _run_one(
            data=data,
            country_configurations=country_configurations,
            country_code=country_code,
            seed=seed,
            t_max=t_max,
            output_dir=shocks_dir / spec.name,
            shock_spec=spec,
            data_paths=data_paths,
        )
        rows.append(
            {
                "seed": int(seed),
                "shock_name": spec.name,
                "shock_kind": spec.kind,
                "shock_period": spec.period,
                "shock_magnitude": spec.magnitude,
                "shock_duration": spec.duration,
                "shock_mode": spec.mode,
                "baseline_h5": baseline_h5,
                "shock_h5": shock_h5,
            }
        )
    return rows


def _shock_specs_frame(shock_specs: tuple[ShockSpec, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "shock_name": spec.name,
                "shock_kind": spec.kind,
                "shock_period": spec.period,
                "shock_magnitude": spec.magnitude,
                "shock_duration": spec.duration,
                "shock_mode": spec.mode,
            }
            for spec in shock_specs
        ]
    )


def run_irf_experiment(
    *,
    seeds: list[int],
    t_max: int,
    shock_specs: tuple[ShockSpec, ...],
    horizon_periods: int,
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
) -> dict[str, Path]:
    """Run paired IRF simulations, aggregate responses, and write outputs."""

    seed_list = _validate_unique_seeds(seeds)
    if not shock_specs:
        raise ValueError("At least one shock specification is required.")
    if len({spec.name for spec in shock_specs}) != len(shock_specs):
        raise ValueError("Shock names must be unique.")
    if n_jobs == 0:
        raise ValueError("n_jobs must be non-zero.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if horizon_periods <= 0:
        raise ValueError("horizon_periods must be positive.")
    if any(spec.period + spec.duration > t_max for spec in shock_specs):
        raise ValueError("A shock period plus duration extends beyond the simulation horizon.")
    if any(spec.period + 1 + horizon_periods > t_max + 1 for spec in shock_specs):
        raise ValueError("A shock period plus horizon extends beyond available HDF5 rows.")

    output_dir = Path(output_dir)
    _run_config, data, country_configurations, country_code, resolved_raw_data_path = _prepare_irf_inputs(
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
    shocks_dir = output_dir / "shocks"
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _shock_specs_frame(shock_specs).to_csv(analysis_dir / "irf_shock_specs.csv", index=False)

    run_rows_nested = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose, batch_size=batch_size)(
        delayed(_run_seed)(
            data=data,
            country_configurations=country_configurations,
            country_code=country_code,
            seed=seed,
            t_max=t_max,
            baseline_dir=baseline_dir,
            shocks_dir=shocks_dir,
            shock_specs=shock_specs,
            data_paths=data_paths,
        )
        for seed in seed_list
    )
    run_rows = [row for rows in run_rows_nested for row in rows]
    run_index = pd.DataFrame(run_rows)
    run_index.to_csv(analysis_dir / "irf_run_index.csv", index=False)

    panels = []
    spec_by_name = {spec.name: spec for spec in shock_specs}
    for row in run_rows:
        spec = spec_by_name[str(row["shock_name"])]
        panels.append(
            build_irf_panel(
                baseline_h5=row["baseline_h5"],
                shock_h5=row["shock_h5"],
                seed=int(row["seed"]),
                shock_name=spec.name,
                shock_kind=spec.kind,
                shock_period=spec.period + 1,
                shock_magnitude=spec.magnitude,
                shock_duration=spec.duration,
                shock_mode=spec.mode,
                horizon_periods=horizon_periods,
                country_code=country_code,
            )
        )
    panel = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()
    summary = summarize_irf_panel(panel)
    panel.to_csv(analysis_dir / "irf_panel.csv", index=False)
    summary.to_csv(analysis_dir / "irf_summary.csv", index=False)
    write_irf_plots(summary, analysis_dir / "plots", value_column="delta")
    write_irf_plots(summary, analysis_dir / "plots_pct", value_column="pct_delta")
    return {"baseline_dir": baseline_dir, "shocks_dir": shocks_dir, "analysis_dir": analysis_dir}


def _load_shocks(path: str | Path) -> tuple[ShockSpec, ...]:
    with Path(path).open() as handle:
        payload = yaml.safe_load(handle) or {}
    shocks = payload["shocks"] if isinstance(payload, dict) and "shocks" in payload else payload
    if not isinstance(shocks, list):
        raise ValueError("Shock config must be a list or contain a top-level 'shocks' list.")
    return tuple(ShockSpec(**shock) for shock in shocks)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired impulse-response shock experiments.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--t-max", type=int, default=DEFAULT_T_MAX)
    parser.add_argument("--country", default=None, help="Optional country ISO3 override.")
    parser.add_argument("--raw-data-path", default=None, help="Optional raw data path override.")
    parser.add_argument("--config-dir", default=None, help="Optional config directory override.")
    parser.add_argument("--force-rebuild-data", action="store_true", help="Force rebuilding the prepared data cache.")
    parser.add_argument(
        "--single-hfcs-survey", action="store_true", help="Use one HFCS survey during data preparation."
    )
    parser.add_argument("--shock-config", required=True, type=Path, help="YAML file containing a 'shocks' list.")
    parser.add_argument("--horizon-periods", type=int, default=DEFAULT_HORIZON_PERIODS)
    parser.add_argument("--output-dir", type=Path, default=Path("/private/tmp/inet-irf-experiment"))
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of joblib workers for seed-level parallelism.")
    parser.add_argument("--backend", default="loky", help="Joblib backend. Default: loky.")
    parser.add_argument("--verbose", type=int, default=0, help="Joblib verbosity level.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of seeds batched per worker.")
    args = parser.parse_args()
    args.seeds = _validate_unique_seeds(args.seeds)
    args.shock_specs = _load_shocks(args.shock_config)
    return args


if __name__ == "__main__":
    parsed = _parse_args()
    outputs = run_irf_experiment(
        seeds=parsed.seeds,
        t_max=parsed.t_max,
        shock_specs=parsed.shock_specs,
        horizon_periods=parsed.horizon_periods,
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
    )
    print({name: str(path) for name, path in outputs.items()})
