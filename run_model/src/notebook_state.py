"""Stateful orchestration and run manifests for run-model notebooks."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from src.notebook_workflow import (
    BenchmarkResult,
    NotebookRunConfig,
    PreparedData,
    SimulationRunResult,
    build_country_config,
    prepare_data,
    run_benchmark,
    run_single_simulation,
)

from macromodel.configurations import CountryConfiguration


@dataclass(frozen=True)
class NotebookRunState:
    """All state produced by one standard notebook run."""

    config: NotebookRunConfig
    scenario_name: str
    scenario_overrides: dict[str, Any]
    prepared: PreparedData
    country_configurations: dict[str, CountryConfiguration]
    simulation: SimulationRunResult
    benchmark: BenchmarkResult | None
    manifest_path: Path | None = None

    @property
    def country_code(self) -> str:
        return self.prepared.cfg.country_iso3

    @property
    def model(self):
        return self.simulation.model

    @property
    def df_scenario(self) -> pd.DataFrame:
        return self.simulation.df_base

    @property
    def df_benchmark(self) -> pd.DataFrame | None:
        return None if self.benchmark is None else self.benchmark.df_benchmark


def validate_notebook_state(state: NotebookRunState) -> None:
    """Fail early when notebook cells are using inconsistent or stale run state."""
    if state.country_code not in state.model.countries:
        raise ValueError(f"Country {state.country_code!r} is missing from the simulation model.")
    if state.df_scenario is None or state.df_scenario.empty:
        raise ValueError("The simulation macro dataframe is empty.")
    if not state.simulation.model_h5_path.exists():
        raise FileNotFoundError(f"Simulation output is missing: {state.simulation.model_h5_path}")
    if state.config.run_benchmark and state.benchmark is None:
        raise ValueError("run_benchmark=True but no benchmark result is attached to the run state.")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _git_revision(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        result = subprocess.run(["git", *args], cwd=repo_root, check=False, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def build_run_manifest(state: NotebookRunState) -> dict[str, Any]:
    """Build a JSON-serializable record of configuration and output locations."""
    config = {field.name: _jsonable(getattr(state.config, field.name)) for field in fields(state.config)}
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git": _git_revision(repo_root),
        "country_code": state.country_code,
        "seed": state.prepared.cfg.seed,
        "t_max": state.prepared.cfg.t_max,
        "scenario": {"name": state.scenario_name, "overrides": _jsonable(state.scenario_overrides)},
        "run_config": config,
        "outputs": {
            "model_h5": str(state.simulation.model_h5_path),
            "benchmark_model_h5": str(state.benchmark.model_h5_path) if state.benchmark else None,
            "benchmark_dataframe": str(state.benchmark.df_benchmark_path) if state.benchmark else None,
            "prepared_data_cache": str(state.prepared.data_pkl_path),
        },
    }


def write_run_manifest(state: NotebookRunState, path: str | Path | None = None) -> Path:
    """Write the standard run manifest and return its path."""
    target = (
        Path(path)
        if path is not None
        else state.prepared.output_dir
        / f"run_manifest_{state.country_code}_seed{state.prepared.cfg.seed}_t{state.prepared.cfg.t_max}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_run_manifest(state), indent=2, sort_keys=True) + "\n")
    return target


def run_notebook_workflow(
    config: NotebookRunConfig,
    *,
    scenario_name: str,
    scenario_overrides: Mapping[str, Any] | None = None,
    previous_state: NotebookRunState | None = None,
    write_manifest: bool = True,
) -> NotebookRunState:
    """Prepare data, configure, run, benchmark, validate, and record one notebook run."""

    overrides = dict(scenario_overrides or {})

    prepared = (
        previous_state.prepared
        if previous_state is not None and previous_state.config == config
        else prepare_data(config)
    )
    # prepared = prepare_data(config)
    country_configurations = build_country_config(data=prepared.data, config=config, overrides=overrides)
    simulation = run_single_simulation(data=prepared.data, country_configurations=country_configurations, config=config)
    benchmark = run_benchmark(config) if config.run_benchmark else None
    state = NotebookRunState(
        config=config,
        scenario_name=scenario_name,
        scenario_overrides=overrides,
        prepared=prepared,
        country_configurations=country_configurations,
        simulation=simulation,
        benchmark=benchmark,
    )
    validate_notebook_state(state)
    if write_manifest:
        state = replace(state, manifest_path=write_run_manifest(state))
    return state
