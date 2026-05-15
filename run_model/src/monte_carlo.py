from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from joblib import Parallel, delayed
from src.visual_helpers import build_macro_output_df

from macro_data import DataWrapper
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.simulation import Simulation


@dataclass(frozen=True)
class MonteCarloResult:
    """Container for Monte Carlo outputs indexed by simulation seed."""

    by_seed: dict[int, pd.DataFrame]
    combined: pd.DataFrame

    def get_seed(self, seed: int) -> pd.DataFrame:
        """Return the output dataframe for a single seed."""
        return self.by_seed[int(seed)]

    def to_parquet(self, path: str) -> None:
        """Persist the combined output table for later inspection."""
        self.combined.to_parquet(path)

    def to_pickle(self, path: str) -> None:
        """Persist the combined output table when parquet is not desired."""
        self.combined.to_pickle(path)


def _chunked(values: list[int], chunk_size: int) -> list[list[int]]:
    """Split a list of seeds into fixed-size chunks."""
    if chunk_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    iterator = iter(values)
    chunks = []
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            break
        chunks.append(chunk)
    return chunks


def _dump_country_configurations(
    country_configurations: dict[str, CountryConfiguration],
) -> dict[str, dict]:
    """Serialize pydantic country configurations into plain dicts.

    Joblib workers (loky) can end up with multiple copies of the same module on
    sys.path (editable installs + repo checkout), so passing pydantic model
    instances across processes may fail validation due to class identity
    mismatches. Dumping to plain dicts avoids that.
    """
    return {code: config.model_dump() for code, config in country_configurations.items()}


def _run_single_seed(
    *,
    seed: int,
    datawrapper: DataWrapper,
    country_configurations: dict[str, CountryConfiguration],
    t_max: int,
    country_code: str,
    simulation_configuration: Optional[SimulationConfiguration] = None,
    save_h5_dir: str | Path | None = None,
) -> tuple[int, pd.DataFrame]:
    """Run one seeded simulation and extract macro output time series."""
    country_payload = _dump_country_configurations(country_configurations)
    if simulation_configuration is None:
        configuration = SimulationConfiguration.model_validate(
            {
                "country_configurations": country_payload,
                "t_max": t_max,
                "seed": int(seed),
            }
        )
    else:
        payload = simulation_configuration.model_dump()
        payload.update(
            {
                "country_configurations": country_payload,
                "t_max": t_max,
                "seed": int(seed),
            }
        )
        configuration = SimulationConfiguration.model_validate(payload)

    model = Simulation.from_datawrapper(
        datawrapper=datawrapper,
        simulation_configuration=configuration,
    )
    model.run()

    if save_h5_dir is not None:
        seed_dir = Path(save_h5_dir) / f"seed-{int(seed)}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        model.save(save_dir=seed_dir, file_name="multi_country_simulation.h5")

    output_df = build_macro_output_df(model, country_code=country_code).copy()
    output_df.index.name = output_df.index.name or "time"

    return int(seed), output_df


def _run_seed_batch(
    *,
    seeds: list[int],
    datawrapper: DataWrapper,
    country_configurations: dict[str, CountryConfiguration],
    t_max: int,
    country_code: str,
    simulation_configuration: Optional[SimulationConfiguration] = None,
    save_h5_dir: str | Path | None = None,
) -> list[tuple[int, pd.DataFrame]]:
    """Run a batch of seeds sequentially inside one worker process."""
    return [
        _run_single_seed(
            seed=seed,
            datawrapper=datawrapper,
            country_configurations=country_configurations,
            t_max=t_max,
            country_code=country_code,
            simulation_configuration=simulation_configuration,
            save_h5_dir=save_h5_dir,
        )
        for seed in seeds
    ]


def run_seeded_monte_carlo(
    *,
    datawrapper: DataWrapper,
    country_configurations: dict[str, CountryConfiguration],
    country_code: str,
    seeds: Iterable[int],
    t_max: int,
    simulation_configuration: Optional[SimulationConfiguration] = None,
    n_jobs: int = -1,
    backend: str = "loky",
    verbose: int = 0,
    batch_size: int = 1,
    save_h5_dir: str | Path | None = None,
) -> MonteCarloResult:
    """Run the model in parallel over a predefined list of random seeds.

    Parameters
    ----------
    datawrapper:
        The preprocessed model data used to instantiate each simulation.
    country_configurations:
        Mapping passed to ``SimulationConfiguration(country_configurations=...)``.
    country_code:
        ISO3 code used to extract outputs with ``build_macro_output_df``.
    seeds:
        Predefined random seeds, one per Monte Carlo run.
    t_max:
        Simulation horizon passed into ``SimulationConfiguration``.
    simulation_configuration:
        Optional baseline configuration to clone before overriding seed, t_max,
        and country_configurations.
    n_jobs:
        Number of parallel workers. Uses all available workers by default.
    backend:
        Joblib backend. ``loky`` is the default process-based backend.
    verbose:
        Joblib verbosity level.
    batch_size:
        Number of seeds handled sequentially by each joblib worker. Values
        greater than 1 reduce process-spawning and serialization overhead for
        large Monte Carlo runs.
    save_h5_dir:
        Optional directory for per-seed HDF5 outputs. When omitted, the existing
        Monte Carlo behavior is unchanged and only the combined dataframe is
        returned.

    Returns
    -------
    MonteCarloResult
        ``by_seed`` stores one dataframe per seed.
        ``combined`` concatenates all runs into one dataframe indexed by
        ``seed`` and simulation time.
    """
    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise ValueError("At least one seed must be provided.")
    if len(set(seed_list)) != len(seed_list):
        raise ValueError("Seeds must be unique. Duplicate seeds would overwrite results in the output mapping.")
    if country_code not in country_configurations:
        raise ValueError(f"Country code '{country_code}' is missing from country_configurations.")
    if country_code not in datawrapper.synthetic_countries:
        raise ValueError(f"Country code '{country_code}' is not available in the provided datawrapper.")

    seed_batches = _chunked(seed_list, batch_size)
    batch_runs = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(_run_seed_batch)(
            seeds=seed_batch,
            datawrapper=datawrapper,
            country_configurations=country_configurations,
            t_max=t_max,
            country_code=country_code,
            simulation_configuration=simulation_configuration,
            save_h5_dir=save_h5_dir,
        )
        for seed_batch in seed_batches
    )

    runs = [run for batch in batch_runs for run in batch]
    by_seed = {seed: df for seed, df in runs}
    combined = pd.concat(by_seed, names=["seed"])

    return MonteCarloResult(by_seed=by_seed, combined=combined)
