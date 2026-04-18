from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any, Iterable, Sequence

import pandas as pd

from macro_data import DataWrapper
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from src.monte_carlo import MonteCarloResult, run_seeded_monte_carlo


@dataclass(frozen=True)
class SensitivityResult:
    """Container for one-parameter sensitivity outputs."""

    by_value: dict[Any, MonteCarloResult]
    combined: pd.DataFrame
    parameter_name: str
    parameter_path: tuple[Any, ...]

    def get_value(self, value: Any) -> MonteCarloResult:
        """Return the Monte Carlo result for one parameter value."""
        return self.by_value[value]

    def to_parquet(self, path: str) -> None:
        """Persist the combined sensitivity output table."""
        self.combined.to_parquet(path)

    def to_pickle(self, path: str) -> None:
        """Persist the combined sensitivity output table."""
        self.combined.to_pickle(path)


def _set_nested_value(target: Any, path: Sequence[Any], value: Any) -> None:
    """Set a nested attribute or dict key on a copied configuration object."""
    current = target
    for key in path[:-1]:
        if isinstance(current, dict):
            current = current[key]
        else:
            current = getattr(current, key)

    last = path[-1]
    if isinstance(current, dict):
        current[last] = value
    else:
        setattr(current, last, value)


def _normalise_parameter_path(
    parameter_path: Sequence[Any] | str,
    country_configurations: dict[str, CountryConfiguration],
    country_code: str,
) -> tuple[Any, ...]:
    """Resolve whether the override path is rooted at country_configurations or a country config."""
    if isinstance(parameter_path, str):
        path = _parse_parameter_path(parameter_path)
    else:
        path = tuple(parameter_path)

    if not path:
        raise ValueError("parameter_path must contain at least one element.")

    if path[0] == country_code:
        return path

    if path[0] in country_configurations:
        return path

    return (country_code, *path)


def _parse_parameter_path(parameter_path: str) -> tuple[str, ...]:
    """Parse dotted paths with optional dict indexing into path components.

    Example
    -------
    ``firms.functions.offered_wage_setter.parameters['labour_market_tightness_markup_scale']``
    becomes
    ``("firms", "functions", "offered_wage_setter", "parameters", "labour_market_tightness_markup_scale")``
    """
    pattern = r"[A-Za-z_][A-Za-z0-9_]*|\[['\"]([^'\"]+)['\"]\]"
    parts = []
    for match in re.finditer(pattern, parameter_path):
        token = match.group(0)
        key = match.group(1)
        parts.append(key if key is not None else token)

    if not parts:
        raise ValueError(f"Could not parse parameter_path: {parameter_path!r}")

    return tuple(parts)


def build_country_configurations_with_override(
    *,
    country_configurations: dict[str, CountryConfiguration],
    country_code: str,
    parameter_path: Sequence[Any] | str,
    parameter_value: Any,
) -> dict[str, CountryConfiguration]:
    """Return a deep-copied configuration mapping with one parameter overridden."""
    configs = deepcopy(country_configurations)
    resolved_path = _normalise_parameter_path(parameter_path, configs, country_code)
    _set_nested_value(configs, resolved_path, parameter_value)
    return configs


def run_parameter_sensitivity(
    *,
    datawrapper: DataWrapper,
    country_configurations: dict[str, CountryConfiguration],
    country_code: str,
    parameter_path: Sequence[Any] | str,
    parameter_values: Iterable[Any],
    seeds: Iterable[int],
    t_max: int,
    simulation_configuration: SimulationConfiguration | None = None,
    n_jobs: int = -1,
    backend: str = "loky",
    verbose: int = 0,
    batch_size: int = 1,
) -> SensitivityResult:
    """Run seeded Monte Carlo simulations for a range of values of one parameter.

    The returned dataframe uses a MultiIndex with levels:
    ``parameter``, ``value``, ``seed``, and simulation ``time``.
    """
    value_list = list(parameter_values)
    if not value_list:
        raise ValueError("At least one parameter value must be provided.")
    if country_code not in country_configurations:
        raise ValueError(f"Country code '{country_code}' is missing from country_configurations.")

    resolved_path = _normalise_parameter_path(parameter_path, country_configurations, country_code)
    parameter_name = str(resolved_path[-1])

    by_value: dict[Any, MonteCarloResult] = {}
    combined_by_value: dict[Any, pd.DataFrame] = {}

    for value in value_list:
        overridden_configurations = build_country_configurations_with_override(
            country_configurations=country_configurations,
            country_code=country_code,
            parameter_path=resolved_path,
            parameter_value=value,
        )
        mc_result = run_seeded_monte_carlo(
            datawrapper=datawrapper,
            country_configurations=overridden_configurations,
            country_code=country_code,
            seeds=seeds,
            t_max=t_max,
            simulation_configuration=simulation_configuration,
            n_jobs=n_jobs,
            backend=backend,
            verbose=verbose,
            batch_size=batch_size,
        )
        by_value[value] = mc_result
        combined_by_value[value] = mc_result.combined

    combined = pd.concat(combined_by_value, names=["value"])
    combined["parameter"] = parameter_name
    combined = combined.set_index("parameter", append=True)
    combined = combined.reorder_levels(["parameter", "value", "seed", "time"]).sort_index()

    return SensitivityResult(
        by_value=by_value,
        combined=combined,
        parameter_name=parameter_name,
        parameter_path=resolved_path,
    )
