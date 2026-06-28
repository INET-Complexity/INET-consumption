from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from config import Config
from src.helpers import align_country_configuration_to_data
from src.visual_helpers import build_macro_output_df

from macro_data import DataWrapper
from macro_data.configuration import DataConfiguration, split_country_configs
from macro_data.readers.default_readers import DataPaths
from macro_data.readers.permanent_income_forecast import forecast_common_permanent_income
from macro_data.readers.permanent_income_mapping import (
    FORECAST_READER_TO_SIMULATION_SOURCE_NAME,
    PermanentIncomeSimulationSources,
    build_permanent_income_forecast_regressors,
    rebase_real_pc_income_index,
)
from macromodel.configurations import CountryConfiguration, SimulationConfiguration, load_country_configuration
from macromodel.simulation import Simulation

RUN_MODEL_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class NotebookRunConfig:
    """User-facing inputs for the run-model notebook workflow."""

    seed: int = 32
    t_max: int = 50
    country_iso3: str | None = None
    raw_data_path: str | Path | None = None
    output_dir: str | Path | None = None
    config_dir: str | Path | None = None
    data_cache_name: str = "data.pkl"
    model_file_name: str | None = None
    force_rebuild_data: bool = False
    single_hfcs_survey: bool = False
    run_benchmark: bool = False
    force_rerun_benchmark: bool = False
    benchmark_data_cache_name: str = "data_benchmark.pkl"
    benchmark_df_cache_name: str | None = None
    benchmark_model_file_name: str | None = None
    benchmark_overrides: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PreparedData:
    cfg: Config
    data_config: DataConfiguration
    creator: DataWrapper
    data: DataWrapper
    national_accounts: pd.DataFrame
    data_pkl_path: Path
    output_dir: Path
    raw_data_path: Path


@dataclass(frozen=True)
class SimulationRunResult:
    model: Simulation
    df_base: pd.DataFrame
    model_h5_path: Path


@dataclass(frozen=True)
class BenchmarkResult:
    model: Simulation | None
    df_benchmark: pd.DataFrame
    model_h5_path: Path
    df_benchmark_path: Path
    benchmark_spec: dict[str, Any]
    loaded_from_cache: bool


PERMANENT_INCOME_LOG_RATIO_DATASETS = {
    "ln_y_p_over_y": "target_consumption_permanent_income_log_ratio",
    "zeta_times_posterior_mean": "target_consumption_permanent_income_log_ratio_individual",
    "common_log_ratio": "target_consumption_permanent_income_log_ratio_common",

}
PERMANENT_INCOME_LOG_RATIO_LABELS = {
    "ln_y_p_over_y": "ln(y^p / y)",
    "zeta_times_posterior_mean": "zeta * posterior_mean",
    "common_log_ratio": "common_log_ratio",
    "log_real_pc_income_t": "log_real_pc_income_t",
    "real_pc_income_idx": "real_pc_income_idx",
}


def _resolve_run_model_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return RUN_MODEL_DIR / resolved


def _resolve_model_h5_path(source: str | Path | SimulationRunResult | BenchmarkResult) -> Path:
    if hasattr(source, "model_h5_path"):
        return Path(getattr(source, "model_h5_path"))
    return Path(source)


def _resolve_simulation_model(source: Simulation | SimulationRunResult | BenchmarkResult) -> Simulation:
    if isinstance(source, Simulation):
        return source
    model = getattr(source, "model", None)
    if isinstance(model, Simulation):
        return model
    raise TypeError(
        "source must be a Simulation or a result object with a loaded .model. "
        "BenchmarkResult loaded from cache does not carry a live model."
    )


def _resolve_single_country_code(handle: h5py.File, country_code: str | None) -> str:
    if country_code is not None:
        return country_code
    country_codes = list(handle.keys())
    if len(country_codes) != 1:
        raise ValueError("country_code is required when the HDF5 contains multiple country groups.")
    return str(country_codes[0])


def _read_log_real_pc_income_series_from_h5(handle: h5py.File, country_code: str) -> np.ndarray:
    try:
        income = np.asarray(handle[f"{country_code}/households/income"], dtype=float)
        cpi_fixed_basket = np.asarray(handle[f"{country_code}/economy/cpi_fixed_basket"], dtype=float).reshape(-1)
        n_individuals = np.asarray(handle[f"{country_code}/individuals/n_individuals"], dtype=float).reshape(-1)
    except KeyError as exc:
        raise KeyError(
            "HDF5 is missing one of the datasets required to reconstruct log_real_pc_income_t: "
            "households/income, economy/cpi_fixed_basket, individuals/n_individuals."
        ) from exc

    if income.ndim != 2:
        raise ValueError(f"Expected 2D household income time series, got shape {income.shape}.")

    total_income_history = income.sum(axis=1)
    if total_income_history.size != cpi_fixed_basket.size or total_income_history.size != n_individuals.size:
        raise ValueError(
            "income, cpi_fixed_basket, and n_individuals histories must have matching lengths to reconstruct "
            "log_real_pc_income_t."
        )

    real_pc_income_levels = total_income_history / cpi_fixed_basket / n_individuals
    rebased_income_index = rebase_real_pc_income_index(real_pc_income_levels, base_period_index=0)
    return np.log(rebased_income_index)


def build_permanent_income_log_ratio_decomposition_df(
    source: str | Path | SimulationRunResult | BenchmarkResult,
    *,
    country_code: str | None = None,
    reducer: str = "mean",
    include_log_real_pc_income: bool = False,
) -> pd.DataFrame:
    """Aggregate saved household ``ln(y^p / y)`` diagnostics into notebook-friendly series."""
    if reducer not in {"mean", "median"}:
        raise ValueError("reducer must be 'mean' or 'median'.")

    model_h5_path = _resolve_model_h5_path(source)
    reducer_fn = np.nanmean if reducer == "mean" else np.nanmedian

    with h5py.File(model_h5_path, "r") as handle:
        resolved_country_code = _resolve_single_country_code(handle, country_code)
        household_group = handle[f"{resolved_country_code}/households"]

        series_by_name: dict[str, np.ndarray] = {}
        expected_shape: tuple[int, int] | None = None
        for output_name, dataset_name in PERMANENT_INCOME_LOG_RATIO_DATASETS.items():
            if dataset_name not in household_group:
                raise KeyError(
                    f"HDF5 dataset '{dataset_name}' is missing. Re-run the simulation with the updated diagnostics."
                )
            values = np.asarray(household_group[dataset_name], dtype=float)
            if values.ndim != 2:
                raise ValueError(f"Expected 2D household time series for '{dataset_name}', got shape {values.shape}.")
            if expected_shape is None:
                expected_shape = values.shape
            elif values.shape != expected_shape:
                raise ValueError(
                    f"Permanent-income decomposition datasets must share the same shape, got {values.shape} "
                    f"and {expected_shape}."
                )
            series_by_name[output_name] = reducer_fn(values, axis=1)

        if include_log_real_pc_income:
            series_by_name["log_real_pc_income_t"] = _read_log_real_pc_income_series_from_h5(
                handle,
                resolved_country_code,
            )
            series_by_name['real_pc_income_idx']=np.exp(series_by_name['log_real_pc_income_t'])


    decomposition_df = pd.DataFrame(series_by_name, index=pd.RangeIndex(expected_shape[0], name="period"))
    decomposition_df.attrs["country_code"] = resolved_country_code
    decomposition_df.attrs["model_h5_path"] = str(model_h5_path)
    decomposition_df.attrs["reducer"] = reducer
    return decomposition_df


def plot_permanent_income_log_ratio_decomposition(
    source: str | Path | SimulationRunResult | BenchmarkResult,
    *,
    country_code: str | None = None,
    reducer: str = "mean",
    columns: list[str] | tuple[str, ...] | None = None,
    title: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Plot the aggregate ``ln(y^p / y) = zeta * posterior_mean + common_log_ratio`` decomposition."""
    default_columns = list(PERMANENT_INCOME_LOG_RATIO_DATASETS)
    selected_columns = default_columns if columns is None else list(columns)
    if not selected_columns:
        raise ValueError("columns must contain at least one decomposition series.")
    unknown_columns = [column for column in selected_columns if column not in PERMANENT_INCOME_LOG_RATIO_LABELS]
    if unknown_columns:
        raise ValueError(
            "Unknown columns requested: "
            + ", ".join(map(str, unknown_columns))
            + ". Valid options are: "
            + ", ".join(PERMANENT_INCOME_LOG_RATIO_LABELS)
            + "."
        )

    decomposition_df = build_permanent_income_log_ratio_decomposition_df(
        source,
        country_code=country_code,
        reducer=reducer,
        include_log_real_pc_income=True,
        # include_log_real_pc_income="log_real_pc_income_t" in selected_columns,
    )

    fig = go.Figure()
    for column in selected_columns:
        fig.add_trace(
            go.Scatter(
                x=decomposition_df.index,
                y=decomposition_df[column],
                mode="lines",
                name=PERMANENT_INCOME_LOG_RATIO_LABELS[column],
            )
        )
    fig.update_layout(
        title_text=title or f"Permanent-income log-ratio decomposition ({reducer})",
        template="plotly_white",
        xaxis_title="Period",
        yaxis_title="Value",
    )
    if show:
        fig.show()
    return fig


def build_permanent_income_forecast_contribution_table(
    source: Simulation | SimulationRunResult | BenchmarkResult,
    *,
    country_code: str,
    periods: list[int] | tuple[int, ...] = (1, 2, 3, 4),
    include_fixed: bool = True,
) -> pd.DataFrame:
    """Return regressor-level common-forecast contributions for selected simulation periods."""
    try:
        simulation_source_name_map = FORECAST_READER_TO_SIMULATION_SOURCE_NAME
    except NameError:  # pragma: no cover - defensive fallback for stale notebook module state
        from macro_data.readers.permanent_income_mapping import (
            FORECAST_READER_TO_SIMULATION_SOURCE_NAME as simulation_source_name_map,
        )

    model = _resolve_simulation_model(source)
    country = model.countries[country_code]
    forecast_inputs = getattr(country, "_permanent_income_forecast_inputs", None)
    design_matrix = getattr(country, "_permanent_income_design_matrix", None)
    if forecast_inputs is None or design_matrix is None:
        raise ValueError(f"Permanent-income forecast inputs are unavailable for country {country_code!r}.")

    requested_periods = [int(period) for period in periods]
    if not requested_periods:
        raise ValueError("periods must contain at least one simulation period.")
    if any(period < 0 for period in requested_periods):
        raise ValueError("periods must be non-negative.")

    cpi_fixed_basket = np.asarray(
        [np.asarray(value).reshape(-1)[0] for value in country.economy.ts.dicts["cpi_fixed_basket"]],
        dtype=float,
    )
    unemployment_rate = np.asarray(
        [np.asarray(value).reshape(-1)[0] for value in country.economy.ts.dicts["unemployment_rate"]],
        dtype=float,
    )
    policy_rate_full = np.asarray(
        [np.asarray(value).reshape(-1)[0] for value in country.central_bank.ts.dicts["policy_rate"]],
        dtype=float,
    )
    n_individuals_history = np.asarray(
        [np.asarray(value).reshape(-1)[0] for value in country.individuals.ts.dicts["n_individuals"]],
        dtype=float,
    )
    total_income_history = np.asarray(
        [float(np.sum(value)) for value in country.households.ts.dicts["income"]],
        dtype=float,
    )
    real_pc_income_levels = total_income_history / cpi_fixed_basket / n_individuals_history
    real_pc_income = rebase_real_pc_income_index(real_pc_income_levels, base_period_index=0)

    max_period = len(real_pc_income) - 1
    if any(period > max_period for period in requested_periods):
        raise ValueError(f"Requested period exceeds available history: max period is {max_period}.")

    rows: list[dict[str, float | int | str]] = []
    estimation_epoch = design_matrix.index.min()
    coefficients = forecast_inputs.coefficient_table["coefficient"].astype(float)

    for period in requested_periods:
        history_length = period + 1
        current_period = country.start_period + period
        policy_rate = policy_rate_full[:history_length]
        sources = PermanentIncomeSimulationSources(
            current_period=current_period,
            real_pc_income=real_pc_income[:history_length],
            policy_rate=policy_rate,
            cpi_fixed_basket=cpi_fixed_basket[:history_length],
            unemployment_rate=unemployment_rate[:history_length],
        )
        x_t = build_permanent_income_forecast_regressors(
            sources=sources,
            design_matrix=design_matrix,
            start_period=country.start_period,
            estimation_epoch=estimation_epoch,
        )
        forecast = forecast_common_permanent_income(x_t, forecast_inputs)
        contributions = x_t.astype(float) * coefficients
        for regressor in x_t.index:
            simulation_source = simulation_source_name_map.get(regressor, "unknown")
            rows.append(
                {
                    "period": period,
                    "date": str(current_period),
                    "regressor": regressor,
                    "simulation_source": simulation_source,
                    "is_fixed": simulation_source == "frozen_design_matrix_initial_period",
                    "x_t": float(x_t[regressor]),
                    "coefficient": float(coefficients[regressor]),
                    "contribution": float(contributions[regressor]),
                    "point_forecast": float(forecast.point_forecast),
                }
            )

    contribution_df = pd.DataFrame(rows)
    if not include_fixed:
        contribution_df = contribution_df.loc[~contribution_df["is_fixed"]].reset_index(drop=True)
    contribution_df.attrs["country_code"] = country_code
    contribution_df.attrs["periods"] = requested_periods
    contribution_df.attrs["include_fixed"] = include_fixed
    return contribution_df


def _resolve_runtime_config(config: NotebookRunConfig) -> tuple[Config, Path, Path, Path]:
    cfg = Config.from_env()
    cfg.config_dir = _resolve_run_model_path(config.config_dir or cfg.config_dir)
    if config.country_iso3 is not None:
        cfg.country_iso3 = config.country_iso3
    cfg.seed = int(config.seed)
    cfg.t_max = int(config.t_max)

    raw_data_path = Path(config.raw_data_path) if config.raw_data_path is not None else Path(cfg.raw_data_path)
    output_dir = Path(config.output_dir) if config.output_dir is not None else Path(cfg.output_path)
    raw_data_path = _resolve_run_model_path(raw_data_path) if not raw_data_path.is_absolute() else raw_data_path
    output_dir = _resolve_run_model_path(output_dir) if not output_dir.is_absolute() else output_dir
    cfg.raw_data_path = raw_data_path
    cfg.output_path = output_dir
    return cfg, raw_data_path, output_dir, cfg.config_dir


def _filename_only(value: str | Path, field_name: str) -> str:
    """Return a filename after rejecting values that include directories."""
    file_name = str(value)
    if Path(file_name).name != file_name:
        raise ValueError(f"{field_name} must be a file name, not a path: {file_name}")
    return file_name


def _file_fingerprint(path: Path) -> dict[str, Any]:
    """Build a lightweight fingerprint for cache invalidation."""
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _benchmark_data_cache_spec(cfg: Config, raw_data_path: Path, config: NotebookRunConfig) -> dict[str, Any]:
    """Describe the synthetic benchmark data inputs stored in the pickle cache."""
    data_config_path = Path(cfg.config_dir) / f"data_config_{cfg.country_iso3}.yaml"
    country_config_path = Path(cfg.config_dir) / f"country_config_{cfg.country_iso3}.yaml"
    return {
        "country_iso3": cfg.country_iso3,
        "raw_data_path": str(raw_data_path),
        "single_hfcs_survey": config.single_hfcs_survey,
        "config_dir": str(cfg.config_dir),
        "data_config": _file_fingerprint(data_config_path),
        "country_config": _file_fingerprint(country_config_path),
    }


REQUIRED_INDIVIDUAL_POPULATION_COLUMNS = frozenset(
    {
        "Gender",
        "Age",
        "Education",
        "college",
        "Activity Status",
        "Employment Industry",
        "Employee Income",
        "Income from Unemployment Benefits",
        "Income",
        "Corresponding Household ID",
        "Relation to Reference Person",
        "Corresponding Invested Firm",
        "Corresponding Invested Bank",
    }
)


def _read_json_cache_spec(path: Path) -> dict[str, Any] | None:
    """Read cache metadata, returning None for missing or malformed metadata."""
    try:
        with path.open() as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _write_json_cache_spec(path: Path, spec: Mapping[str, Any]) -> None:
    """Persist cache metadata beside a pickle cache."""
    with path.open("w") as f:
        json.dump(dict(spec), f, indent=2, sort_keys=True)


def _benchmark_specs_match(cached_spec: Mapping[str, Any] | None, requested_spec: Mapping[str, Any]) -> bool:
    """Return True when a benchmark dataframe cache matches the requested scenario."""
    if not isinstance(cached_spec, Mapping):
        return False
    for key, requested_value in requested_spec.items():
        if key == "n_industries" and requested_value is None:
            continue
        if cached_spec.get(key) != requested_value:
            return False
    return True


def _requires_population_schema_cache_rebuild(data: DataWrapper, country_iso3: str) -> bool:
    """Return true when cached synthetic populations predate required columns."""
    synthetic_country = data.synthetic_countries.get(country_iso3)
    population = getattr(synthetic_country, "population", None)
    individual_data = getattr(population, "individual_data", None)
    columns = getattr(individual_data, "columns", None)
    if columns is None:
        return True
    return not REQUIRED_INDIVIDUAL_POPULATION_COLUMNS.issubset(set(columns))


def _load_data_config(cfg: Config) -> DataConfiguration:
    config_path = Path(cfg.config_dir) / f"data_config_{cfg.country_iso3}.yaml"
    with config_path.open() as f:
        data_config_dict = yaml.safe_load(f)
    data_config_dict["country_configs"] = split_country_configs(data_config_dict["country_configs"])
    return DataConfiguration(**data_config_dict)


def _load_country_config(cfg: Config) -> CountryConfiguration:
    config_path = Path(cfg.config_dir) / f"country_config_{cfg.country_iso3}.yaml"
    return load_country_configuration(config_path, country_iso3=cfg.country_iso3)


def _parse_override_path(path: str) -> tuple[str, ...]:
    parts: list[str] = []
    token = []
    i = 0
    while i < len(path):
        char = path[i]
        if char == ".":
            if token:
                parts.append("".join(token))
                token = []
            i += 1
            continue
        if char == "[":
            if token:
                parts.append("".join(token))
                token = []
            end = path.index("]", i)
            key = path[i + 1 : end].strip().strip("'\"")
            parts.append(key)
            i = end + 1
            continue
        token.append(char)
        i += 1
    if token:
        parts.append("".join(token))
    if not parts:
        raise ValueError(f"Could not parse override path: {path!r}")
    return tuple(parts)


def _set_nested_value(target: Any, path: tuple[str, ...], value: Any) -> None:
    current = target
    for key in path[:-1]:
        current = current[key] if isinstance(current, dict) else getattr(current, key)
    last = path[-1]
    if isinstance(current, dict):
        current[last] = value
    else:
        setattr(current, last, value)


def apply_country_config_overrides(country_cfg: CountryConfiguration, overrides: Mapping[str, Any] | None) -> None:
    """Apply notebook scenario overrides to a copied country configuration."""
    for path, value in (overrides or {}).items():
        _set_nested_value(country_cfg, _parse_override_path(path), value)
    if country_cfg.firms.functions.productivity_investment_planner.name == "TargetIntensityTFPInvestmentPlanner":
        growth_phi = country_cfg.firms.functions.productivity_growth.parameters.get("investment_effectiveness")
        if growth_phi is None:
            raise ValueError(
                "TargetIntensityTFPInvestmentPlanner requires "
                "productivity_growth.parameters['investment_effectiveness']; "
                "realised TFP effectiveness is the canonical phi used by the planner."
            )
        country_cfg.firms.functions.productivity_investment_planner.parameters["investment_effectiveness"] = growth_phi


def summarize_country_config(country_cfg: CountryConfiguration) -> dict[str, Any]:
    """Return the compact scenario summary shown in the notebook."""
    return {
        "productivity_growth": country_cfg.firms.functions.productivity_growth.name,
        "productivity_investment_planner": country_cfg.firms.functions.productivity_investment_planner.name,
        "labour_market": {
            "name": country_cfg.labour_market.functions.clearing.name,
            "parameters": country_cfg.labour_market.functions.clearing.parameters,
        },
        "policy_rate_rule": country_cfg.central_bank.functions.policy_rate.name,
        "benefit_rule": {
            "name": country_cfg.central_government.functions.social_benefits.name,
            "parameters": country_cfg.central_government.functions.social_benefits.parameters,
        },
        "government_consumption": {
            "name": country_cfg.government_entities.functions.consumption.name,
            "parameters": country_cfg.government_entities.functions.consumption.parameters,
        },
        "wage_setter": {
            "name": country_cfg.firms.functions.wage_setter.name,
            "parameters": country_cfg.firms.functions.wage_setter.parameters,
        },
        "assume_zero_noise": country_cfg.assume_zero_noise,
    }


def _safe_len(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return len(value)
    except TypeError:
        return None


def summarize_agent_counts(data: DataWrapper, country_code: str) -> dict[str, int | None]:
    """Return synthetic agent counts used by the notebook scenario."""
    synthetic_country = data.synthetic_countries[country_code]
    population = getattr(synthetic_country, "population", None)
    firms = getattr(synthetic_country, "firms", None)
    banks = getattr(synthetic_country, "banks", None)
    government_entities = getattr(synthetic_country, "government_entities", None)

    return {
        "industries": getattr(data, "n_industries", None),
        "firms": _safe_len(getattr(firms, "firm_data", None)),
        "households": _safe_len(getattr(population, "household_data", None)),
        "individuals": _safe_len(getattr(population, "individual_data", None)),
        "banks": getattr(banks, "number_of_banks", _safe_len(getattr(banks, "bank_data", None))),
        "government_entities": getattr(
            government_entities,
            "number_of_entities",
            _safe_len(getattr(government_entities, "gov_entity_data", None)),
        ),
        "central_bank": 1 if getattr(synthetic_country, "central_bank", None) is not None else None,
        "central_government": 1 if getattr(synthetic_country, "central_government", None) is not None else None,
        "rest_of_world": 1 if getattr(data, "synthetic_rest_of_the_world", None) is not None else None,
    }


def _requires_cfc_rate_cache_rebuild(data: DataWrapper, data_config: DataConfiguration) -> bool:
    """Return true when cached firm CFC rates predate stock-rate semantics."""
    country_configs = getattr(data_config, "country_configs", {})
    synthetic_countries = getattr(data, "synthetic_countries", {})
    for country, country_config in country_configs.items():
        firms_config = getattr(country_config, "firms_configuration", None)
        if getattr(firms_config, "capital_depreciation_accounting_mode", "none") != "eurostat_cfc":
            continue

        synthetic_country = synthetic_countries.get(str(country), synthetic_countries.get(country))
        synthetic_firms = getattr(synthetic_country, "firms", None)
        if getattr(synthetic_firms, "capital_depreciation_rate_basis", None) != "capital_stock":
            return True
    return False


def prepare_data(config: NotebookRunConfig) -> PreparedData:
    """Build or load notebook data and return the objects needed for simulation."""
    cfg, raw_data_path, output_dir, _ = _resolve_runtime_config(config)
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data path not found: {raw_data_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_pkl_path = output_dir / config.data_cache_name
    data_config = _load_data_config(cfg)

    if data_pkl_path.exists() and not config.force_rebuild_data:
        data = DataWrapper.init_from_pickle(str(data_pkl_path))
        if _requires_cfc_rate_cache_rebuild(data, data_config):
            print(f"Rebuilding stale CFC-rate data cache: {data_pkl_path}")
            creator = DataWrapper.from_config(
                configuration=data_config,
                raw_data_path=raw_data_path,
                single_hfcs_survey=config.single_hfcs_survey,
            )
            creator.save(str(data_pkl_path))
            data = DataWrapper.init_from_pickle(str(data_pkl_path))
        elif _requires_population_schema_cache_rebuild(data, cfg.country_iso3):
            print(f"Rebuilding stale population-schema data cache: {data_pkl_path}")
            creator = DataWrapper.from_config(
                configuration=data_config,
                raw_data_path=raw_data_path,
                single_hfcs_survey=config.single_hfcs_survey,
            )
            creator.save(str(data_pkl_path))
            data = DataWrapper.init_from_pickle(str(data_pkl_path))
        else:
            creator = data
    else:
        creator = DataWrapper.from_config(
            configuration=data_config,
            raw_data_path=raw_data_path,
            single_hfcs_survey=config.single_hfcs_survey,
        )
        creator.save(str(data_pkl_path))
        data = DataWrapper.init_from_pickle(str(data_pkl_path))

    national_accounts = data.synthetic_countries[cfg.country_iso3].exogenous_data.national_accounts
    print(
        {
            "seed": cfg.seed,
            "country": cfg.country_iso3,
            "t_max": cfg.t_max,
            "raw_data_path": str(raw_data_path),
            "output_dir": str(output_dir),
            "data_cache": str(data_pkl_path),
        }
    )
    return PreparedData(
        cfg=cfg,
        data_config=data_config,
        creator=creator,
        data=data,
        national_accounts=national_accounts,
        data_pkl_path=data_pkl_path,
        output_dir=output_dir,
        raw_data_path=raw_data_path,
    )


def build_country_config(
    data: DataWrapper,
    config: NotebookRunConfig,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, CountryConfiguration]:
    """Load, align, and optionally override the country configuration."""
    cfg, _, _, _ = _resolve_runtime_config(config)
    country_cfg = _load_country_config(cfg)
    agent_counts = summarize_agent_counts(data, cfg.country_iso3)
    country_cfg = align_country_configuration_to_data(
        country_cfg,
        n_industries=data.n_industries,
        n_firms=agent_counts["firms"],
    )
    apply_country_config_overrides(country_cfg, overrides)
    country_configurations = {cfg.country_iso3: country_cfg}
    summary = summarize_country_config(country_cfg)
    summary["agent_counts"] = agent_counts
    print("Configuration summary")
    pprint(summary, sort_dicts=False)
    return country_configurations


def run_single_simulation(
    data: DataWrapper,
    country_configurations: dict[str, CountryConfiguration],
    config: NotebookRunConfig,
) -> SimulationRunResult:
    """Run one simulation and return the model plus canonical output dataframe."""
    cfg, raw_data_path, output_dir, _ = _resolve_runtime_config(config)
    random.seed(cfg.seed)
    model_file_name = _filename_only(config.model_file_name or f"simulation_{cfg.country_iso3}.h5", "model_file_name")
    model_h5_path = output_dir / model_file_name

    simulation_config = SimulationConfiguration(
        country_configurations=country_configurations,
        t_max=cfg.t_max,
        seed=cfg.seed,
    )
    data_paths = DataPaths.default_paths(raw_data_path, [data.configuration.year])
    model = Simulation.from_datawrapper(
        datawrapper=data, simulation_configuration=simulation_config, data_paths=data_paths
    )
    model.run()
    model.save(save_dir=output_dir, file_name=model_h5_path.name)
    df_base = build_macro_output_df(model, country_code=cfg.country_iso3)
    print(f"Simulation complete. Output saved to: {model_h5_path}")
    return SimulationRunResult(model=model, df_base=df_base, model_h5_path=model_h5_path)


def run_benchmark(
    config: NotebookRunConfig,
    overrides: Mapping[str, Any] | None = None,
) -> BenchmarkResult:
    """Run or load the notebook benchmark scenario."""
    cfg, raw_data_path, output_dir, _ = _resolve_runtime_config(config)
    benchmark_overrides = overrides if overrides is not None else config.benchmark_overrides
    output_dir.mkdir(parents=True, exist_ok=True)
    data_pkl_path = output_dir / _filename_only(config.benchmark_data_cache_name, "benchmark_data_cache_name")
    data_cache_spec_path = data_pkl_path.with_suffix(data_pkl_path.suffix + ".meta.json")
    df_benchmark_path = output_dir / _filename_only(
        config.benchmark_df_cache_name or f"{cfg.country_iso3}_df_benchmark.pkl",
        "benchmark_df_cache_name",
    )
    model_h5_path = output_dir / _filename_only(
        config.benchmark_model_file_name or f"{cfg.country_iso3}_benchmark.h5",
        "benchmark_model_file_name",
    )
    data_cache_spec = _benchmark_data_cache_spec(cfg, raw_data_path, config)
    benchmark_spec = {
        "country_iso3": cfg.country_iso3,
        "seed": cfg.seed,
        "t_max": cfg.t_max,
        "raw_data_path": str(raw_data_path),
        "single_hfcs_survey": config.single_hfcs_survey,
        "data_cache_spec": data_cache_spec,
        "n_industries": None,
        "overrides": dict(benchmark_overrides or {}),
    }

    if df_benchmark_path.exists() and not config.force_rerun_benchmark:
        df_benchmark = pd.read_pickle(df_benchmark_path)
        cached_benchmark_spec = df_benchmark.attrs.get("benchmark_spec")
        if _benchmark_specs_match(cached_benchmark_spec, benchmark_spec):
            print(f"Loaded cached benchmark dataframe from: {df_benchmark_path}")
            return BenchmarkResult(
                model=None,
                df_benchmark=df_benchmark,
                model_h5_path=model_h5_path,
                df_benchmark_path=df_benchmark_path,
                benchmark_spec=dict(cached_benchmark_spec),
                loaded_from_cache=True,
            )
        print(f"Rebuilding stale benchmark dataframe cache: {df_benchmark_path}")

    data_config = _load_data_config(cfg)
    cached_data_spec = _read_json_cache_spec(data_cache_spec_path)
    if data_pkl_path.exists() and cached_data_spec == data_cache_spec and not config.force_rerun_benchmark:
        data = DataWrapper.init_from_pickle(str(data_pkl_path))
        if _requires_cfc_rate_cache_rebuild(data, data_config):
            print(f"Rebuilding stale benchmark CFC-rate data cache: {data_pkl_path}")
            data = DataWrapper.from_config(
                configuration=data_config,
                raw_data_path=raw_data_path,
                single_hfcs_survey=config.single_hfcs_survey,
            )
            data.save(str(data_pkl_path))
            _write_json_cache_spec(data_cache_spec_path, data_cache_spec)
            data = DataWrapper.init_from_pickle(str(data_pkl_path))
        elif _requires_population_schema_cache_rebuild(data, cfg.country_iso3):
            print(f"Rebuilding stale benchmark population-schema data cache: {data_pkl_path}")
            data = DataWrapper.from_config(
                configuration=data_config,
                raw_data_path=raw_data_path,
                single_hfcs_survey=config.single_hfcs_survey,
            )
            data.save(str(data_pkl_path))
            _write_json_cache_spec(data_cache_spec_path, data_cache_spec)
            data = DataWrapper.init_from_pickle(str(data_pkl_path))
    else:
        if data_pkl_path.exists() and not config.force_rerun_benchmark:
            print(f"Rebuilding stale benchmark data cache: {data_pkl_path}")
        data = DataWrapper.from_config(
            configuration=data_config,
            raw_data_path=raw_data_path,
            single_hfcs_survey=config.single_hfcs_survey,
        )
        data.save(str(data_pkl_path))
        _write_json_cache_spec(data_cache_spec_path, data_cache_spec)
        data = DataWrapper.init_from_pickle(str(data_pkl_path))

    benchmark_spec["n_industries"] = data.n_industries
    country_configurations = build_country_config(
        data=data,
        config=config,
        overrides=benchmark_overrides,
    )
    benchmark_config = deepcopy(config)
    model_result = run_single_simulation(
        data=data,
        country_configurations=country_configurations,
        config=NotebookRunConfig(
            seed=benchmark_config.seed,
            t_max=benchmark_config.t_max,
            country_iso3=benchmark_config.country_iso3,
            raw_data_path=benchmark_config.raw_data_path,
            output_dir=benchmark_config.output_dir,
            config_dir=benchmark_config.config_dir,
            model_file_name=model_h5_path.name,
            single_hfcs_survey=benchmark_config.single_hfcs_survey,
        ),
    )
    df_benchmark = model_result.df_base
    df_benchmark.attrs["benchmark_spec"] = benchmark_spec
    df_benchmark.to_pickle(df_benchmark_path)
    print(f"Benchmark dataframe cached at: {df_benchmark_path}")
    return BenchmarkResult(
        model=model_result.model,
        df_benchmark=df_benchmark,
        model_h5_path=model_h5_path,
        df_benchmark_path=df_benchmark_path,
        benchmark_spec=benchmark_spec,
        loaded_from_cache=False,
    )
