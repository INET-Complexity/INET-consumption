from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Any, Mapping

import pandas as pd
import yaml
from config import Config
from src.helpers import align_country_configuration_to_data
from src.visual_helpers import build_macro_output_df

from macro_data import DataWrapper
from macro_data.configuration import DataConfiguration, split_country_configs
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
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


def _resolve_run_model_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return RUN_MODEL_DIR / resolved


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


def _load_data_config(cfg: Config) -> DataConfiguration:
    config_path = Path(cfg.config_dir) / f"data_config_{cfg.country_iso3}.yaml"
    with config_path.open() as f:
        data_config_dict = yaml.safe_load(f)
    data_config_dict["country_configs"] = split_country_configs(data_config_dict["country_configs"])
    return DataConfiguration(**data_config_dict)


def _load_country_config(cfg: Config) -> CountryConfiguration:
    config_path = Path(cfg.config_dir) / f"country_config_{cfg.country_iso3}.yaml"
    with config_path.open() as f:
        country_config_dict = yaml.safe_load(f)
    return CountryConfiguration(**country_config_dict[cfg.country_iso3])


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
    cfg, _, output_dir, _ = _resolve_runtime_config(config)
    random.seed(cfg.seed)
    model_file_name = config.model_file_name or f"simulation_{cfg.country_iso3}.h5"
    model_h5_path = output_dir / model_file_name

    simulation_config = SimulationConfiguration(
        country_configurations=country_configurations,
        t_max=cfg.t_max,
        seed=cfg.seed,
    )
    model = Simulation.from_datawrapper(datawrapper=data, simulation_configuration=simulation_config)
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
    data_pkl_path = output_dir / config.benchmark_data_cache_name
    df_benchmark_path = output_dir / (config.benchmark_df_cache_name or f"{cfg.country_iso3}_df_benchmark.pkl")
    model_h5_path = output_dir / (config.benchmark_model_file_name or f"{cfg.country_iso3}_benchmark.h5")
    benchmark_spec = {
        "country_iso3": cfg.country_iso3,
        "seed": cfg.seed,
        "t_max": cfg.t_max,
        "raw_data_path": str(raw_data_path),
        "single_hfcs_survey": config.single_hfcs_survey,
        "n_industries": None,
        "overrides": dict(benchmark_overrides or {}),
    }

    if df_benchmark_path.exists() and not config.force_rerun_benchmark:
        df_benchmark = pd.read_pickle(df_benchmark_path)
        print(f"Loaded cached benchmark dataframe from: {df_benchmark_path}")
        return BenchmarkResult(
            model=None,
            df_benchmark=df_benchmark,
            model_h5_path=model_h5_path,
            df_benchmark_path=df_benchmark_path,
            benchmark_spec=df_benchmark.attrs.get("benchmark_spec", benchmark_spec),
            loaded_from_cache=True,
        )

    data_config = _load_data_config(cfg)
    if data_pkl_path.exists() and not config.force_rerun_benchmark:
        data = DataWrapper.init_from_pickle(str(data_pkl_path))
        if _requires_cfc_rate_cache_rebuild(data, data_config):
            print(f"Rebuilding stale benchmark CFC-rate data cache: {data_pkl_path}")
            data = DataWrapper.from_config(
                configuration=data_config,
                raw_data_path=raw_data_path,
                single_hfcs_survey=config.single_hfcs_survey,
            )
            data.save(str(data_pkl_path))
            data = DataWrapper.init_from_pickle(str(data_pkl_path))
    else:
        data = DataWrapper.from_config(
            configuration=data_config,
            raw_data_path=raw_data_path,
            single_hfcs_survey=config.single_hfcs_survey,
        )
        data.save(str(data_pkl_path))
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
