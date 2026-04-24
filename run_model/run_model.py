from __future__ import annotations

import argparse
import logging
import random
import sys
import warnings
from pathlib import Path

import yaml

RUN_MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_MODEL_DIR.parent
for path in (str(REPO_ROOT), str(RUN_MODEL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from config import Config  # noqa: E402
from src.helpers import align_country_configuration_to_data  # noqa: E402
from src.visual_helpers import build_macro_output_df  # noqa: E402

from macro_data import DataWrapper  # noqa: E402
from macro_data.configuration import DataConfiguration, split_country_configs  # noqa: E402
from macromodel.configurations import CountryConfiguration, SimulationConfiguration  # noqa: E402
from macromodel.simulation import Simulation  # noqa: E402

logging.getLogger().setLevel(logging.ERROR)

DEFAULT_SEED = 15
DEFAULT_T_MAX = 50


def _resolve_run_model_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return RUN_MODEL_DIR / resolved


def main(seed: int | None = DEFAULT_SEED, t_max: int | None = DEFAULT_T_MAX) -> dict[str, object]:
    # Optional overrides (None means use Config/default env values)
    country_override = None
    raw_data_path_override = None
    output_dir_override = None

    cfg = Config.from_env()
    cfg.config_dir = _resolve_run_model_path(cfg.config_dir)
    if country_override is not None:
        cfg.country_iso3 = country_override
    if t_max is not None:
        cfg.t_max = t_max
    if seed is not None:
        cfg.seed = seed

    random.seed(cfg.seed)

    raw_data_path = Path(raw_data_path_override) if raw_data_path_override else cfg.raw_data_path
    output_dir = Path(output_dir_override) if output_dir_override else cfg.output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    data_pkl_path = output_dir / "data.pkl"
    model_h5_path = output_dir / "multi_country_simulation.h5"

    print(
        {
            "seed": cfg.seed,
            "country": cfg.country_iso3,
            "t_max": cfg.t_max,
            "raw_data_path": str(raw_data_path),
            "output_dir": str(output_dir),
        }
    )

    # Data Preparation
    assert raw_data_path.exists(), f"Raw data path not found: {raw_data_path}"

    with (cfg.config_dir / f"data_config_{cfg.country_iso3}.yaml").open() as f:
        data_config_dict = yaml.safe_load(f)
    data_config_dict["country_configs"] = split_country_configs(data_config_dict["country_configs"])
    data_config = DataConfiguration(**data_config_dict)

    creator = DataWrapper.from_config(
        configuration=data_config,
        raw_data_path=raw_data_path,
        single_hfcs_survey=False,
    )
    creator.save(str(data_pkl_path))
    data = DataWrapper.init_from_pickle(str(data_pkl_path))

    synthetic_country = creator.synthetic_countries[cfg.country_iso3]
    national_accounts = synthetic_country.exogenous_data.national_accounts

    # Load country configuration
    with (cfg.config_dir / f"country_config_{cfg.country_iso3}.yaml").open() as f:
        country_config_dict = yaml.safe_load(f)
    country_cfg = CountryConfiguration(**country_config_dict[cfg.country_iso3])

    country_cfg = align_country_configuration_to_data(country_cfg, n_industries=data.n_industries)
    country_configurations = {cfg.country_iso3: country_cfg}

    # Modify loaded country config
    country_cfg = country_configurations[cfg.country_iso3]
    country_cfg.firms.functions.productivity_growth.name = "SimpleTFPGrowth"  # "NoOpTFPGrowth"
    country_cfg.labour_market.functions.clearing.name = "PolednaLabourMarketClearer"
    country_cfg.central_bank.functions.policy_rate.name = "SmoothTaylorRule"
    country_cfg.government_entities.functions.consumption.name = "AutoregressiveGovernmentConsumptionSetter"
    country_cfg.central_government.functions.social_benefits.name = "ConstantSocialBenefitsSetter"

    print("Configuration summary")
    print(
        {
            "productivity_growth": country_cfg.firms.functions.productivity_growth.name,
            "labour_market_clearer": country_cfg.labour_market.functions.clearing.name,
            "policy_rate_rule": country_cfg.central_bank.functions.policy_rate.name,
            "benefit_rule": country_cfg.central_government.functions.social_benefits.name,
        }
    )

    # Run single Simulation
    configuration = SimulationConfiguration(
        country_configurations=country_configurations,
        t_max=cfg.t_max,
        seed=cfg.seed,
    )

    model = Simulation.from_datawrapper(
        datawrapper=data,
        simulation_configuration=configuration,
    )

    warnings.simplefilter("always", RuntimeWarning)

    model.run()
    model.save(save_dir=output_dir, file_name=model_h5_path.name)

    print(f"Simulation complete. Output saved to: {model_h5_path}")

    df_base = build_macro_output_df(model, country_code=cfg.country_iso3)

    return {
        "cfg": cfg,
        "data": data,
        "model": model,
        "df_base": df_base,
        "national_accounts": national_accounts,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the macro model through the single-simulation step.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Simulation seed. Default: {DEFAULT_SEED}.")
    parser.add_argument("--t-max", type=int, default=DEFAULT_T_MAX, help=f"Simulation horizon. Default: {DEFAULT_T_MAX}.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(seed=args.seed, t_max=args.t_max)
