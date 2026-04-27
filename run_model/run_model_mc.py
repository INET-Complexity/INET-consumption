from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import yaml

RUN_MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_MODEL_DIR.parent
for path in (str(REPO_ROOT), str(RUN_MODEL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from config import Config  # noqa: E402
from src.helpers import align_country_configuration_to_data  # noqa: E402
from src.monte_carlo import run_seeded_monte_carlo  # noqa: E402

from macro_data import DataWrapper  # noqa: E402
from macro_data.configuration import DataConfiguration, split_country_configs  # noqa: E402
from macromodel.configurations import CountryConfiguration  # noqa: E402

logging.getLogger().setLevel(logging.ERROR)

DEFAULT_T_MAX = 50
DEFAULT_SEEDS = [19, 23, 27, 32, 37, 43, 54, 57, 71, 85, 98]
DEFAULT_GOVERNMENT_CONSUMPTION_SETTER = "AutoregressiveGovernmentConsumptionSetter"
DEFAULT_GOVERNMENT_SECTORAL_WEIGHTS = "previous_desired"
GOVERNMENT_CONSUMPTION_SETTER_CHOICES = (
    "AutoregressiveGovernmentConsumptionSetter",
    "ConstantGrowthGovernmentConsumptionSetter",
    "ExogenousGovernmentConsumptionSetter",
)
GOVERNMENT_SECTORAL_WEIGHTS_CHOICES = ("previous_desired", "initial")


def _resolve_run_model_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return RUN_MODEL_DIR / resolved


def _parse_optional_bool(value: str | None) -> bool:
    if value is None:
        return True
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}.")


def _validate_unique_seeds(seeds: list[int]) -> list[int]:
    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise argparse.ArgumentTypeError("At least one seed must be provided.")
    if len(set(seed_list)) != len(seed_list):
        raise argparse.ArgumentTypeError("Seeds must be unique.")
    return seed_list


def main(
    seeds: list[int] | None = None,
    t_max: int | None = DEFAULT_T_MAX,
    government_consumption_setter: str = DEFAULT_GOVERNMENT_CONSUMPTION_SETTER,
    assume_zero_noise: bool | None = None,
    government_sectoral_weights: str = DEFAULT_GOVERNMENT_SECTORAL_WEIGHTS,
    government_consumption_consistency: float | None = None,
    n_jobs: int = -1,
    backend: str = "loky",
    verbose: int = 0,
    batch_size: int = 1,
    output_file: str | Path | None = None,
) -> dict[str, object]:
    seed_list = _validate_unique_seeds(DEFAULT_SEEDS if seeds is None else seeds)

    country_override = None
    raw_data_path_override = None
    output_dir_override = None

    cfg = Config.from_env()
    cfg.config_dir = _resolve_run_model_path(cfg.config_dir)
    if country_override is not None:
        cfg.country_iso3 = country_override
    if t_max is not None:
        cfg.t_max = t_max

    random.seed(cfg.seed)

    raw_data_path = Path(raw_data_path_override) if raw_data_path_override else cfg.raw_data_path
    output_dir = Path(output_dir_override) if output_dir_override else cfg.output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    data_pkl_path = output_dir / "data.pkl"
    output_path = Path(output_file) if output_file is not None else output_dir / f"monte_carlo_{cfg.country_iso3}.pkl"
    if not output_path.is_absolute():
        output_path = output_dir / output_path

    print(
        {
            "seeds": seed_list,
            "country": cfg.country_iso3,
            "t_max": cfg.t_max,
            "raw_data_path": str(raw_data_path),
            "output_dir": str(output_dir),
            "output_file": str(output_path),
            "n_jobs": n_jobs,
            "backend": backend,
            "batch_size": batch_size,
        }
    )

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

    with (cfg.config_dir / f"country_config_{cfg.country_iso3}.yaml").open() as f:
        country_config_dict = yaml.safe_load(f)
    country_cfg = CountryConfiguration(**country_config_dict[cfg.country_iso3])

    country_cfg = align_country_configuration_to_data(country_cfg, n_industries=data.n_industries)
    country_configurations = {cfg.country_iso3: country_cfg}

    country_cfg = country_configurations[cfg.country_iso3]
    country_cfg.firms.functions.productivity_growth.name = "SimpleTFPGrowth"
    country_cfg.labour_market.functions.clearing.name = "PolednaLabourMarketClearer"
    country_cfg.central_bank.functions.policy_rate.name = "SmoothTaylorRule"
    country_cfg.government_entities.functions.consumption.name = government_consumption_setter
    if government_consumption_consistency is not None:
        country_cfg.government_entities.functions.consumption.parameters[
            "consistency"
        ] = government_consumption_consistency
    if government_sectoral_weights != DEFAULT_GOVERNMENT_SECTORAL_WEIGHTS:
        country_cfg.government_entities.functions.consumption.parameters[
            "sectoral_weights"
        ] = government_sectoral_weights
    country_cfg.central_government.functions.social_benefits.name = "ConstantSocialBenefitsSetter"
    if assume_zero_noise is not None:
        country_cfg.assume_zero_noise = assume_zero_noise

    print("Configuration summary")
    print(
        {
            "productivity_growth": country_cfg.firms.functions.productivity_growth.name,
            "labour_market_clearer": country_cfg.labour_market.functions.clearing.name,
            "policy_rate_rule": country_cfg.central_bank.functions.policy_rate.name,
            "government_consumption_setter": country_cfg.government_entities.functions.consumption.name,
            "government_sectoral_weights": country_cfg.government_entities.functions.consumption.parameters.get(
                "sectoral_weights", DEFAULT_GOVERNMENT_SECTORAL_WEIGHTS
            ),
            "government_consumption_consistency": country_cfg.government_entities.functions.consumption.parameters.get(
                "consistency"
            ),
            "benefit_rule": country_cfg.central_government.functions.social_benefits.name,
            "assume_zero_noise": country_cfg.assume_zero_noise,
        }
    )

    mc = run_seeded_monte_carlo(
        datawrapper=data,
        country_configurations=country_configurations,
        country_code=cfg.country_iso3,
        seeds=seed_list,
        t_max=cfg.t_max,
        n_jobs=n_jobs,
        backend=backend,
        verbose=verbose,
        batch_size=batch_size,
    )
    mc.to_pickle(str(output_path))

    print(f"Monte Carlo complete. Combined output saved to: {output_path}")

    return {
        "cfg": cfg,
        "data": data,
        "country_configurations": country_configurations,
        "mc": mc,
        "output_path": output_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run seeded Monte Carlo simulations for the macro model.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"Unique integer simulation seeds. Default: {' '.join(str(seed) for seed in DEFAULT_SEEDS)}.",
    )
    parser.add_argument(
        "--t-max", type=int, default=DEFAULT_T_MAX, help=f"Simulation horizon. Default: {DEFAULT_T_MAX}."
    )
    parser.add_argument(
        "--government-consumption-setter",
        choices=GOVERNMENT_CONSUMPTION_SETTER_CHOICES,
        default=DEFAULT_GOVERNMENT_CONSUMPTION_SETTER,
        help=(
            "Government-consumption setter to use. "
            f"Default: {DEFAULT_GOVERNMENT_CONSUMPTION_SETTER}."
        ),
    )
    parser.add_argument(
        "--assume-zero-noise",
        nargs="?",
        const=True,
        default=None,
        type=_parse_optional_bool,
        help="Override the country configuration to suppress stochastic noise.",
    )
    parser.add_argument(
        "--government-sectoral-weights",
        choices=GOVERNMENT_SECTORAL_WEIGHTS_CHOICES,
        default=DEFAULT_GOVERNMENT_SECTORAL_WEIGHTS,
        help=(
            "Sectoral weights for autoregressive government consumption. "
            f"Default: {DEFAULT_GOVERNMENT_SECTORAL_WEIGHTS}."
        ),
    )
    parser.add_argument(
        "--government-consumption-consistency",
        choices=(0.0, 1.0),
        default=None,
        type=float,
        help="Override AR government-consumption consistency. Default: use country configuration.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of joblib workers. Default: all workers.")
    parser.add_argument("--backend", default="loky", help="Joblib backend. Default: loky.")
    parser.add_argument("--verbose", type=int, default=0, help="Joblib verbosity. Default: 0.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of seeds run sequentially inside each worker. Default: 1.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Pickle output path. Relative paths are resolved under the configured output directory.",
    )
    args = parser.parse_args()
    args.seeds = _validate_unique_seeds(args.seeds)
    return args


if __name__ == "__main__":
    args = _parse_args()
    main(
        seeds=args.seeds,
        t_max=args.t_max,
        government_consumption_setter=args.government_consumption_setter,
        assume_zero_noise=args.assume_zero_noise,
        government_sectoral_weights=args.government_sectoral_weights,
        government_consumption_consistency=args.government_consumption_consistency,
        n_jobs=args.n_jobs,
        backend=args.backend,
        verbose=args.verbose,
        batch_size=args.batch_size,
        output_file=args.output_file,
    )
