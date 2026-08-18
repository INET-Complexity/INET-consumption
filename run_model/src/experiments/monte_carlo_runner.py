import argparse
import logging
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

RUN_MODEL_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = RUN_MODEL_DIR.parent
for path in (str(REPO_ROOT), str(RUN_MODEL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from config import Config  # noqa: E402
from src.collapse_debug import summarize_government_bridge_run  # noqa: E402
from src.helpers import align_country_configuration_to_data  # noqa: E402
from src.monte_carlo import run_seeded_monte_carlo  # noqa: E402

from macro_data import DataWrapper  # noqa: E402
from macro_data.configuration import DataConfiguration, split_country_configs  # noqa: E402
from macro_data.readers.default_readers import DataPaths  # noqa: E402
from macromodel.configurations import CountryConfiguration, load_country_configuration  # noqa: E402

logging.getLogger().setLevel(logging.ERROR)

DEFAULT_T_MAX = 50
DEFAULT_SEEDS = [19, 23, 27, 32, 37, 43, 54, 57, 71, 85, 98]
DEFAULT_GOVERNMENT_BRIDGE_SEEDS = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
DEFAULT_GOVERNMENT_CONSISTENCY_INITIAL_WEIGHTS_SEEDS = list(range(12, 62))
DEFAULT_GOVERNMENT_BRIDGE_LABEL = "2026-04-27-government-consumption-bridge-mc"
DEFAULT_GOVERNMENT_CONSISTENCY_INITIAL_WEIGHTS_LABEL = (
    "2026-04-27-government-consumption-consistency-initial-weights-mc50"
)
DEFAULT_GOVERNMENT_CONSUMPTION_SETTER = "ExpectedGrowthGovernmentConsumptionSetter"
DEFAULT_GOVERNMENT_SECTORAL_WEIGHTS = "initial_fixed"
GOVERNMENT_CONSUMPTION_SETTER_CHOICES = (
    "AutoregressiveGovernmentConsumptionSetter",
    "AutoregressiveGrowthGovernmentConsumptionSetter",
    "ConstantGrowthGovernmentConsumptionSetter",
    "ExpectedGrowthGovernmentConsumptionSetter",
    "ExogenousGovernmentConsumptionSetter",
)
GOVERNMENT_SECTORAL_WEIGHTS_CHOICES = ("previous_desired", "initial", "initial_price_normalized", "initial_fixed")


@dataclass(frozen=True)
class GovernmentBridgeArm:
    name: str
    setter: str
    consistency: float
    sectoral_weights: str


GOVERNMENT_BRIDGE_ARMS = (
    GovernmentBridgeArm(
        "baseline_consistency1",
        "AutoregressiveGovernmentConsumptionSetter",
        1.0,
        "previous_desired",
    ),
    GovernmentBridgeArm(
        "consistency0_previous_desired",
        "AutoregressiveGovernmentConsumptionSetter",
        0.0,
        "previous_desired",
    ),
    GovernmentBridgeArm(
        "consistency1_initial_weights",
        "AutoregressiveGovernmentConsumptionSetter",
        1.0,
        "initial",
    ),
    GovernmentBridgeArm(
        "consistency0_initial_weights",
        "AutoregressiveGovernmentConsumptionSetter",
        0.0,
        "initial",
    ),
)
GOVERNMENT_CONSISTENCY_INITIAL_WEIGHTS_ARMS = (
    GovernmentBridgeArm("ar_consistency1_initial", "AutoregressiveGovernmentConsumptionSetter", 1.0, "initial"),
    GovernmentBridgeArm("ar_consistency0_initial", "AutoregressiveGovernmentConsumptionSetter", 0.0, "initial"),
    GovernmentBridgeArm(
        "argrowth_consistency1_initial",
        "AutoregressiveGrowthGovernmentConsumptionSetter",
        1.0,
        "initial",
    ),
    GovernmentBridgeArm(
        "argrowth_consistency1_initial_price_normalized",
        "AutoregressiveGrowthGovernmentConsumptionSetter",
        1.0,
        "initial_price_normalized",
    ),
    GovernmentBridgeArm(
        "argrowth_consistency1_initial_fixed",
        "AutoregressiveGrowthGovernmentConsumptionSetter",
        1.0,
        "initial_fixed",
    ),
    GovernmentBridgeArm(
        "expectedgrowth_initial_fixed",
        "ExpectedGrowthGovernmentConsumptionSetter",
        1.0,
        "initial_fixed",
    ),
    GovernmentBridgeArm(
        "argrowth_consistency0_initial",
        "AutoregressiveGrowthGovernmentConsumptionSetter",
        0.0,
        "initial",
    ),
    GovernmentBridgeArm("constantgrowth_initial", "ConstantGrowthGovernmentConsumptionSetter", 1.0, "initial"),
)
GOVERNMENT_BRIDGE_ARM_BY_NAME = {arm.name: arm for arm in GOVERNMENT_BRIDGE_ARMS}
GOVERNMENT_CONSISTENCY_INITIAL_WEIGHTS_ARM_BY_NAME = {
    arm.name: arm for arm in GOVERNMENT_CONSISTENCY_INITIAL_WEIGHTS_ARMS
}


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


def _scale_tfp_target_intensities(country_cfg: CountryConfiguration, multiplier: float) -> None:
    """Scale sector TFP target intensities in-memory for calibration experiments."""
    if multiplier <= 0:
        raise ValueError("TFP target-intensity multiplier must be strictly positive.")
    if multiplier == 1.0:
        return

    planner_params = country_cfg.firms.functions.productivity_investment_planner.parameters
    sector_targets = planner_params.get("sector_innovation_intensity")
    if sector_targets is None:
        raise ValueError("Cannot scale TFP targets: sector_innovation_intensity is not configured.")
    planner_params["sector_innovation_intensity"] = {
        sector: float(target) * multiplier for sector, target in sector_targets.items()
    }


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
    save_h5_dir: str | Path | None = None,
    tfp_target_intensity_multiplier: float = 1.0,
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

    country_cfg = load_country_configuration(
        cfg.config_dir / f"country_config_{cfg.country_iso3}.yaml",
        country_iso3=cfg.country_iso3,
    )

    synthetic_country = data.synthetic_countries[cfg.country_iso3]
    country_cfg = align_country_configuration_to_data(
        country_cfg,
        n_industries=data.n_industries,
        n_firms=len(synthetic_country.firms.firm_data),
    )
    _scale_tfp_target_intensities(country_cfg, tfp_target_intensity_multiplier)
    country_configurations = {cfg.country_iso3: country_cfg}

    country_cfg = country_configurations[cfg.country_iso3]
    country_cfg.firms.functions.productivity_growth.name = "SimpleTFPGrowth"
    country_cfg.labour_market.functions.clearing.name = "PolednaLabourMarketClearer"
    country_cfg.central_bank.functions.policy_rate.name = "SmoothTaylorRule"
    country_cfg.government_entities.functions.consumption.name = government_consumption_setter
    if government_consumption_consistency is not None:
        country_cfg.government_entities.functions.consumption.parameters["consistency"] = (
            government_consumption_consistency
        )
    if government_sectoral_weights != DEFAULT_GOVERNMENT_SECTORAL_WEIGHTS:
        country_cfg.government_entities.functions.consumption.parameters["sectoral_weights"] = (
            government_sectoral_weights
        )
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
            "debt_interest_rule": {
                "name": country_cfg.central_government.functions.debt_interest.name,
                "parameters": country_cfg.central_government.functions.debt_interest.parameters,
            },
            "assume_zero_noise": country_cfg.assume_zero_noise,
            "tfp_target_intensity_multiplier": tfp_target_intensity_multiplier,
        }
    )

    data_paths = DataPaths.default_paths(raw_data_path, [data_config.year])
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
        save_h5_dir=save_h5_dir,
        data_paths=data_paths,
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


def _nan_count(values: pd.Series) -> int:
    return int(values.notna().sum())


def _first_time_threshold_rows(summary: pd.DataFrame) -> pd.DataFrame:
    threshold_columns = [
        "government_fce_first_below_1m",
        "government_fce_first_zero",
        "desired_government_consumption_first_below_1bn",
        "desired_government_consumption_first_zero",
        "desired_government_consumption_first_drop_gt_50pct",
        "desired_government_consumption_first_jump_gt_50pct",
        "realised_desired_first_below_0_9",
        "realised_desired_first_below_0_5",
    ]
    rows = []
    for _, row in summary.iterrows():
        for column in threshold_columns:
            rows.append(
                {
                    "seed": int(row["seed"]),
                    "arm": row["arm"],
                    "threshold": column,
                    "first_time": row[column],
                    "triggered": pd.notna(row[column]),
                }
            )
    return pd.DataFrame(rows)


def _build_arm_aggregates(summary: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "desired_government_consumption_t0",
        "desired_government_consumption_t20",
        "desired_government_consumption_t21",
        "desired_government_consumption_t22",
        "desired_government_consumption_t50",
        "desired_government_consumption_cv_t1_t50",
        "gdp_t50_change",
        "government_fce_min",
        "firm_profits_t50",
        "unemployment_t50",
        "vacancy_t50",
        "debt_gdp_t50",
        "sector15_desired_share_t20",
        "sector15_desired_share_t21",
        "sector15_supply_cover_t20",
        "sector15_supply_cover_t21",
    ]
    grouped = summary.groupby("arm", sort=False)
    aggregates = grouped[metric_columns].agg(["mean", "median", "std", "min", "max"])
    aggregates.columns = [f"{metric}_{stat}" for metric, stat in aggregates.columns]
    aggregates = aggregates.reset_index()

    count_rows = []
    for arm, group in summary.groupby("arm", sort=False):
        runs = int(len(group))
        count_rows.append(
            {
                "arm": arm,
                "runs": runs,
                "government_fce_exact_zero_runs": _nan_count(group["government_fce_first_zero"]),
                "government_fce_exact_zero_share": _nan_count(group["government_fce_first_zero"]) / runs,
                "desired_government_below_1bn_runs": _nan_count(
                    group["desired_government_consumption_first_below_1bn"]
                ),
                "desired_government_below_1bn_share": _nan_count(
                    group["desired_government_consumption_first_below_1bn"]
                )
                / runs,
                "gdp_t50_change_below_minus_20pct_runs": int((group["gdp_t50_change"] < -20.0).sum()),
                "gdp_t50_change_below_minus_20pct_share": float((group["gdp_t50_change"] < -20.0).sum() / runs),
                "unemployment_t50_above_40pct_runs": int((group["unemployment_t50"] > 0.40).sum()),
                "unemployment_t50_above_40pct_share": float((group["unemployment_t50"] > 0.40).sum() / runs),
                "debt_gdp_t50_above_5_runs": int((group["debt_gdp_t50"] > 5.0).sum()),
                "debt_gdp_t50_above_5_share": float((group["debt_gdp_t50"] > 5.0).sum() / runs),
            }
        )
    counts = pd.DataFrame(count_rows)
    return aggregates.merge(counts, on="arm", how="left")


def _write_government_bridge_analysis(
    *,
    experiment_dir: Path,
    arms: list[GovernmentBridgeArm],
    seeds: list[int],
    country_code: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for arm in arms:
        for seed in seeds:
            h5_path = experiment_dir / arm.name / f"seed-{seed}" / "multi_country_simulation.h5"
            metrics = summarize_government_bridge_run(h5_path, country_code=country_code)
            rows.append({"seed": seed, "arm": arm.name, **metrics})

    summary = pd.DataFrame(rows)
    thresholds = _first_time_threshold_rows(summary)
    arm_aggregates = _build_arm_aggregates(summary)

    analysis_dir = experiment_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(analysis_dir / "summary.csv", index=False)
    thresholds.to_csv(analysis_dir / "thresholds.csv", index=False)
    arm_aggregates.to_csv(analysis_dir / "arm_aggregates.csv", index=False)

    return summary, thresholds, arm_aggregates


def run_government_bridge_experiment(
    *,
    seeds: list[int] | None,
    t_max: int | None,
    label: str,
    arm_names: list[str],
    arm_lookup: dict[str, GovernmentBridgeArm] | None = None,
    default_seeds: list[int] | None = None,
    n_jobs: int,
    backend: str,
    verbose: int,
    batch_size: int,
) -> dict[str, object]:
    if arm_lookup is None:
        arm_lookup = GOVERNMENT_BRIDGE_ARM_BY_NAME
    if default_seeds is None:
        default_seeds = DEFAULT_GOVERNMENT_BRIDGE_SEEDS

    seed_list = _validate_unique_seeds(default_seeds if seeds is None else seeds)
    arms = [arm_lookup[name] for name in arm_names]

    cfg = Config.from_env()
    output_dir = cfg.output_path
    experiment_dir = output_dir / "experiments" / label
    experiment_dir.mkdir(parents=True, exist_ok=True)
    default_mc_path = output_dir / f"monte_carlo_{cfg.country_iso3}.pkl"

    arm_results = {}
    for arm in arms:
        arm_dir = experiment_dir / arm.name
        arm_dir.mkdir(parents=True, exist_ok=True)
        result = main(
            seeds=seed_list,
            t_max=t_max,
            government_consumption_setter=arm.setter,
            assume_zero_noise=True,
            government_sectoral_weights=arm.sectoral_weights,
            government_consumption_consistency=arm.consistency,
            n_jobs=n_jobs,
            backend=backend,
            verbose=verbose,
            batch_size=batch_size,
            output_file=arm_dir / f"monte_carlo_{cfg.country_iso3}.pkl",
            save_h5_dir=arm_dir,
        )
        arm_mc_path = arm_dir / f"monte_carlo_{cfg.country_iso3}.pkl"
        arm_results[arm.name] = {**result, "arm_output_path": arm_mc_path}

    if arms:
        shutil.copyfile(arm_results[arms[-1].name]["arm_output_path"], default_mc_path)

    summary, thresholds, arm_aggregates = _write_government_bridge_analysis(
        experiment_dir=experiment_dir,
        arms=arms,
        seeds=seed_list,
        country_code=cfg.country_iso3,
    )

    print(f"Government bridge MC complete. Default output convention preserved at: {default_mc_path}")
    print(f"Experiment outputs saved under: {experiment_dir}")
    print(f"Analysis saved under: {experiment_dir / 'analysis'}")

    return {
        "cfg": cfg,
        "seeds": seed_list,
        "arms": arms,
        "experiment_dir": experiment_dir,
        "default_mc_path": default_mc_path,
        "arm_results": arm_results,
        "summary": summary,
        "thresholds": thresholds,
        "arm_aggregates": arm_aggregates,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run seeded Monte Carlo simulations for the macro model.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Unique integer simulation seeds. Default: "
            f"{' '.join(str(seed) for seed in DEFAULT_SEEDS)} for the standard MC runner; "
            f"{' '.join(str(seed) for seed in DEFAULT_GOVERNMENT_BRIDGE_SEEDS)} for government bridge MC."
        ),
    )
    parser.add_argument(
        "--t-max", type=int, default=DEFAULT_T_MAX, help=f"Simulation horizon. Default: {DEFAULT_T_MAX}."
    )
    parser.add_argument(
        "--government-consumption-setter",
        choices=GOVERNMENT_CONSUMPTION_SETTER_CHOICES,
        default=DEFAULT_GOVERNMENT_CONSUMPTION_SETTER,
        help=(f"Government-consumption setter to use. Default: {DEFAULT_GOVERNMENT_CONSUMPTION_SETTER}."),
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
    parser.add_argument(
        "--save-h5-dir",
        default=None,
        help="Optional directory for per-seed HDF5 outputs.",
    )
    parser.add_argument(
        "--tfp-target-intensity-multiplier",
        type=float,
        default=1.0,
        help="Scale sector_innovation_intensity targets in-memory for revised TFP calibration runs.",
    )
    parser.add_argument(
        "--government-bridge-experiment",
        action="store_true",
        help="Run the four-arm government-consumption bridge Monte Carlo experiment.",
    )
    parser.add_argument(
        "--government-consistency-initial-weights-experiment",
        action="store_true",
        help="Run the expanded consistency-by-initial-weights government MC experiment.",
    )
    parser.add_argument(
        "--label",
        default=DEFAULT_GOVERNMENT_BRIDGE_LABEL,
        help=(f"Experiment label for government MC outputs. Default: {DEFAULT_GOVERNMENT_BRIDGE_LABEL}."),
    )
    parser.add_argument(
        "--bridge-arms",
        nargs="+",
        choices=tuple(GOVERNMENT_BRIDGE_ARM_BY_NAME),
        default=list(GOVERNMENT_BRIDGE_ARM_BY_NAME),
        help="Subset of government bridge arms to run. Default: all arms.",
    )
    parser.add_argument(
        "--initial-weight-arms",
        nargs="+",
        choices=tuple(GOVERNMENT_CONSISTENCY_INITIAL_WEIGHTS_ARM_BY_NAME),
        default=list(GOVERNMENT_CONSISTENCY_INITIAL_WEIGHTS_ARM_BY_NAME),
        help="Subset of expanded initial-weight arms to run. Default: all expanded arms.",
    )
    args = parser.parse_args()
    if args.seeds is not None:
        args.seeds = _validate_unique_seeds(args.seeds)
    return args


if __name__ == "__main__":
    args = _parse_args()
    if args.government_consistency_initial_weights_experiment:
        label = (
            DEFAULT_GOVERNMENT_CONSISTENCY_INITIAL_WEIGHTS_LABEL
            if args.label == DEFAULT_GOVERNMENT_BRIDGE_LABEL
            else args.label
        )
        run_government_bridge_experiment(
            seeds=args.seeds,
            t_max=args.t_max,
            label=label,
            arm_names=args.initial_weight_arms,
            arm_lookup=GOVERNMENT_CONSISTENCY_INITIAL_WEIGHTS_ARM_BY_NAME,
            default_seeds=DEFAULT_GOVERNMENT_CONSISTENCY_INITIAL_WEIGHTS_SEEDS,
            n_jobs=args.n_jobs,
            backend=args.backend,
            verbose=args.verbose,
            batch_size=args.batch_size,
        )
    elif args.government_bridge_experiment:
        run_government_bridge_experiment(
            seeds=args.seeds,
            t_max=args.t_max,
            label=args.label,
            arm_names=args.bridge_arms,
            arm_lookup=GOVERNMENT_BRIDGE_ARM_BY_NAME,
            default_seeds=DEFAULT_GOVERNMENT_BRIDGE_SEEDS,
            n_jobs=args.n_jobs,
            backend=args.backend,
            verbose=args.verbose,
            batch_size=args.batch_size,
        )
    else:
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
            save_h5_dir=args.save_h5_dir,
            tfp_target_intensity_multiplier=args.tfp_target_intensity_multiplier,
        )
