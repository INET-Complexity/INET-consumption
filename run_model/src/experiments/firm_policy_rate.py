"""Verification experiment for firm credit, investment, and TFP transmission."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

RUN_MODEL_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = RUN_MODEL_DIR.parent
for path in (str(REPO_ROOT), str(RUN_MODEL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.diagnostics.firm_policy_rate import (  # noqa: E402
    aggregate_firm_policy_rate_irf,
    build_firm_policy_rate_irf_panel,
    classify_tfp_dose_response,
    macro_policy_rate_variables,
    summarize_firm_policy_rate_irf,
)
from src.experiments.irf import run_irf_experiment  # noqa: E402
from src.irf_analysis import build_irf_panel, summarize_irf_panel, write_irf_plots  # noqa: E402

from macromodel.utils.prehooks.irf_shocks import ShockSpec  # noqa: E402

DEFAULT_SEEDS = tuple(range(12, 22))
DEFAULT_T_MAX = 80
DEFAULT_HORIZON_PERIODS = 50
DEFAULT_SHOCK_DURATION = 30
DEFAULT_BOOTSTRAP_DRAWS = 500
DEFAULT_OUTPUT_DIR = Path("/private/tmp/inet-firm-policy-rate-verification")


def _shock_specs() -> tuple[ShockSpec, ...]:
    """Return the policy-rate dose-response shocks used by the verification."""
    return tuple(
        ShockSpec(
            name=f"policy_rate_{basis_points}bp_30q",
            kind="policy_rate",
            period=0,
            magnitude=basis_points / 10_000.0,
            duration=DEFAULT_SHOCK_DURATION,
            mode="additive",
        )
        for basis_points in (50, 100, 200)
    )


def _safe_name(value: object) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in str(value))


def _write_firm_policy_rate_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    """Write focused HTML plots for credit buckets and the TFP channel."""
    if summary.empty:
        return
    import plotly.graph_objects as go

    variables = {
        "tfp_investment_effective_cost_rate",
        "planned_tfp_investment",
        "executed_tfp_investment",
        "target_long_term_credit",
        "ordinary_target_short_term_credit",
        "target_debt_rollover_credit",
        "target_overdraft_refinance_credit",
    }
    selected = summary[
        summary["variable"].isin(variables)
        & (summary["liquidity_group"] == "all")
        & (summary["debt_stressed"] == "all")
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    for (shock_name, variable), group in selected.groupby(["shock_name", "variable"], sort=False):
        group = group.sort_values("horizon")
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=group["horizon"],
                y=group["delta_mean"],
                mode="lines+markers",
                name="paired mean",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=pd.concat([group["horizon"], group["horizon"].iloc[::-1]]),
                y=pd.concat([group["bootstrap_ci_high"], group["bootstrap_ci_low"].iloc[::-1]]),
                fill="toself",
                line={"color": "rgba(0,0,0,0)"},
                name="95% bootstrap interval",
            )
        )
        figure.update_layout(
            title=f"{shock_name}: {variable}",
            template="plotly_white",
            xaxis_title="horizon",
            yaxis_title="paired shock minus baseline",
        )
        figure.write_html(output_dir / f"{_safe_name(shock_name)}_{_safe_name(variable)}.html")


def _write_verification_report(
    output_dir: Path,
    *,
    seeds: list[int],
    t_max: int,
    horizon_periods: int,
    shock_duration: int,
    bootstrap_draws: int,
    firm_summary: pd.DataFrame,
    dose_response: pd.DataFrame,
) -> Path:
    """Write a reproducibility manifest and a compact, data-backed report."""
    report_path = output_dir / "verification_report.md"
    all_tfp = dose_response[
        (dose_response["dscr_enabled"] == True)  # noqa: E712
        & (dose_response["liquidity_group"] == "all")
        & (dose_response["debt_stressed"] == "all")
    ]
    monotone_share = float(all_tfp["monotone_nonincreasing"].mean()) if not all_tfp.empty else float("nan")
    nonpositive_share = float(all_tfp["all_nonpositive"].mean()) if not all_tfp.empty else float("nan")

    lines = [
        "# Firm policy-rate transmission verification",
        "",
        "This report is generated from paired baseline/shock simulations. It is diagnostic and does not modify firm behaviour.",
        "",
        "## Design",
        "",
        f"- Seeds: `{','.join(str(seed) for seed in seeds)}`",
        f"- Simulation length: `{t_max}` periods",
        f"- IRF horizon: `{horizon_periods}` periods",
        f"- Policy shock duration: `{shock_duration}` periods",
        f"- Bootstrap draws: `{bootstrap_draws}`",
        "- DSCR arms: current configuration and an explicit DSCR-off control",
        "",
        "## Automated checks",
        "",
        f"- TFP dose-response rows with non-increasing mean response: `{monotone_share:.3f}`",
        f"- TFP dose-response rows with non-positive mean response: `{nonpositive_share:.3f}`",
        f"- Firm summary rows: `{len(firm_summary)}`",
        "",
        "The detailed CSV outputs should be used for the final economic interpretation, including liquidity groups, debt-stress buckets, DSCR binding reasons, and post-shock persistence.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_firm_policy_rate_verification(
    *,
    seeds: list[int] | tuple[int, ...] = DEFAULT_SEEDS,
    t_max: int = DEFAULT_T_MAX,
    horizon_periods: int = DEFAULT_HORIZON_PERIODS,
    shock_duration: int = DEFAULT_SHOCK_DURATION,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    country_iso3: str = "FRA",
    n_jobs: int = 1,
    backend: str = "loky",
    batch_size: int = 1,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    random_state: int = 20260910,
) -> dict[str, Path]:
    """Run DSCR controls, rate dose response, and firm-level post-processing."""
    seed_list = [int(seed) for seed in seeds]
    if not seed_list or len(set(seed_list)) != len(seed_list):
        raise ValueError("seeds must be non-empty and unique")
    if shock_duration < 30:
        raise ValueError("shock_duration must be at least 30 periods")
    if t_max < shock_duration + 1:
        raise ValueError("t_max must leave at least one saved period after the policy shock")
    if horizon_periods <= 0 or shock_duration + 1 + horizon_periods > t_max + 1:
        raise ValueError("horizon_periods does not fit within t_max")
    if bootstrap_draws < 1:
        raise ValueError("bootstrap_draws must be at least 1")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    on_dir = root / "dscr-on"
    off_dir = root / "dscr-off"
    dose_specs = _shock_specs()
    if shock_duration != DEFAULT_SHOCK_DURATION:
        dose_specs = tuple(
            ShockSpec(
                name=f"policy_rate_{int(round(spec.magnitude * 10_000))}bp_{shock_duration}q",
                kind=spec.kind,
                period=spec.period,
                magnitude=spec.magnitude,
                duration=shock_duration,
                mode=spec.mode,
            )
            for spec in dose_specs
        )

    on_paths = run_irf_experiment(
        seeds=seed_list,
        t_max=t_max,
        shock_specs=dose_specs,
        horizon_periods=horizon_periods,
        output_dir=on_dir,
        country_iso3=country_iso3,
        n_jobs=n_jobs,
        backend=backend,
        batch_size=batch_size,
        firm_loan_dscr=True,
    )
    off_specs = (dose_specs[1],)
    off_paths = run_irf_experiment(
        seeds=seed_list,
        t_max=t_max,
        shock_specs=off_specs,
        horizon_periods=horizon_periods,
        output_dir=off_dir,
        country_iso3=country_iso3,
        n_jobs=n_jobs,
        backend=backend,
        batch_size=batch_size,
        firm_loan_dscr=False,
    )

    run_sets = ((on_paths, True), (off_paths, False))
    firm_aggregate_panels: list[pd.DataFrame] = []
    macro_panels: list[pd.DataFrame] = []
    for paths, dscr_enabled in run_sets:
        run_index = pd.read_csv(paths["analysis_dir"] / "irf_run_index.csv")
        for row in run_index.to_dict(orient="records"):
            shock_period = int(row["shock_period"]) + 1
            firm_panel = build_firm_policy_rate_irf_panel(
                row["baseline_h5"],
                row["shock_h5"],
                seed=int(row["seed"]),
                shock_name=str(row["shock_name"]),
                shock_kind=str(row["shock_kind"]),
                shock_period=shock_period,
                shock_magnitude=float(row["shock_magnitude"]),
                shock_duration=int(row["shock_duration"]),
                shock_mode=str(row["shock_mode"]),
                horizon_periods=horizon_periods,
                country_code=country_iso3,
                dscr_enabled=dscr_enabled,
            )
            firm_aggregate_panels.append(aggregate_firm_policy_rate_irf(firm_panel))
            macro_panels.append(
                build_irf_panel(
                    baseline_h5=row["baseline_h5"],
                    shock_h5=row["shock_h5"],
                    seed=int(row["seed"]),
                    shock_name=str(row["shock_name"]),
                    shock_kind=str(row["shock_kind"]),
                    shock_period=shock_period,
                    shock_magnitude=float(row["shock_magnitude"]),
                    shock_duration=int(row["shock_duration"]),
                    shock_mode=str(row["shock_mode"]),
                    horizon_periods=horizon_periods,
                    country_code=country_iso3,
                    variables=macro_policy_rate_variables(),
                    strict=False,
                )
            )

    firm_aggregate = pd.concat(firm_aggregate_panels, ignore_index=True)
    firm_summary = summarize_firm_policy_rate_irf(
        firm_aggregate,
        n_bootstrap=bootstrap_draws,
        random_state=random_state,
    )
    dose_response = classify_tfp_dose_response(firm_summary)
    macro_panel = pd.concat(macro_panels, ignore_index=True)
    macro_summary = summarize_irf_panel(macro_panel)

    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    firm_aggregate.to_csv(analysis_dir / "firm_group_period_irf.csv", index=False)
    firm_summary.to_csv(analysis_dir / "firm_irf_summary.csv", index=False)
    dose_response.to_csv(analysis_dir / "tfp_dose_response.csv", index=False)
    macro_panel.to_csv(analysis_dir / "macro_irf_panel.csv", index=False)
    macro_summary.to_csv(analysis_dir / "macro_irf_summary.csv", index=False)
    write_irf_plots(macro_summary, analysis_dir / "macro_plots", value_column="delta")
    write_irf_plots(macro_summary, analysis_dir / "macro_plots_pct", value_column="pct_delta")
    _write_firm_policy_rate_plots(firm_summary, analysis_dir / "firm_plots")

    metadata = {
        "country_iso3": country_iso3,
        "seeds": seed_list,
        "t_max": t_max,
        "horizon_periods": horizon_periods,
        "shock_duration": shock_duration,
        "shock_magnitudes": [float(spec.magnitude) for spec in dose_specs],
        "bootstrap_draws": bootstrap_draws,
        "random_state": random_state,
        "dscr_on_run": str(on_dir),
        "dscr_off_run": str(off_dir),
    }
    (analysis_dir / "verification_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report = _write_verification_report(
        analysis_dir,
        seeds=seed_list,
        t_max=t_max,
        horizon_periods=horizon_periods,
        shock_duration=shock_duration,
        bootstrap_draws=bootstrap_draws,
        firm_summary=firm_summary,
        dose_response=dose_response,
    )
    return {
        "output_dir": root,
        "analysis_dir": analysis_dir,
        "firm_panel": analysis_dir / "firm_group_period_irf.csv",
        "firm_summary": analysis_dir / "firm_irf_summary.csv",
        "macro_summary": analysis_dir / "macro_irf_summary.csv",
        "dose_response": analysis_dir / "tfp_dose_response.csv",
        "report": report,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify firm policy-rate borrowing and investment transmission.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--t-max", type=int, default=DEFAULT_T_MAX)
    parser.add_argument("--horizon-periods", type=int, default=DEFAULT_HORIZON_PERIODS)
    parser.add_argument("--shock-duration", type=int, default=DEFAULT_SHOCK_DURATION)
    parser.add_argument("--country", default="FRA")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--backend", default="loky")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--random-state", type=int, default=20260910)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = run_firm_policy_rate_verification(
        seeds=args.seeds,
        t_max=args.t_max,
        horizon_periods=args.horizon_periods,
        shock_duration=args.shock_duration,
        output_dir=args.output_dir,
        country_iso3=args.country,
        n_jobs=args.n_jobs,
        backend=args.backend,
        batch_size=args.batch_size,
        bootstrap_draws=args.bootstrap_draws,
        random_state=args.random_state,
    )
    print({name: str(path) for name, path in outputs.items()})


if __name__ == "__main__":
    main()
