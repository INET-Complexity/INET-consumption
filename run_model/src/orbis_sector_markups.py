from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RUN_MODEL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ORBIS_DIR = RUN_MODEL_DIR / "data" / "raw_data" / "orbis"

REVENUE_COL = "operating_revenue_turnover_"
SECTOR_COL = "nace_rev_2_main_section"


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.parts and candidate.parts[0] == "run_model":
        return Path.cwd() / candidate
    if candidate.exists():
        return candidate
    return RUN_MODEL_DIR / candidate


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return float("nan")
    vals = values[mask]
    wts = weights[mask]
    order = np.argsort(vals)
    vals = vals[order]
    wts = wts[order]
    cumsum = np.cumsum(wts)
    cutoff = 0.5 * wts.sum()
    idx = int(np.searchsorted(cumsum, cutoff, side="left"))
    return float(vals[min(idx, len(vals) - 1)])


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0, 1].")
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return float("nan")
    vals = values[mask]
    wts = weights[mask]
    order = np.argsort(vals)
    vals = vals[order]
    wts = wts[order]
    cdf = np.cumsum(wts) / wts.sum()
    idx = int(np.searchsorted(cdf, q, side="left"))
    return float(vals[min(idx, len(vals) - 1)])


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return float("nan")
    vals = values[mask]
    wts = weights[mask]
    mean = np.average(vals, weights=wts)
    var = np.average((vals - mean) ** 2, weights=wts)
    return float(np.sqrt(max(var, 0.0)))


def _winsorize_series(series: pd.Series, winsor_pct: float) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return series
    lower = clean.quantile(winsor_pct)
    upper = clean.quantile(1.0 - winsor_pct)
    return series.clip(lower=lower, upper=upper)


def _load_classification_sector_map(path: Path, sector_col: str) -> tuple[pd.DataFrame, int]:
    classifications = pd.read_csv(path, usecols=["bvd_id_number", sector_col], low_memory=False)
    classifications = classifications.dropna(subset=["bvd_id_number"])
    classified = classifications.dropna(subset=[sector_col]).loc[:, ["bvd_id_number", sector_col]]

    # Resolve conflicting sector mappings deterministically using the modal sector
    # per entity (tie-break by lexicographic order).
    conflicts = classified.groupby("bvd_id_number")[sector_col].nunique()
    conflict_count = int((conflicts > 1).sum())
    mapped = (
        classified.groupby("bvd_id_number")[sector_col]
        .agg(lambda x: x.value_counts().sort_index().idxmax())
        .reset_index()
    )
    return mapped, conflict_count


def _prepare_base_frame(
    financials_path: Path,
    sector_map: pd.DataFrame,
    unconsolidated_codes: set[str],
    country_prefix: str,
    missing_cost_policy: str,
    nrows: int | None,
) -> pd.DataFrame:
    usecols = [
        "bvd_id_number",
        "closing_date",
        "consolidation_code",
        REVENUE_COL,
        "material_costs",
        "costs_of_employees",
        "depreciation_amortization",
        "interest_paid",
    ]
    financials = pd.read_csv(financials_path, usecols=usecols, low_memory=False, nrows=nrows)
    financials["bvd_id_number"] = financials["bvd_id_number"].astype("string").str.strip()

    for col in [REVENUE_COL, "material_costs", "costs_of_employees", "depreciation_amortization", "interest_paid"]:
        financials[col] = pd.to_numeric(financials[col], errors="coerce")

    financials["year"] = pd.to_datetime(financials["closing_date"], errors="coerce").dt.year
    merged = financials.merge(sector_map, on="bvd_id_number", how="left")

    merged["consolidation_code"] = merged["consolidation_code"].astype("string").str.strip()
    merged = merged[merged["bvd_id_number"].str.startswith(country_prefix, na=False)]
    merged = merged[merged["consolidation_code"].isin(unconsolidated_codes)]
    merged = merged[merged[REVENUE_COL] > 0.0]
    merged = merged[merged["material_costs"] >= 0.0]
    merged = merged[merged["costs_of_employees"] >= 0.0]
    merged = merged.dropna(subset=["year", SECTOR_COL])

    if missing_cost_policy == "zero":
        merged["depreciation_amortization"] = merged["depreciation_amortization"].fillna(0.0)
        merged["interest_paid"] = merged["interest_paid"].fillna(0.0)
    elif missing_cost_policy != "drop":
        raise ValueError("--missing-cost-policy must be one of: drop, zero")

    merged["UC_OP"] = merged["material_costs"] + merged["costs_of_employees"]
    merged["UC_FC"] = merged["UC_OP"] + merged["depreciation_amortization"]
    merged["UC_ALL"] = merged["UC_FC"] + merged["interest_paid"]

    merged["MU_OP"] = np.where(merged["UC_OP"] > 0.0, merged[REVENUE_COL] / merged["UC_OP"], np.nan)
    merged["MU_FC"] = np.where(merged["UC_FC"] > 0.0, merged[REVENUE_COL] / merged["UC_FC"], np.nan)
    merged["MU_ALL"] = np.where(merged["UC_ALL"] > 0.0, merged[REVENUE_COL] / merged["UC_ALL"], np.nan)
    return merged


def _aggregate_markups(
    df: pd.DataFrame,
    median_interval_half_width: float,
    mean_interval_std_multiplier: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (year, sector), group in df.groupby(["year", SECTOR_COL], sort=True):
        result: dict[str, float | int | str] = {
            "year": int(year),
            SECTOR_COL: str(sector),
            "n_rows": int(len(group)),
            "n_entities": int(group["bvd_id_number"].nunique()),
            "sum_weights": float(group[REVENUE_COL].sum(skipna=True)),
        }
        for metric in ["MU_OP", "MU_FC", "MU_ALL"]:
            metric_mask = group[metric].notna() & group[REVENUE_COL].notna() & (group[REVENUE_COL] > 0.0)
            vals = group.loc[metric_mask, metric].to_numpy(dtype=float)
            wts = group.loc[metric_mask, REVENUE_COL].to_numpy(dtype=float)
            result[f"{metric.lower()}_n"] = int(vals.size)
            if vals.size == 0:
                result[f"{metric.lower()}_weighted_mean"] = float("nan")
                result[f"{metric.lower()}_weighted_median"] = float("nan")
                result[f"{metric.lower()}_median_interval_low"] = float("nan")
                result[f"{metric.lower()}_median_interval_high"] = float("nan")
                result[f"{metric.lower()}_mean_interval_low"] = float("nan")
                result[f"{metric.lower()}_mean_interval_high"] = float("nan")
            else:
                mean_val = float(np.average(vals, weights=wts))
                median_val = _weighted_median(vals, wts)
                std_val = _weighted_std(vals, wts)

                q_low = max(0.0, 0.5 - median_interval_half_width)
                q_high = min(1.0, 0.5 + median_interval_half_width)
                median_low = _weighted_quantile(vals, wts, q_low)
                median_high = _weighted_quantile(vals, wts, q_high)

                result[f"{metric.lower()}_weighted_mean"] = mean_val
                result[f"{metric.lower()}_weighted_median"] = median_val
                result[f"{metric.lower()}_median_interval_low"] = median_low
                result[f"{metric.lower()}_median_interval_high"] = median_high
                result[f"{metric.lower()}_mean_interval_low"] = mean_val - mean_interval_std_multiplier * std_val
                result[f"{metric.lower()}_mean_interval_high"] = mean_val + mean_interval_std_multiplier * std_val
        rows.append(result)
    return pd.DataFrame(rows)


def _winsorize_by_group(df: pd.DataFrame, metrics: list[str], winsor_pct: float) -> pd.DataFrame:
    if winsor_pct == 0.0:
        return df
    out = df.copy()
    group_cols = ["year", SECTOR_COL]
    for metric in metrics:
        out[metric] = out.groupby(group_cols, group_keys=False)[metric].transform(
            lambda s: _winsorize_series(s, winsor_pct)
        )
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute ORBIS markups and aggregate by NACE Rev.2 main section.")
    parser.add_argument(
        "--financials",
        type=Path,
        default=DEFAULT_ORBIS_DIR / "orbis_academics_quarterly_industry_global_financials_and_ratios_2014.csv",
        help="Path to ORBIS financials CSV.",
    )
    parser.add_argument(
        "--classifications",
        type=Path,
        default=DEFAULT_ORBIS_DIR / "orbis_academics_quarterly_industry_classifications.csv",
        help="Path to ORBIS classifications CSV.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_ORBIS_DIR / "orbis_markups_by_nace_rev_2_main_section.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--winsor-pct",
        type=float,
        default=0.01,
        help="Two-sided winsorization quantile in [0.0, 0.5). Example 0.01 means 1%% and 99%% caps.",
    )
    parser.add_argument(
        "--unconsolidated-codes",
        nargs="+",
        default=["U1"],
        help="Consolidation codes treated as unconsolidated accounts.",
    )
    parser.add_argument(
        "--country-prefix",
        type=str,
        default="FR",
        help="Keep only entities where bvd_id_number starts with this prefix (e.g., FR).",
    )
    parser.add_argument(
        "--missing-cost-policy",
        choices=["drop", "zero"],
        default="drop",
        help="How to treat missing depreciation/interest in MU_FC and MU_ALL.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional row cap for financials file (useful for smoke tests).",
    )
    parser.add_argument(
        "--median-interval-half-width",
        type=float,
        default=0.25,
        help="Half-width around 50th percentile for median-centered interval. 0.25 -> [p25, p75].",
    )
    parser.add_argument(
        "--mean-interval-std-multiplier",
        type=float,
        default=1.0,
        help="Std-dev multiplier for mean-centered interval. 1.0 -> mean +/- 1 weighted std.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not (0.0 <= args.winsor_pct < 0.5):
        raise ValueError("--winsor-pct must be in [0.0, 0.5).")
    if not (0.0 <= args.median_interval_half_width <= 0.5):
        raise ValueError("--median-interval-half-width must be in [0.0, 0.5].")
    if args.mean_interval_std_multiplier < 0.0:
        raise ValueError("--mean-interval-std-multiplier must be >= 0.")

    financials_path = _resolve_path(args.financials)
    classifications_path = _resolve_path(args.classifications)
    output_path = _resolve_path(args.out)
    unconsolidated_codes = {str(code).strip() for code in args.unconsolidated_codes}
    country_prefix = args.country_prefix.strip().upper()

    sector_map, conflict_count = _load_classification_sector_map(classifications_path, SECTOR_COL)
    frame = _prepare_base_frame(
        financials_path=financials_path,
        sector_map=sector_map,
        unconsolidated_codes=unconsolidated_codes,
        country_prefix=country_prefix,
        missing_cost_policy=args.missing_cost_policy,
        nrows=args.nrows,
    )

    metrics = ["MU_OP", "MU_FC", "MU_ALL"]
    frame = _winsorize_by_group(frame, metrics=metrics, winsor_pct=args.winsor_pct)

    aggregated = _aggregate_markups(
        frame,
        median_interval_half_width=args.median_interval_half_width,
        mean_interval_std_multiplier=args.mean_interval_std_multiplier,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(output_path, index=False)

    print(
        f"Wrote {len(aggregated)} rows to {output_path} "
        f"(country_prefix={country_prefix}, sector_conflicts_resolved={conflict_count}, "
        f"missing_cost_policy={args.missing_cost_policy})"
    )


if __name__ == "__main__":
    main()
