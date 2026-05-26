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


def _winsorize_series(series: pd.Series, winsor_pct: float) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return series
    lower = clean.quantile(winsor_pct)
    upper = clean.quantile(1.0 - winsor_pct)
    return series.clip(lower=lower, upper=upper)


def _load_classification_sector_map(path: Path, sector_col: str) -> pd.DataFrame:
    classifications = pd.read_csv(path, usecols=["bvd_id_number", sector_col], low_memory=False)
    classifications = classifications.dropna(subset=["bvd_id_number"])
    # Keep one sector mapping per entity to avoid one-to-many row explosions.
    mapped = (
        classifications.dropna(subset=[sector_col])
        .drop_duplicates(subset=["bvd_id_number"], keep="first")
        .loc[:, ["bvd_id_number", sector_col]]
    )
    return mapped


def _prepare_base_frame(
    financials_path: Path,
    sector_map: pd.DataFrame,
    unconsolidated_codes: set[str],
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

    for col in [REVENUE_COL, "material_costs", "costs_of_employees", "depreciation_amortization", "interest_paid"]:
        financials[col] = pd.to_numeric(financials[col], errors="coerce")

    financials["year"] = pd.to_datetime(financials["closing_date"], errors="coerce").dt.year
    merged = financials.merge(sector_map, on="bvd_id_number", how="left")

    merged["consolidation_code"] = merged["consolidation_code"].astype("string").str.strip()
    merged = merged[merged["consolidation_code"].isin(unconsolidated_codes)]
    merged = merged[merged[REVENUE_COL] > 0.0]
    merged = merged[merged["material_costs"] >= 0.0]
    merged = merged[merged["costs_of_employees"] >= 0.0]
    merged = merged.dropna(subset=["year", SECTOR_COL])

    merged["UC_OP"] = merged["material_costs"] + merged["costs_of_employees"]
    merged["UC_FC"] = merged["UC_OP"] + merged["depreciation_amortization"]
    merged["UC_ALL"] = merged["UC_FC"] + merged["interest_paid"]

    merged["MU_OP"] = np.where(merged["UC_OP"] > 0.0, merged[REVENUE_COL] / merged["UC_OP"], np.nan)
    merged["MU_FC"] = np.where(merged["UC_FC"] > 0.0, merged[REVENUE_COL] / merged["UC_FC"], np.nan)
    merged["MU_ALL"] = np.where(merged["UC_ALL"] > 0.0, merged[REVENUE_COL] / merged["UC_ALL"], np.nan)
    return merged


def _aggregate_markups(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (year, sector), group in df.groupby(["year", SECTOR_COL], sort=True):
        result: dict[str, float | int | str] = {
            "year": int(year),
            SECTOR_COL: str(sector),
            "n_entities": int(len(group)),
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
            else:
                result[f"{metric.lower()}_weighted_mean"] = float(np.average(vals, weights=wts))
                result[f"{metric.lower()}_weighted_median"] = _weighted_median(vals, wts)
        rows.append(result)
    return pd.DataFrame(rows)


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
        "--nrows",
        type=int,
        default=None,
        help="Optional row cap for financials file (useful for smoke tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not (0.0 <= args.winsor_pct < 0.5):
        raise ValueError("--winsor-pct must be in [0.0, 0.5).")

    financials_path = _resolve_path(args.financials)
    classifications_path = _resolve_path(args.classifications)
    output_path = _resolve_path(args.out)
    unconsolidated_codes = {str(code).strip() for code in args.unconsolidated_codes}

    sector_map = _load_classification_sector_map(classifications_path, SECTOR_COL)
    frame = _prepare_base_frame(
        financials_path=financials_path,
        sector_map=sector_map,
        unconsolidated_codes=unconsolidated_codes,
        nrows=args.nrows,
    )

    for metric in ["MU_OP", "MU_FC", "MU_ALL"]:
        frame[metric] = _winsorize_series(frame[metric], args.winsor_pct)

    aggregated = _aggregate_markups(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(output_path, index=False)

    print(f"Wrote {len(aggregated)} rows to {output_path}")


if __name__ == "__main__":
    main()
