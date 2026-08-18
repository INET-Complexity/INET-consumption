from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _read(handle: h5py.File, path: str) -> np.ndarray:
    return np.asarray(handle[path], dtype=float)


def _aggregate_flat_sector_panel(values: np.ndarray, columns: np.ndarray, n_sectors: int) -> np.ndarray:
    """Aggregate a flattened entity-sector panel back to time x sector."""
    out = np.zeros((values.shape[0], n_sectors), dtype=float)
    for idx, column in enumerate(columns):
        sector = int(column[1])
        out[:, sector] += values[:, idx]
    return out


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray, fill: float = np.nan) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full(np.broadcast_shapes(numerator.shape, denominator.shape), fill, dtype=float),
        where=denominator != 0.0,
    )


def _first_below(values: np.ndarray, threshold: float) -> float:
    hits = np.where(np.isfinite(values) & (values < threshold))[0]
    return float(hits[0]) if len(hits) else np.nan


def _first_zero(values: np.ndarray) -> float:
    hits = np.where(values == 0.0)[0]
    return float(hits[0]) if len(hits) else np.nan


def _row_nanargmax(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[0], np.nan)
    valid = np.isfinite(values).any(axis=1)
    result[valid] = np.nanargmax(values[valid], axis=1)
    return result


def _row_nanmax(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[0], np.nan)
    valid = np.isfinite(values).any(axis=1)
    result[valid] = np.nanmax(values[valid], axis=1)
    return result


def _row_nanmin(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[0], np.nan)
    valid = np.isfinite(values).any(axis=1)
    result[valid] = np.nanmin(values[valid], axis=1)
    return result


def build_government_bridge_diagnostics(
    h5_path: Path,
    country: str = "FRA",
    top_k: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    with h5py.File(h5_path, "r") as handle:
        base = f"/{country}"
        desired_sector = _read(handle, f"{base}/government_entities/desired_consumption_in_lcu")
        realised_sector = _read(handle, f"{base}/government_entities/consumption_in_lcu")
        prices = _read(handle, f"{base}/economy/good_prices")
        initial_prices = _read(handle, f"{base}/economy/initial_price")[0]
        production = _read(handle, f"{base}/firms/production")
        inventory = _read(handle, f"{base}/firms/inventory")

        n_sectors = desired_sector.shape[1]
        if f"{base}/government_entities/real_amount_bought" in handle:
            real_bought = _read(handle, f"{base}/government_entities/real_amount_bought")
            real_bought_columns = np.asarray(handle[f"{base}/government_entities/real_amount_bought_columns"])
            realised_real_bought_sector = _aggregate_flat_sector_panel(real_bought, real_bought_columns, n_sectors)
        else:
            realised_real_bought_sector = np.full_like(desired_sector, np.nan)

    desired_total = desired_sector.sum(axis=1)
    realised_total = realised_sector.sum(axis=1)
    realised_desired_ratio = _safe_divide(realised_total, desired_total)

    price_ratio = _safe_divide(prices, initial_prices)
    government_real_demand = _safe_divide(desired_sector, prices, fill=0.0)
    sector_supply = production + inventory
    supply_cover = _safe_divide(sector_supply, government_real_demand)

    desired_share = _safe_divide(desired_sector, desired_total[:, None])
    realised_share = _safe_divide(realised_sector, realised_total[:, None])

    timeseries = pd.DataFrame(
        {
            "time": np.arange(len(desired_total)),
            "desired_government_consumption": desired_total,
            "realised_government_consumption": realised_total,
            "realised_desired_ratio": realised_desired_ratio,
            "price_ratio_min": np.nanmin(price_ratio, axis=1),
            "price_ratio_median": np.nanmedian(price_ratio, axis=1),
            "price_ratio_max": np.nanmax(price_ratio, axis=1),
            "dominant_desired_sector": _row_nanargmax(desired_share),
            "dominant_desired_share": _row_nanmax(desired_share),
            "dominant_realised_sector": _row_nanargmax(realised_share),
            "dominant_realised_share": _row_nanmax(realised_share),
            "min_supply_cover_for_government_real_demand": _row_nanmin(supply_cover),
        }
    )

    top_rows = []
    for t in range(desired_sector.shape[0]):
        order = np.argsort(np.nan_to_num(desired_share[t], nan=-np.inf))[::-1][:top_k]
        for rank, sector in enumerate(order, start=1):
            top_rows.append(
                {
                    "time": t,
                    "rank": rank,
                    "sector": int(sector),
                    "desired_share": desired_share[t, sector],
                    "realised_share": realised_share[t, sector],
                    "desired_government_consumption": desired_sector[t, sector],
                    "realised_government_consumption": realised_sector[t, sector],
                    "price_ratio": price_ratio[t, sector],
                    "government_real_demand": government_real_demand[t, sector],
                    "sector_supply": sector_supply[t, sector],
                    "supply_cover_for_government_real_demand": supply_cover[t, sector],
                    "realised_real_amount_bought": realised_real_bought_sector[t, sector],
                }
            )
    sector_top = pd.DataFrame(top_rows)

    thresholds = {
        "first_ratio_below_0_9": _first_below(realised_desired_ratio, 0.9),
        "first_ratio_below_0_5": _first_below(realised_desired_ratio, 0.5),
        "first_ratio_below_0_25": _first_below(realised_desired_ratio, 0.25),
        "first_desired_below_1bn": _first_below(desired_total, 1_000_000_000.0),
        "first_desired_below_1m": _first_below(desired_total, 1_000_000.0),
        "first_desired_zero": _first_zero(desired_total),
    }
    return timeseries, sector_top, thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build desired-vs-realised government bridge diagnostics.")
    parser.add_argument("h5_path", type=Path)
    parser.add_argument("--country", default="FRA")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeseries-out", type=Path, required=True)
    parser.add_argument("--sector-top-out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timeseries, sector_top, thresholds = build_government_bridge_diagnostics(
        h5_path=args.h5_path,
        country=args.country,
        top_k=args.top_k,
    )
    args.timeseries_out.parent.mkdir(parents=True, exist_ok=True)
    args.sector_top_out.parent.mkdir(parents=True, exist_ok=True)
    timeseries.to_csv(args.timeseries_out, index=False)
    sector_top.to_csv(args.sector_top_out, index=False)

    print(f"wrote {args.timeseries_out}")
    print(f"wrote {args.sector_top_out}")
    for key, value in thresholds.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
