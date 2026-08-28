"""Multi-seed comparison of the wage arms (t_max=50, 10 seeds each)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/Users/andone/Documents/python_projects/inet-wage-restructure/run_model/data/output_data/wage-arms")

ARMS = [
    ("baseline", "A0 baseline (unchanged)"),
    ("no_incumbent_markup", "A1 no (1+m) on incumbents"),
    ("no_incumbent_effort", "A2 no u on incumbents"),
    ("no_incumbent_tfp", "A3 no TFP indexation"),
    ("cpi_indexation_090", "A4 CPI pass-through psi=0.9"),
    ("restructured", "A5 no u and no TFP"),
]

# (column, label, display scale, is_rate) — is_rate metrics are differenced in
# percentage points; level metrics are compared as percent changes.
METRICS = [
    ("cpi_chained_basket_yoy_change", "CPI YoY", 100, True),
    ("ppi_chained_yoy_change", "PPI YoY", 100, True),
    ("unemployment_rate", "Unemp", 100, True),
    ("vacancy_rate", "Vacancy", 100, True),
    ("real_gdp", "realGDP bn", 1e-9, False),
    ("avg_tfp_multiplier", "TFP", 1, False),
    ("economy_wage_rate", "wage m", 1e-6, False),
    ("deficit_to_gdp", "deficit/GDP", 100, True),
    ("labour_share", "labourShare", 100, True),
]


def load(arm):
    p = OUT / f"mc_{arm}.pkl"
    if not p.exists():
        return None
    df = pd.read_pickle(p)
    # Aggregate labour share = nominal wage bill / nominal GDP.  This is the
    # economy-level counterpart of the real unit labour cost whose drift Step 1
    # identified as the source of the loop-gain rise.
    if "wages" in df.columns and "gdp" in df.columns:
        df["labour_share"] = df["wages"] / df["gdp"]
    return df


def at(df, t, col):
    """Per-seed value of col at time t -> array."""
    if col not in df.columns:
        return np.array([])
    s = df.xs(t, level="time")[col]
    return s.to_numpy(dtype=float)


def main():
    frames = {a: load(a) for a, _ in ARMS}
    have = [a for a, _ in ARMS if frames[a] is not None]
    if not have:
        print("no arm outputs yet")
        return
    print(f"arms available: {have}")

    base = frames[have[0]]
    tmax = base.index.get_level_values("time").max()
    nseeds = base.index.get_level_values("seed").nunique()
    print(f"t_max={tmax}  seeds={nseeds}")

    # ---- Headline table: terminal values, mean +/- sd across seeds ----
    print()
    print("=" * 118)
    print(f"TERMINAL (t={tmax}) ACROSS SEEDS — mean [sd]")
    print("=" * 118)
    hdr = f"{'arm':<30}" + "".join(f"{lab:>16}" for _, lab, _, _ in METRICS)
    print(hdr)
    rows = {}
    for arm, label in ARMS:
        df = frames.get(arm)
        if df is None:
            continue
        cells, vals = [], {}
        for col, lab, scale, _is_rate in METRICS:
            v = at(df, tmax, col)
            if v.size == 0 or not np.isfinite(v).any():
                cells.append(f"{'n/a':>16}")
                continue
            v = v * scale
            vals[col] = v
            cells.append(f"{np.nanmean(v):>10.3f}[{np.nanstd(v):>4.2f}]")
        rows[arm] = vals
        print(f"{label:<30}" + "".join(cells))

    # ---- Deltas vs baseline, with a paired test across seeds ----
    if "baseline" in rows:
        print()
        print("=" * 118)
        print(f"DELTA vs BASELINE at t={tmax} (paired across seeds; ratio for level variables)")
        print("=" * 118)
        b = rows["baseline"]
        for arm, label in ARMS:
            if arm == "baseline" or arm not in rows:
                continue
            out = [f"{label:<30}"]
            for col, lab, scale, is_rate in METRICS:
                if col not in rows[arm] or col not in b:
                    out.append(f"{'n/a':>16}")
                    continue
                if is_rate:
                    d = rows[arm][col] - b[col]
                    out.append(f"{np.nanmean(d):>+12.3f}pp ")
                else:
                    r = (rows[arm][col] / b[col] - 1.0) * 100
                    out.append(f"{np.nanmean(r):>+12.3f}%  ")
            print("".join(out))

    # ---- Inflation path: is acceleration removed, or only the level? ----
    print()
    print("=" * 118)
    print("CPI YoY PATH (mean across seeds, %)")
    print("=" * 118)
    times = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    times = [t for t in times if t <= tmax]
    print(f"{'arm':<30}" + "".join(f"{'t=' + str(t):>9}" for t in times) + f"{'accel':>12}")
    for arm, label in ARMS:
        df = frames.get(arm)
        if df is None:
            continue
        path = [np.nanmean(at(df, t, "cpi_chained_basket_yoy_change")) * 100 for t in times]
        accel = path[-1] - path[times.index(25)] if 25 in times else np.nan
        print(f"{label:<30}" + "".join(f"{v:>9.3f}" for v in path) + f"{accel:>+12.3f}pp")

    print()
    print("=" * 118)
    print("UNEMPLOYMENT PATH (mean across seeds, %)")
    print("=" * 118)
    print(f"{'arm':<30}" + "".join(f"{'t=' + str(t):>9}" for t in times))
    for arm, label in ARMS:
        df = frames.get(arm)
        if df is None:
            continue
        path = [np.nanmean(at(df, t, "unemployment_rate")) * 100 for t in times]
        print(f"{label:<30}" + "".join(f"{v:>9.3f}" for v in path))

    print()
    print("=" * 118)
    print("LABOUR SHARE PATH (wages/GDP, mean across seeds, %) — the gain-drift mechanism")
    print("=" * 118)
    print(f"{'arm':<30}" + "".join(f"{'t=' + str(t):>9}" for t in times) + f"{'drift':>12}")
    for arm, label in ARMS:
        df = frames.get(arm)
        if df is None or "labour_share" not in df.columns:
            continue
        path = [np.nanmean(at(df, t, "labour_share")) * 100 for t in times]
        drift = path[-1] - path[times.index(15)] if 15 in times else np.nan
        print(f"{label:<30}" + "".join(f"{v:>9.3f}" for v in path) + f"{drift:>+12.3f}pp")

    # ---- Per-seed sign consistency for the headline claim ----
    if "baseline" in rows:
        print()
        print("=" * 118)
        print("PER-SEED CONSISTENCY: seeds with LOWER terminal CPI YoY than baseline (out of n)")
        print("=" * 118)
        for arm, label in ARMS:
            if arm == "baseline" or arm not in rows:
                continue
            d = rows[arm]["cpi_chained_basket_yoy_change"] - rows["baseline"]["cpi_chained_basket_yoy_change"]
            print(f"{label:<30} {int((d < 0).sum())}/{d.size}   mean {np.nanmean(d):+.4f}pp   "
                  f"min {np.nanmin(d):+.4f}  max {np.nanmax(d):+.4f}")


if __name__ == "__main__":
    main()
