"""20-seed wage-arm comparison with the pricing cost decomposition.

Answers the question left open at 10 seeds: with the labour channel fixed, what
is still driving CPI acceleration?
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/Users/andone/Documents/python_projects/inet-wage-restructure/run_model/data/output_data/wage-arms-20")

ARMS = [
    ("baseline", "A0 baseline"),
    ("contract_firm_anchor", "U-A anchor"),
    ("contract_individual", "U-A-h indiv"),
    ("contract_realised_anchor", "U-A2b anchor"),
    ("contract_realised_individual", "U-A2b-h indiv"),
]

TIMES = [15, 20, 25, 30, 35, 40, 45, 50]
PAIRS = [(25, 30), (30, 35), (35, 40), (40, 45), (45, 50)]


def load(arm):
    p = OUT / f"mc_{arm}.pkl"
    if not p.exists():
        return None
    df = pd.read_pickle(p)
    df["labour_share"] = df["wages"] / df["gdp"]
    return df


def mean_at(df, t, col):
    if col not in df.columns:
        return np.nan
    return df.xs(t, level="time")[col].mean()


def main():
    frames = {a: load(a) for a, _ in ARMS}
    frames = {k: v for k, v in frames.items() if v is not None}
    n = next(iter(frames.values())).index.get_level_values("seed").nunique()
    print(f"seeds per arm: {n}")

    def table(title, col, scale=1.0, fmt="{:>9.3f}"):
        print()
        print("=" * 104)
        print(title)
        print("=" * 104)
        print(f"{'arm':<15}" + "".join(f"{'t=' + str(t):>9}" for t in TIMES) + f"{'t25->50':>11}")
        for arm, lab in ARMS:
            df = frames.get(arm)
            if df is None:
                continue
            vals = [mean_at(df, t, col) * scale for t in TIMES]
            delta = vals[-1] - vals[TIMES.index(25)]
            print(f"{lab:<15}" + "".join(fmt.format(v) for v in vals) + f"{delta:>+11.3f}")

    table("CPI YoY (%)", "cpi_chained_basket_yoy_change", 100)
    table("LABOUR SHARE wages/GDP (%)", "labour_share", 100)
    table("PRICING LABOUR MC (mean)", "pricing_labour_mc_mean", 1, "{:>9.4f}")
    table("PRICING MATERIAL MC (mean)", "pricing_material_mc_mean", 1, "{:>9.4f}")
    table("PRICING DEPRECIATION UNIT COST (mean)", "pricing_depreciation_unit_cost_mean", 1, "{:>9.4f}")
    table("PRICING AC (mean)", "pricing_ac_mean", 1, "{:>9.4f}")
    table("AC-FLOOR BINDING SHARE (%)", "pricing_ac_floor_binding_mean", 100)
    table("MARKUP mu (mean)", "pricing_markup_mu_mean", 1, "{:>9.4f}")

    # ---- Real (deflated) unit costs: pure pass-through vs gain drift ----
    print()
    print("=" * 104)
    print("REAL UNIT COSTS — cost deflated by the lagged price level")
    print("flat = pure pass-through (gain 1); rising = the loop gain is drifting")
    print("=" * 104)
    print(f"{'arm':<15}{'real labour UC':>32}{'real material UC':>32}")
    print(f"{'':<15}{'t=25':>10}{'t=50':>10}{'chg %':>12}{'t=25':>10}{'t=50':>10}{'chg %':>12}")
    for arm, lab in ARMS:
        df = frames.get(arm)
        if df is None:
            continue
        row = []
        for cost, index in (
            ("pricing_labour_mc_mean", "cpi_chained_basket"),
            ("pricing_material_mc_mean", "ppi_chained"),
        ):
            a = mean_at(df, 25, cost) / mean_at(df, 24, index)
            b = mean_at(df, 50, cost) / mean_at(df, 49, index)
            row += [a, b, (b / a - 1) * 100]
        print(
            f"{lab:<15}"
            + f"{row[0]:>10.4f}{row[1]:>10.4f}{row[2]:>+12.2f}"
            + f"{row[3]:>10.4f}{row[4]:>10.4f}{row[5]:>+12.2f}"
        )

    # ---- Is the rise itself slowing? ----
    print()
    print("=" * 104)
    print("CPI YoY per-period change (pp) — constant = linear rise; growing = accelerating")
    print("=" * 104)
    print(f"{'arm':<15}" + "".join(f"{f't{a}-{b}':>10}" for a, b in PAIRS) + f"{'last-first':>12}")
    for arm, lab in ARMS:
        df = frames.get(arm)
        if df is None:
            continue
        p = {t: mean_at(df, t, "cpi_chained_basket_yoy_change") * 100 for t in [25, 30, 35, 40, 45, 50]}
        ch = [p[b] - p[a] for a, b in PAIRS]
        print(f"{lab:<15}" + "".join(f"{c:>10.3f}" for c in ch) + f"{ch[-1] - ch[0]:>12.3f}")

    # ---- Terminal summary with dispersion ----
    print()
    print("=" * 104)
    print(f"TERMINAL t=50, mean [sd] across {n} seeds")
    print("=" * 104)
    metrics = [
        ("cpi_chained_basket_yoy_change", "CPI YoY %", 100),
        ("unemployment_rate", "Unemp %", 100),
        ("labour_share", "LabShare %", 100),
        ("real_gdp", "realGDP bn", 1e-9),
        ("economy_wage_rate", "wage m", 1e-6),
    ]
    print(f"{'arm':<15}" + "".join(f"{lab:>18}" for _, lab, _ in metrics))
    base = {}
    for arm, lab in ARMS:
        df = frames.get(arm)
        if df is None:
            continue
        cells = []
        for col, _, sc in metrics:
            v = df.xs(50, level="time")[col].to_numpy(dtype=float) * sc
            if arm == "baseline":
                base[col] = v
            cells.append(f"{np.nanmean(v):>11.3f}[{np.nanstd(v):>4.2f}]")
        print(f"{lab:<15}" + "".join(cells))

    print()
    print("Per-seed consistency vs baseline (terminal CPI YoY lower):")
    for arm, lab in ARMS:
        if arm == "baseline":
            continue
        df = frames.get(arm)
        if df is None:
            continue
        v = df.xs(50, level="time")["cpi_chained_basket_yoy_change"].to_numpy(dtype=float) * 100
        d = v - base["cpi_chained_basket_yoy_change"]
        print(
            f"  {lab:<15} {int((d < 0).sum())}/{d.size}  mean {np.nanmean(d):+.4f}pp  "
            f"min {np.nanmin(d):+.4f}  max {np.nanmax(d):+.4f}"
        )


if __name__ == "__main__":
    main()
