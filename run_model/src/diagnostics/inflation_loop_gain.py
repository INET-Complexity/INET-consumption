"""Step 1 (H0): reconstruct the inflation loop gain G_t from persisted diagnostics.

Test: labour MC = CPI_{t-1} * tau * sum(w_real) / E.  If the real wage is pinned
near a constant, then labour_mc / CPI_{t-1} is flat and the labour side of the
loop is pure pass-through (gain contribution 1).  Any drift in that ratio is
real unit labour cost, which is what makes G_t itself rise.
"""

import numpy as np
import pandas as pd

CSV = (
    "/Users/andone/Documents/python_projects/INET-consumption/run_model/data/"
    "output_data/issue-145-wage-price-decomposition/seed-18/period-decomposition.csv"
)

df = pd.read_csv(CSV)

# Rebuild CPI/PPI *levels* from the YoY series (quarterly model, 4 periods/year).
# YoY at t compares t to t-4, so chain the level forward in 4-period strides.
def level_from_yoy(yoy: pd.Series, periods_per_year: int = 4) -> np.ndarray:
    lvl = np.full(len(yoy), np.nan)
    lvl[:periods_per_year] = 1.0
    for t in range(periods_per_year, len(yoy)):
        g = yoy.iloc[t]
        lvl[t] = lvl[t - periods_per_year] * (1.0 + g) if np.isfinite(g) else np.nan
    return lvl


df["cpi_level"] = level_from_yoy(df["cpi_chained_basket_yoy_change"])
df["ppi_level"] = level_from_yoy(df["ppi_chained_yoy_change"])

# Real (deflated) unit costs.  Lag the deflator by one period: the wage
# obligation uses the CPI level stored before current-period prices form.
df["cpi_lag"] = df["cpi_level"].shift(1)
df["ppi_lag"] = df["ppi_level"].shift(1)
df["real_labour_uc"] = df["pricing_labour_mc_mean"] / df["cpi_lag"]
df["real_material_uc"] = df["pricing_material_mc_mean"] / df["ppi_lag"]
df["real_ac"] = df["pricing_ac_mean"] / df["cpi_lag"]

# Implied markup: price candidate is max(mu*MC, phi*AC); back out the realised
# gross margin as AC-inclusive unit cost vs the accounting unit cost.
df["mc_over_ac"] = df["pricing_mc_mean"] / df["pricing_ac_mean"]
df["labour_share_of_mc"] = df["pricing_labour_mc_mean"] / df["pricing_mc_mean"]

# Period-on-period growth of the loop's inputs.
for col in ["pricing_labour_mc_mean", "pricing_material_mc_mean",
            "pricing_mc_mean", "pricing_ac_mean", "cpi_level", "ppi_level",
            "real_labour_uc", "real_material_uc"]:
    df[f"g_{col}"] = df[col].pct_change()

rows = [15, 20, 25, 30, 35, 40, 45, 50]
rows = [r for r in rows if r < len(df)]

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:.5f}")

print("=" * 100)
print("LEVELS: nominal unit costs vs real (deflated) unit costs")
print("=" * 100)
print(df.loc[rows, [
    "time", "cpi_level", "ppi_level",
    "pricing_labour_mc_mean", "real_labour_uc",
    "pricing_material_mc_mean", "real_material_uc",
    "labour_share_of_mc", "pricing_ac_floor_binding_share",
]].to_string(index=False))

print()
print("=" * 100)
print("GAIN TEST: is the drift pure pass-through (real UC flat) or gain>1 (real UC rising)?")
print("=" * 100)
seg = df[(df["time"] >= 15) & (df["time"] <= 50)]
for name, col in [("nominal labour MC", "pricing_labour_mc_mean"),
                  ("real labour UC   ", "real_labour_uc"),
                  ("nominal materialMC", "pricing_material_mc_mean"),
                  ("real material UC ", "real_material_uc"),
                  ("CPI level        ", "cpi_level"),
                  ("PPI level        ", "ppi_level")]:
    a, b = seg[col].iloc[0], seg[col].iloc[-1]
    tot = (b / a - 1.0) * 100.0
    per = ((b / a) ** (1.0 / (len(seg) - 1)) - 1.0) * 100.0
    print(f"{name}  t=15 {a:10.5f} -> t=50 {b:10.5f}   total {tot:+7.2f}%   per-period {per:+6.3f}%")

print()
print("=" * 100)
print("ACCELERATION: is YoY inflation rising, and is the gain rising with it?")
print("=" * 100)
print(df.loc[rows, [
    "time", "cpi_chained_basket_yoy_change", "ppi_chained_yoy_change",
    "g_real_labour_uc", "g_real_material_uc", "unemployment_rate", "vacancy_rate",
]].to_string(index=False))

# Correlation between real-UC drift and subsequent inflation.
sub = df[(df["time"] >= 12) & np.isfinite(df["g_real_labour_uc"])]
if len(sub) > 5:
    print()
    print(f"corr(g_real_labour_uc, cpi_yoy)   = {sub['g_real_labour_uc'].corr(sub['cpi_chained_basket_yoy_change']):+.3f}")
    print(f"corr(g_real_material_uc, ppi_yoy) = {sub['g_real_material_uc'].corr(sub['ppi_chained_yoy_change']):+.3f}")
    print(f"mean g_real_labour_uc  (t>=15)    = {df.loc[df.time>=15,'g_real_labour_uc'].mean()*100:+.4f}% per period")
    print(f"mean g_real_material_uc(t>=15)    = {df.loc[df.time>=15,'g_real_material_uc'].mean()*100:+.4f}% per period")
