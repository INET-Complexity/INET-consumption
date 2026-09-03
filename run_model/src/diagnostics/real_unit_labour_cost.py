"""Step 1b: decompose the real unit labour cost drift found in step1.

real_labour_uc = tau * sum_j w_real_j / E   and   w_real = (1+m) TFP u w_init / tau
=> real_labour_uc ~ TFP * u * (1+m) * (employment / E)

If TFP is the driver, productivity growth is INFLATIONARY in this model, which
is structurally backwards.
"""

import h5py
import numpy as np

P = (
    "/Users/andone/Documents/python_projects/INET-consumption/run_model/data/"
    "output_data/issue-145-wage-price-decomposition/seed-18/multi_country_simulation.h5"
)

f = h5py.File(P, "r")


def get(name):
    return np.asarray(f[f"FRA/{name}"])


tfp = get("firms/tfp_multiplier")
lpf = get("firms/labour_productivity_factor")
li = get("firms/labour_inputs")
nemp = get("firms/number_of_employees")
rwpc = get("firms/real_wage_per_capita")
totwage = get("firms/total_wage")
prod = get("firms/production")

print("shapes:", tfp.shape, lpf.shape, li.shape, rwpc.shape)

T = tfp.shape[0]


def agg(x, w=None):
    """Cross-firm mean (optionally weighted) per period."""
    out = np.full(T, np.nan)
    for t in range(T):
        v = np.asarray(x[t], dtype=float).ravel()
        m = np.isfinite(v)
        if w is None:
            if m.any():
                out[t] = v[m].mean()
        else:
            ww = np.asarray(w[t], dtype=float).ravel()
            m = m & np.isfinite(ww) & (ww > 0)
            if m.any():
                out[t] = np.average(v[m], weights=ww[m])
    return out


tfp_m = agg(tfp, li)
lpf_m = agg(lpf, li)
li_tot = np.array([np.nansum(li[t]) for t in range(T)])
emp_tot = np.array([np.nansum(nemp[t]) for t in range(T)])
rw_m = agg(rwpc, nemp)
tw_tot = np.array([np.nansum(totwage[t]) for t in range(T)])
prod_tot = np.array([np.nansum(prod[t]) for t in range(T)])

rows = [15, 20, 25, 30, 35, 40, 45, 50]
print()
print("=" * 108)
print("REAL WAGE / TFP / EFFORT / LABOUR-INPUT PATHS  (labour-input weighted firm means)")
print("=" * 108)
hdr = f"{'t':>4} {'TFP':>10} {'u (effort)':>11} {'real wage pc':>14} {'labour inp':>13} {'employment':>12} {'wagebill/li':>13} {'output/li':>12}"
print(hdr)
for t in rows:
    if t >= T:
        continue
    print(
        f"{t:>4} {tfp_m[t]:>10.5f} {lpf_m[t]:>11.5f} {rw_m[t]:>14.5f} "
        f"{li_tot[t]:>13.4g} {emp_tot[t]:>12.4g} {tw_tot[t] / li_tot[t]:>13.5f} "
        f"{prod_tot[t] / li_tot[t]:>12.5f}"
    )

print()
print("=" * 108)
print("GROWTH t=15 -> t=50 (total %, and per-period %)")
print("=" * 108)


def show(label, series):
    a, b = series[15], series[50]
    if not (np.isfinite(a) and np.isfinite(b) and a != 0):
        print(f"{label:<34} n/a")
        return
    tot = (b / a - 1) * 100
    per = ((b / a) ** (1 / 35) - 1) * 100
    print(f"{label:<34} {a:>12.5g} -> {b:>12.5g}   total {tot:+8.2f}%   per-period {per:+7.4f}%")


show("TFP multiplier", tfp_m)
show("work-effort factor u", lpf_m)
show("real wage per capita", rw_m)
show("total labour inputs", li_tot)
show("employment", emp_tot)
show("real wage bill / labour input", tw_tot / li_tot)
show("output / labour input", prod_tot / li_tot)
show("TFP * u", tfp_m * lpf_m)

print()
print("KEY RATIO: real wage bill per unit labour input vs output per unit labour input")
print("If wages track TFP but output per labour input does not, real unit labour cost drifts up.")
ratio = (tw_tot / li_tot) / (prod_tot / li_tot)
show("wagebill/output (real unit lab cost)", ratio)
