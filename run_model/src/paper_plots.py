"""Paper figures: one visual style for MPC and IRF charts.

All functions take already-computed panels/bands and only draw. Boxes show the
household distribution (whiskers at the 5th and 95th percentiles, no outliers),
dots the household mean.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

NAVY = "#1f3b5c"
STEEL = "#7a9cbf"
WARM = "#d08c4b"
GREY = "#8a8a8a"

# Low net liquid assets (first letter L) = hand-to-mouth side: warm; high = cool.
GROUP_SHORT = {
    "LLL": "Poor\nHtM",
    "LLH": "Housing-rich\nHtM",
    "LHL": "Financially\nwealthy\nHtM",
    "LHH": "Doubly\nwealthy\nHtM",
    "HLL": "Liquid\nwealth",
    "HLH": "Liquid\nhomeowners",
    "HHL": "Financially\nwealthy",
    "HHH": "Fully\ndiversified",
}
GROUP_COLORS = {
    "LLL": "#8c3b1f",
    "LLH": "#c0602b",
    "LHL": "#d99155",
    "LHH": "#ecc08f",
    "HLL": "#9fbcd6",
    "HLH": "#6f97bd",
    "HHL": "#3f6f9c",
    "HHH": "#1f3b5c",
}

STYLE = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "savefig.bbox": "tight",
}


def _save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=250)


def _boxes(ax, arrays, labels, colors, *, ylim=None):
    positions = np.arange(1, len(arrays) + 1)
    box = ax.boxplot(
        arrays,
        positions=positions,
        widths=0.6,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=1.4),
        whiskerprops=dict(color=GREY, linewidth=0.8),
        capprops=dict(color=GREY, linewidth=0.8),
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set(facecolor=color, edgecolor=color)
    ax.scatter(positions, [np.mean(a) for a in arrays], s=14, color="black", zorder=4)
    ax.set_xticks(positions, labels)
    ax.axhline(0, color=GREY, linewidth=0.6)
    if ylim is not None:
        ax.set_ylim(*ylim)


def mpc_impact_figure(panel, *, outcome, group_specs, path, ylim=(-0.1, 1.3)):
    """Box panels of the impact MPC.

    ``group_specs``: list of (column, title, ordered labels) for the top row,
    plus the 8 balance-sheet groups (column ``group_code``) on the bottom row
    next to the consumption liquidity constraint.
    Returns a table of group means.
    """
    rows = []

    def collect(column, order):
        arrays, labels = [], []
        for label in order:
            values = panel.loc[panel[column].eq(label).fillna(False), outcome].dropna()
            if values.empty:
                continue
            arrays.append(values.to_numpy())
            labels.append(label)
            rows.append({"panel": column, "group": label, "n": len(values), "mean": values.mean()})
        return arrays, labels

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(11.5, 7.0))
        grid = GridSpec(2, 4, figure=fig, height_ratios=[1, 1.15], hspace=0.45, wspace=0.15)
        top = [fig.add_subplot(grid[0, i]) for i in range(4)]
        bottom = fig.add_subplot(grid[1, :3], sharey=top[0])
        side = fig.add_subplot(grid[1, 3], sharey=top[0])
        for ax, (column, title, order) in zip(top + [side], group_specs):
            arrays, labels = collect(column, order)
            labels = [label.replace("No borrowing", "None").replace("Mortgage\nonly", "Mortgage") for label in labels]
            _boxes(ax, arrays, labels, [STEEL] * len(arrays), ylim=ylim)
            ax.set_title(title)
        codes = [c for c in GROUP_SHORT if (panel.group_code == c).any()]
        arrays, labels = collect("group_code", codes)
        _boxes(bottom, arrays, [GROUP_SHORT[c] for c in labels], [GROUP_COLORS[c] for c in labels], ylim=ylim)
        shares = panel.group_code.value_counts(normalize=True)
        for x, code in enumerate(labels, start=1):
            bottom.annotate(
                f"{code}\n{shares[code]:.0%}",
                (x, 1.0),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=7,
                color=GREY,
            )
        bottom.set_title("Balance-sheet groups (NLA · IFA · HA, high/low vs median)")
        for ax in top[1:] + [side]:
            ax.tick_params(labelleft=False)
        top[0].set_ylabel("Impact MPC")
        bottom.set_ylabel("Impact MPC")
        _save(fig, path)
        plt.show()
    return pd.DataFrame(rows)


def cumulative_mpc_figure(aggregate, by_group, *, path):
    """Cumulative MPC by horizon.

    ``aggregate``: DataFrame indexed by quarters with columns MPC, SE.
    ``by_group``: DataFrame indexed by quarters, one column per group code.
    """
    with plt.rc_context(STYLE):
        fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, gridspec_kw={"wspace": 0.08})
        q = aggregate.index.to_numpy()
        left.fill_between(
            q, aggregate.MPC - 1.96 * aggregate.SE, aggregate.MPC + 1.96 * aggregate.SE, color=STEEL, alpha=0.35, lw=0
        )
        left.plot(q, aggregate.MPC, color=NAVY, marker="o", lw=2)
        for x, y in zip(q, aggregate.MPC):
            left.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
        left.set_title("All households (95% CI across seeds)")
        for code in by_group.columns:
            right.plot(q, by_group[code], color=GROUP_COLORS[code], marker="o", ms=3, lw=1.6, label=code)
        right.set_title("By balance-sheet group")
        right.legend(ncol=2, fontsize=7, loc="lower right", title="NLA · IFA · HA", title_fontsize=7)
        for ax in (left, right):
            ax.set_xticks(q, [f"{int(x)}" for x in q])
            ax.set_xlabel("Quarters since transfer (cumulative)")
        left.set_ylabel("Cumulative MPC")
        _save(fig, path)
        plt.show()


def irf_figure(bands, *, shocks, variables, path):
    """Grid of IRFs: rows = shocks, columns = variables; mean line and 10–90 band."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            len(shocks), len(variables), figsize=(2.3 * len(variables), 2.0 * len(shocks)), squeeze=False
        )
        for i, (shock, shock_label) in enumerate(shocks.items()):
            for j, (var, var_label) in enumerate(variables.items()):
                ax = axes[i, j]
                d = bands[(bands.shock_name == shock) & (bands.variable == var)]
                ax.fill_between(d.horizon, d.p10, d.p90, color=STEEL, alpha=0.35, lw=0)
                ax.plot(d.horizon, d["mean"], color=NAVY, lw=1.8)
                ax.axhline(0, color=GREY, lw=0.6)
                if i == 0:
                    ax.set_title(var_label)
                if i == len(shocks) - 1:
                    ax.set_xlabel("Quarters")
            axes[i, 0].annotate(
                shock_label,
                xy=(-0.42, 0.5),
                xycoords="axes fraction",
                rotation=90,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )
        handles = [Line2D([], [], color=NAVY, lw=1.8), Patch(color=STEEL, alpha=0.35)]
        fig.legend(handles, ["Mean across seeds", "10th–90th percentile"], loc="lower center", ncol=2, fontsize=8)
        fig.tight_layout(rect=(0.02, 0.05, 1, 1))
        _save(fig, path)
        plt.show()


def multiplier_figure(curves, *, path, annotate=(4, 8, 16)):
    """Cumulative transfer multipliers by horizon.

    ``curves``: {regime label: DataFrame indexed by quarters with columns
    gdp, gdp_se, cons, cons_se}. The first regime is solid, others dashed.
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), sharey=True, gridspec_kw={"wspace": 0.08})
        for (label, d), style in zip(curves.items(), ("-", "--", ":")):
            for ax, var, color in ((axes[0], "gdp", NAVY), (axes[1], "cons", WARM)):
                q = d.index.to_numpy()
                ax.fill_between(
                    q, d[var] - 1.96 * d[f"{var}_se"], d[var] + 1.96 * d[f"{var}_se"], color=color, alpha=0.18, lw=0
                )
                ax.plot(q, d[var], color=color, lw=2, ls=style, label=label)
                if style == "-":
                    for h in annotate:
                        if h in d.index:
                            ax.plot(h, d.loc[h, var], "o", color=color, ms=4)
                            ax.annotate(
                                f"{d.loc[h, var]:.2f}",
                                (h, d.loc[h, var]),
                                textcoords="offset points",
                                xytext=(0, 8),
                                ha="center",
                                fontsize=8,
                            )
        axes[0].set_title("Real GDP")
        axes[1].set_title("Real household consumption")
        for ax in axes:
            ax.axhline(1, color=GREY, lw=0.6, ls=":")
            ax.axhline(0, color=GREY, lw=0.6)
            ax.set_xlabel("Quarters after the transfer")
        axes[0].set_ylabel("Cumulative multiplier (€ per € transferred)")
        if len(curves) > 1:
            axes[0].legend(fontsize=8, loc="lower right")
        _save(fig, path)
        plt.show()
