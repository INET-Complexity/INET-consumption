"""Paper figures: one visual style for MPC and IRF charts.

All functions take already-computed panels/bands and only draw. Boxes show the
household distribution (whiskers at the 5th and 95th percentiles, no outliers),
dots the household mean.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

NAVY = "#1f3b5c"
STEEL = "#7a9cbf"
WARM = "#d08c4b"
GREY = "#8a8a8a"
ZERO_LINE = dict(color="#333333", linewidth=0.8, zorder=1)
SHOCK_SHADE = dict(color="#000000", alpha=0.05, lw=0, zorder=0)

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
# Short tick labels for the top-row categories.
LABEL_SHORT = {
    "No borrowing": "None",
    "Consumer\ncredit": "Consumer\ncredit",
    "Mortgage\nonly": "Mort-\ngage",
    "Constrained": "Constrained",
    "Unconstrained": "Uncon-\nstrained",
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
    "axes.edgecolor": "#888888",
    "grid.color": "#e5e5e5",
    "grid.linewidth": 0.6,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "savefig.bbox": "tight",
}


def _save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=250)


def _boxes(ax, arrays, labels, colors, *, ylim=None, whis=(5, 95), showfliers=False):
    positions = np.arange(1, len(arrays) + 1)
    box = ax.boxplot(
        arrays,
        positions=positions,
        widths=0.6,
        whis=whis,
        showfliers=showfliers,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=1.4),
        whiskerprops=dict(color=GREY, linewidth=0.8),
        capprops=dict(color=GREY, linewidth=0.8),
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set(facecolor=color, edgecolor=color)
    ax.scatter(positions, [np.mean(a) for a in arrays], s=14, color="black", zorder=4)
    ax.set_xticks(positions, labels)
    ax.axhline(0, **ZERO_LINE)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _collect(panel, column, order, outcome, rows):
    arrays, labels = [], []
    for label in order:
        values = panel.loc[panel[column].eq(label).fillna(False), outcome].dropna()
        if values.empty:
            continue
        arrays.append(values.to_numpy())
        labels.append(label)
        rows.append({"panel": column, "group": label, "n": len(values), "mean": values.mean()})
    return arrays, labels


def mpc_characteristics_figure(
    panel, *, outcome, group_specs, path, ylabel="Impact MPC", ylim=(-0.1, 1.3), whis=(5, 95), showfliers=False
):
    """One box panel of the impact MPC per household characteristic.

    ``group_specs``: list of (column, title, ordered labels). Boxes show the
    household distribution (``whis`` percentile whiskers, or an IQR
    multiplier), dots the mean. ``ylim=None`` lets the axes fit the data.
    Display options do not filter observations. Returns a table of means.
    """
    rows = []
    n = len(group_specs)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 3.4), sharey=True, gridspec_kw={"wspace": 0.12})
        for ax, (column, title, order) in zip(np.atleast_1d(axes), group_specs):
            arrays, labels = _collect(panel, column, order, outcome, rows)
            labels = [LABEL_SHORT.get(label, label) for label in labels]
            _boxes(ax, arrays, labels, [STEEL] * len(arrays), ylim=ylim, whis=whis, showfliers=showfliers)
            ax.set_title(title)
        np.atleast_1d(axes)[0].set_ylabel(ylabel)
        _save(fig, path)
        plt.show()
    return pd.DataFrame(rows)


def _share_text(share):
    return "<1%" if 0 < share < 0.005 else f"{share:.0%}"


def mpc_group_constraint_figure(
    panel,
    *,
    outcome,
    path,
    split_column="constraint",
    split_order=("Constrained", "Unconstrained"),
    title="Impact MPC by balance-sheet group and liquidity constraint",
    min_split_n=200,
    width_by_share=False,
    ylabel="Impact MPC",
    ylim=(-0.1, 1.3),
    whis=(5, 95),
    showfliers=False,
):
    """Impact MPC of each balance-sheet group, split by liquidity constraint.

    A 2x4 grid, one panel per group (``GROUP_SHORT`` order): low net liquid
    assets (L..) in the top row, high (H..) in the bottom row. Each panel title
    gives the group code and its population share, e.g. "LLL (17%)". Inside, households in the
    first ``split_order`` category (full colour, "C") and the second (light,
    "U"), with their share of the group under each box. A hollow diamond
    between the boxes marks the group mean, the share-weighted average of the
    two subgroup means, with its value printed above it.
    With ``width_by_share`` box widths are 0.8 x the subgroup share (at least
    0.12 so small subgroups stay visible). Subgroups with fewer than ``min_split_n`` pooled observations are
    not drawn. Returns a table of group and subgroup means.
    """
    rows = []
    codes = [c for c in GROUP_SHORT if (panel.group_code == c).any()]
    shares = panel.group_code.value_counts(normalize=True)
    short = {split_order[0]: "C", split_order[1]: "U"}
    n_cols = int(np.ceil(len(codes) / 2))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, n_cols, figsize=(2.6 * n_cols, 5.6), sharey=True, squeeze=False)
        for ax, code in zip(axes.flat, codes):
            in_group = panel.group_code.eq(code)
            group_values = panel.loc[in_group, outcome].dropna()
            group_mean = group_values.mean()
            rows.append({"panel": "group_code", "group": code, "n": len(group_values), "mean": group_mean})
            labels = []
            for j, level in enumerate(split_order):
                sub = panel.loc[in_group & panel[split_column].eq(level).fillna(False), outcome].dropna()
                rows.append({"panel": split_column, "group": f"{code} {level}", "n": len(sub), "mean": sub.mean()})
                share = len(sub) / len(group_values)
                labels.append(f"{short.get(level, level)}\n{_share_text(share)}")
                if len(sub) < min_split_n:
                    continue
                box = ax.boxplot(
                    [sub.to_numpy()],
                    positions=[j],
                    widths=max(0.8 * share, 0.12) if width_by_share else 0.6,
                    whis=whis,
                    showfliers=showfliers,
                    patch_artist=True,
                    medianprops=dict(color="white", linewidth=1.4),
                    whiskerprops=dict(color=GREY, linewidth=0.8),
                    capprops=dict(color=GREY, linewidth=0.8),
                )
                box["boxes"][0].set(
                    facecolor=GROUP_COLORS[code], edgecolor=GROUP_COLORS[code], alpha=1.0 if j == 0 else 0.4
                )
                ax.scatter([j], [sub.mean()], s=14, color="black", zorder=6)
            ax.scatter(
                [0.5], [group_mean], marker="D", s=42, facecolor="white", edgecolor="black", linewidth=1.3, zorder=7
            )
            ax.annotate(
                f"{group_mean:.2f}",
                (0.5, group_mean),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7.5,
            )
            ax.set_xticks([0, 1], labels)
            ax.set_xlim(-0.6, 1.6)
            ax.axhline(0, **ZERO_LINE)
            ax.set_title(f"{code} ({shares[code]:.0%})", fontsize=9.5)
        for ax in axes.flat[len(codes) :]:
            ax.set_visible(False)
        if ylim is not None:
            axes[0, 0].set_ylim(*ylim)
        # ``ylabel`` is one string for both rows, or a (top, bottom) pair.
        top_label, bottom_label = (ylabel, ylabel) if isinstance(ylabel, str) else ylabel
        axes[0, 0].set_ylabel(top_label)
        axes[1, 0].set_ylabel(bottom_label)
        handles = [
            Patch(facecolor=GREY, edgecolor=GREY),
            Patch(facecolor=GREY, edgecolor=GREY, alpha=0.4),
            Line2D([], [], marker="D", ls="", ms=6, markerfacecolor="white", markeredgecolor="black", mew=1.3),
            Line2D([], [], color="black", marker="o", ls="", ms=4),
        ]
        fig.legend(
            handles,
            [f"{split_order[0]} (C)", f"{split_order[1]} (U)", "Group mean", "Subgroup mean"],
            loc="lower center",
            ncol=4,
            fontsize=8,
        )
        fig.suptitle(title, fontweight="bold", fontsize=11)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        _save(fig, path)
        plt.show()
    return pd.DataFrame(rows)


def cumulative_mpc_figure(aggregate, by_group, *, path, ylabel="Cumulative MPC"):
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
        right.legend(ncol=2, fontsize=7, loc="lower right", title="NLA, IFA, HA", title_fontsize=7)
        for ax in (left, right):
            ax.set_xticks(q, [f"{int(x)}" for x in q])
            ax.set_xlabel("Quarters since transfer (cumulative)")
        left.set_ylabel(ylabel)
        _save(fig, path)
        plt.show()


def align_zero(axes, *, pad: float = 0.05) -> None:
    """Set y-limits so that y = 0 sits at the same height in every axis.

    Each axis keeps its own scale. The common zero position is chosen on a grid
    to minimise empty space; each axis then gets the smallest range that holds
    its data (plus ``pad``) with zero at that position.
    """
    extents = []
    for ax in axes:
        lo, hi = ax.dataLim.y0, ax.dataLim.y1
        neg, pos = max(-min(lo, 0.0), 0.0), max(max(hi, 0.0), 0.0)
        if neg + pos == 0.0:
            neg = pos = 1.0
        extents.append((neg * (1 + pad), pos * (1 + pad)))

    def spans(frac):
        return [max(n / frac, p / (1 - frac)) for n, p in extents]

    def waste(frac):
        return sum(1 - (n + p) / s for (n, p), s in zip(extents, spans(frac)))

    frac = min(np.linspace(0.02, 0.98, 97), key=waste)
    for ax, span in zip(axes, spans(frac)):
        ax.set_ylim(-frac * span, (1 - frac) * span)


def irf_figure(bands, *, shocks, variables, path, shock_durations=None, align_zero_rows=True):
    """Grid of IRFs: rows = shocks, columns = variables.

    ``bands`` holds per (shock, variable, horizon) the ``mean`` and the seed
    quantiles ``p10``/``p90`` (outer band) and, if present, ``p25``/``p75``
    (inner band). ``shock_durations`` maps shock name to the number of quarters
    with an active impulse; the shaded band is centred on those horizons,
    h = 0 .. duration - 1, i.e. it spans [-0.5, duration - 0.5]. With
    ``align_zero_rows`` the zero line sits at the same height across each row.
    """
    shock_durations = shock_durations or {}
    inner = {"p25", "p75"}.issubset(bands.columns)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            len(shocks),
            len(variables),
            figsize=(2.3 * len(variables), 2.0 * len(shocks)),
            squeeze=False,
            sharex=True,
        )
        for i, (shock, shock_label) in enumerate(shocks.items()):
            for j, (var, var_label) in enumerate(variables.items()):
                ax = axes[i, j]
                d = bands[(bands.shock_name == shock) & (bands.variable == var)]
                if shock_durations.get(shock):
                    ax.axvspan(-0.5, shock_durations[shock] - 0.5, **SHOCK_SHADE)
                ax.fill_between(d.horizon, d.p10, d.p90, color=STEEL, alpha=0.22, lw=0)
                if inner:
                    ax.fill_between(d.horizon, d.p25, d.p75, color=STEEL, alpha=0.45, lw=0)
                ax.plot(d.horizon, d["mean"], color=NAVY, lw=1.8)
                ax.axhline(0, **ZERO_LINE)
                ax.set_xlim(d.horizon.min() - 0.5, d.horizon.max())
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
                if i == 0:
                    ax.set_title(var_label)
                if i == len(shocks) - 1:
                    ax.set_xlabel("Quarters")
            if align_zero_rows:
                align_zero(axes[i])
            for ax in axes[i]:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
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
        handles = [Line2D([], [], color=NAVY, lw=1.8)]
        labels = ["Mean across seeds"]
        if inner:
            handles.append(Patch(color=STEEL, alpha=0.55))
            labels.append("25th–75th percentile")
        handles.append(Patch(color=STEEL, alpha=0.22))
        labels.append("10th–90th percentile")
        if shock_durations:
            handles.append(Patch(color="#000000", alpha=0.08))
            labels.append("Shock active")
        fig.legend(handles, labels, loc="lower center", ncol=len(handles), fontsize=8)
        fig.tight_layout(rect=(0.02, 0.05, 1, 1))
        _save(fig, path)
        plt.show()


def multiplier_figure(
    curves,
    *,
    path,
    annotate=(4, 8, 16),
    titles=("Real GDP", "Real household consumption (incl. VAT)"),
):
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
        axes[0].set_title(titles[0])
        axes[1].set_title(titles[1])
        for ax in axes:
            ax.axhline(1, color=GREY, lw=0.6, ls=":")
            ax.axhline(0, **ZERO_LINE)
            ax.set_xlabel("Quarters after the transfer")
        axes[0].set_ylabel("€ per € transferred")
        if len(curves) > 1:
            axes[0].legend(fontsize=8, loc="lower right")
        _save(fig, path)
        plt.show()
