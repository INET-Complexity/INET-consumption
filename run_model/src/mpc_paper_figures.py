"""Matplotlib paper figures from the paired household MPC panel."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_paper_figures(panel, output_dir, *, show_transfer_weighted_mean=True, measure="impact"):
    """Plot impact or four-quarter cumulative MPCs using common baseline groups.

    ``measure="cumulative_4q"`` applies the saved cumulative-identification flag,
    when present, after forming groups on the full analysis sample. The transfer
    is counted once in cumulative MPCs. Zero groups are omitted from display.
    """
    from src.mpc_paper_groups import prepare_paper_groups

    if measure not in {"impact", "cumulative_4q"}:
        raise ValueError("measure must be 'impact' or 'cumulative_4q'.")
    cumulative = measure == "cumulative_4q"
    outcome = "real_cmpc_4q" if cumulative else "real_mpc_impact"
    if outcome not in panel:
        raise ValueError(f"Missing MPC outcome: {outcome}.")
    # Keep the same income/asset boundaries as the impact figure; select only afterwards.
    panel, specs = prepare_paper_groups(panel)
    if cumulative and "cumulative_mpc_identified" in panel:
        panel = panel.loc[panel.cumulative_mpc_identified.eq(True)].copy()
    stem = "paper_cumulative_4q_mpc" if cumulative else "paper_impact_mpc"
    means_file = "paper_cumulative_4q_figure_group_means.csv" if cumulative else "paper_figure_group_means.csv"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharey=True)
    rows = []
    for ax, (field, title, order) in zip(axes.flat, specs):
        arrays = []
        labels = []
        weighted = []
        for label in order:
            if label == "Zero":
                continue
            subset = panel.loc[panel[field].eq(label).fillna(False)]
            valid = (
                np.isfinite(subset[outcome]) & np.isfinite(subset.real_shock_amount) & (subset.real_shock_amount > 0)
            )
            subset = subset.loc[valid]
            if subset.empty:
                continue
            arrays.append(subset[outcome])
            labels.append(label)
            weighted.append(np.average(subset[outcome], weights=subset.real_shock_amount))
            rows.append(
                dict(
                    variable=field,
                    group=label,
                    n=len(subset),
                    mean=subset[outcome].mean(),
                    transfer_weighted_mpc=weighted[-1],
                )
            )
        if arrays:
            ax.boxplot(
                arrays,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor="#799AB7"),
                medianprops=dict(color="#14213D", linewidth=1.6),
            )
            positions = np.arange(1, len(arrays) + 1)
            ax.scatter(positions, [a.mean() for a in arrays], s=18, color="black", label="Household mean", zorder=4)
            if show_transfer_weighted_mean:
                ax.scatter(
                    positions + 0.10,
                    weighted,
                    s=22,
                    marker="D",
                    color="#C46A27",
                    label="Transfer-weighted mean",
                    zorder=5,
                )
            ax.set_xticks(positions, labels, rotation=0, fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.axhline(0, color="grey", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    if cumulative:
        fig.suptitle("Cumulative MPC over four quarters", fontsize=14)
    fig.supylabel("MPC", fontsize=14)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0.025, 0.045, 1, 0.96 if cumulative else 1))
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=300)
    plt.close(fig)
    pd.DataFrame(rows, columns=["variable", "group", "n", "mean", "transfer_weighted_mpc"]).to_csv(
        output_dir / means_file, index=False
    )
