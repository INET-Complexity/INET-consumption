"""Paper summaries of paired MPCs, using transfer amounts as weights."""

from pathlib import Path

import numpy as np
import pandas as pd


def write_paper_summaries(panel, output_dir, *, sample):
    """Write means, aggregate MPCs and uncertainty across independent seeds.

    Aggregate MPC = sum(real consumption change) / sum(real transfer).
    Standard errors use seed-level aggregate responses; households within a
    simulation are not treated as independent observations.
    """
    from src.mpc_paper_groups import prepare_paper_groups

    panel, specs = prepare_paper_groups(panel)
    rows = []
    columns = [c for c in panel if c.endswith("mpc_impact") or "cmpc_" in c]
    group_columns = [field for field, _title, _order in specs]
    groups = [("all", "all", panel)]
    for column in group_columns:
        groups.extend((column, str(label), frame) for label, frame in panel.groupby(column, observed=True))
    for variable, label, frame in groups:
        for outcome in columns:
            weight = "real_shock_amount" if "real_" in outcome else "shock_amount"
            if weight not in frame:
                continue
            values = pd.to_numeric(frame[outcome], errors="coerce")
            weights = pd.to_numeric(frame[weight], errors="coerce")
            valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
            if "cmpc_" in outcome and "cumulative_mpc_identified" in frame:
                valid &= frame["cumulative_mpc_identified"].eq(True)
            data = frame.loc[valid, ["seed"]].copy()
            data["change"] = values[valid] * weights[valid]
            data["transfer"] = weights[valid]
            if data.empty:
                continue
            totals = data.groupby("seed")[["change", "transfer"]].sum()
            estimate = data.change.sum() / data.transfer.sum()
            # Cluster ratio variance: each seed contributes its numerator residual.
            n = len(totals)
            residual = totals.change - estimate * totals.transfer
            se = np.sqrt(n / (n - 1) * np.square(residual).sum()) / data.transfer.sum() if n > 1 else np.nan
            rows.append(
                dict(
                    variable=variable,
                    group=label,
                    outcome=outcome,
                    n=int(valid.sum()),
                    seeds=n,
                    mean=values[valid].mean(),
                    median=values[valid].median(),
                    negative_share=(values[valid] < 0).mean(),
                    transfer_weighted_mpc=estimate,
                    seed_cluster_se=se,
                    total_transfer=data.transfer.sum(),
                )
            )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_columns = [
        "variable",
        "group",
        "outcome",
        "n",
        "seeds",
        "mean",
        "median",
        "negative_share",
        "transfer_weighted_mpc",
        "seed_cluster_se",
        "total_transfer",
    ]
    pd.DataFrame(rows, columns=summary_columns).to_csv(output_dir / f"paper_mpc_summary_{sample}.csv", index=False)
    diagnostic = [c for c in panel if c.startswith("diagnostic_")]
    if diagnostic:
        panel[["seed", "household_id", *diagnostic]].to_csv(
            output_dir / f"paper_mpc_cashflows_{sample}.csv", index=False
        )


def write_negative_mpc_diagnostics(panel, output_dir, *, sample):
    """Describe overlapping paired cash-flow changes among negative MPCs."""
    negative = panel.loc[panel.real_mpc_impact < 0].copy()
    rows = []
    for column in panel:
        if not column.startswith("diagnostic_") or not column.endswith("_t1") or "_shock_t" in column:
            continue
        field = column[:-3]
        treated = field + "_shock_t1"
        if treated not in panel:
            continue
        a = pd.to_numeric(negative[column], errors="coerce")
        b = pd.to_numeric(negative[treated], errors="coerce")
        valid = np.isfinite(a) & np.isfinite(b)
        tolerance = 1e-9 * np.maximum(np.maximum(a.abs(), b.abs()), 1.0)
        diff = b - a
        rows.append(
            dict(
                field=field,
                negative_n=len(negative),
                observed_n=int(valid.sum()),
                higher_n=int((valid & (diff > tolerance)).sum()),
                lower_n=int((valid & (diff < -tolerance)).sum()),
                mean_difference=diff[valid].mean(),
                median_difference=diff[valid].median(),
            )
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["field", "negative_n", "observed_n", "higher_n", "lower_n", "mean_difference", "median_difference"]
    pd.DataFrame(rows, columns=columns).to_csv(output_dir / f"negative_mpc_channels_{sample}.csv", index=False)
