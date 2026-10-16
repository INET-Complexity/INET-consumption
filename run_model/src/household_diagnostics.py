"""Household sector diagnostics from saved runs, without loading/running a model.

Money is nominal LCU, flows are per model period, stocks are closing balances.
Household shares/quantiles are unweighted. Missing series remain missing.
"""

from html import escape
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

# Panels specify one unit and one economic concept; never sum rates or shares.
PANELS = {
    "Consumption and income": [
        (
            "Total consumption: desired vs realised",
            "LCU bn / period",
            ["target_consumption_calibrated_total", "consumption_including_housing"],
        ),
        (
            "Goods orders after feasibility vs purchases",
            "LCU bn / period, ex VAT",
            ["target_consumption", "consumption"],
        ),
        (
            "Cash income vs behavioural income",
            "LCU bn / period",
            ["expected_income", "income", "expected_behavioural_income", "behavioural_income"],
        ),
        (
            "Realised cash income components",
            "LCU bn / period",
            [
                "income_employee",
                "income_public_pension",
                "income_unemployment_benefits",
                "income_other_social_transfers",
                "income_rental",
                "income_financial_assets",
            ],
        ),
        (
            "Income forecast error, excluding emergency support",
            "LCU bn / period",
            ["income_forecast_shortfall", "employee_income_forecast_shortfall", "stage5_subsistence_support"],
        ),
        (
            "ECM drivers (household mean)",
            "log points",
            [
                "target_consumption_partial_adjustment_gap",
                "income_growth_contribution",
                "target_consumption_interest_rate_cashflow",
                "target_consumption_uncertainty",
            ],
        ),
    ],
    "Credit and payment constraints": [
        (
            "Consumer credit: request vs grant",
            "LCU bn / period",
            ["credit_requested", "received_consumption_loans", "credit_rationing_gap"],
        ),
        (
            "Consumer service and payment distress",
            "LCU bn / period",
            [
                "scheduled_consumer_payment",
                "actual_consumer_payment",
                "unpaid_consumer_payment",
                "early_consumer_repayment",
            ],
        ),
        (
            "Total principal and interest cash payments",
            "LCU bn / period",
            ["debt_installments", "interest_paid_on_loans", "interest_paid_on_deposits"],
        ),
        (
            "Funding shortfall and available repairs",
            "LCU bn / period",
            [
                "liquidity_shortfall_before_repair",
                "funded_from_liquid_assets",
                "liquidation_planned",
                "consumption_cut_amount",
                "remaining_subsistence_shortfall",
            ],
        ),
        (
            "Constraint incidence",
            "% of households",
            ["dsti_cap_binding", "floor_binding", "consumer_payment_missed", "target_consumption_growth_clipped"],
        ),
        (
            "Realised loan service / cash income",
            "% (positive-income households)",
            ["realised_service_income_p50", "realised_service_income_p90"],
        ),
    ],
    "Balance sheets and overdrafts": [
        (
            "Gross positive liquidity vs overdrafts",
            "LCU bn",
            ["gross_positive_lfa", "overdrafts", "liquid_financial_assets"],
        ),
        (
            "Debt including negative deposits",
            "LCU bn",
            ["mortgage_debt", "consumption_loan_debt", "overdrafts", "total_credit_exposure"],
        ),
        (
            "Assets and net worth",
            "LCU bn",
            ["wealth_real_assets", "illiquid_financial_assets", "liquid_financial_assets", "net_wealth"],
        ),
        ("Overdraft incidence and new entries", "% of households", ["overdraft_share", "new_overdraft_share"]),
        ("Liquid assets distribution", "LCU million / simulated household", ["lfa_p10", "lfa_p50", "lfa_p90"]),
        ("Overdraft before vs after portfolio settlement", "LCU bn", ["pre_portfolio_overdrafts", "overdrafts"]),
    ],
    "Cash ledger and portfolio": [
        (
            "Realised cash sources and uses",
            "LCU bn / period",
            [
                "income_for_residual_saving",
                "consumption_cash_expenditure",
                "realised_household_expenditure",
                "debt_installments",
                "interest_paid",
            ],
        ),
        (
            "Cash saving before financing and new borrowing",
            "LCU bn / period",
            ["cash_saving_before_financing", "received_consumption_loans", "received_mortgages", "change_lfa"],
        ),
        (
            "Committed portfolio flows and costs",
            "LCU bn / period",
            [
                "portfolio_settlement_committed_lfa_flow",
                "portfolio_settlement_committed_ifa_flow",
                "portfolio_settlement_committed_adjustment_cost",
            ],
        ),
        (
            "Portfolio allocation and invalid raw shares",
            "%",
            ["gross_ifa_share", "portfolio_target_illiquid_share", "invalid_raw_portfolio_share"],
        ),
        (
            "Asset returns (household mean)",
            "% / period",
            ["portfolio_liquid_return_rate", "portfolio_illiquid_return_rate"],
        ),
        ("Cash identity residual (max absolute)", "LCU", ["cash_ledger_max_abs_residual"]),
    ],
}

MEANS = {
    "target_consumption_partial_adjustment_gap",
    "target_consumption_income_growth",
    "target_consumption_interest_rate_cashflow",
    "target_consumption_uncertainty",
}
PERCENT = {
    "dsti_cap_binding",
    "floor_binding",
    "consumer_payment_missed",
    "target_consumption_growth_clipped",
    "portfolio_actual_illiquid_share",
    "portfolio_target_illiquid_share",
    "portfolio_liquid_return_rate",
    "portfolio_illiquid_return_rate",
}
EXTRA = {
    "portfolio_actual_illiquid_share",
    "target_consumption_income_growth",
    "stage5_cash_ledger_residual",
    "portfolio_post_return_lfa",
    "target_consumption_real_lagged_consumption",
    "target_consumption_long_run",
    "target_consumption_real_income",
    "target_consumption_loans",
    "shadow_credit_requested",
}
DERIVED = {
    "income_forecast_shortfall",
    "employee_income_forecast_shortfall",
    "income_growth_contribution",
    "credit_requested",
    "credit_rationing_gap",
    "realised_service_income_p50",
    "realised_service_income_p90",
    "gross_positive_lfa",
    "overdrafts",
    "total_credit_exposure",
    "net_wealth",
    "overdraft_share",
    "new_overdraft_share",
    "lfa_p10",
    "lfa_p50",
    "lfa_p90",
    "pre_portfolio_overdrafts",
    "change_lfa",
    "cash_ledger_max_abs_residual",
    "gross_ifa_share",
    "invalid_raw_portfolio_share",
}


def build_household_diagnostics(source, *, country_code, index=None):
    """Read one saved run; return time series and explicit dataset coverage.

    ``source`` is a path or a workflow result with ``model_h5_path``. No alias
    maps illiquid assets back to the obsolete ``wealth_other_financial_assets``.
    Shape mismatches raise rather than aligning different periods silently.
    """
    path = Path(source.model_h5_path if hasattr(source, "model_h5_path") else source)
    names = {name for panels in PANELS.values() for _, _, series in panels for name in series}
    names = (names - DERIVED) | EXTRA | {"expected_income_employee", "wealth", "debt"}
    coverage = []
    with h5py.File(path, "r") as handle:
        group = handle[f"{country_code}/households"]
        shape = group["liquid_financial_assets"].shape
        if len(shape) != 2:
            raise ValueError("liquid_financial_assets must be time by household.")
        n_periods, n_households = shape
        arrays = {}
        for name in sorted(names):
            if name not in group:
                coverage.append({"dataset": name, "status": "missing", "finite_pct": 0.0})
                continue
            values = np.asarray(group[name], dtype=float)
            if f"{name}_columns" in group:
                columns = np.asarray(group[f"{name}_columns"], dtype=int)
                if values.ndim != 2 or columns.shape != (values.shape[1], 2):
                    raise ValueError(f"{name}: invalid saved household/industry columns.")
                household_ids = columns[:, 0]
                if np.any(household_ids < 0) or np.any(household_ids >= n_households):
                    raise ValueError(f"{name}: invalid household IDs in saved columns.")
                reduced = np.zeros(shape)
                np.add.at(reduced.T, household_ids, values.T)
                values = reduced
            elif values.ndim > 2:
                values = values.sum(axis=tuple(range(2, values.ndim)))
            if values.shape != shape:
                raise ValueError(f"{name}: expected {shape}, got {values.shape}.")
            arrays[name] = values
            coverage.append({"dataset": name, "status": "available", "finite_pct": 100 * np.isfinite(values).mean()})
    if index is None:
        index = pd.RangeIndex(n_periods, name="period")
    if len(index) != n_periods:
        raise ValueError("Macro index and household history have different lengths.")
    frame = pd.DataFrame(index=index)
    missing = np.full(shape, np.nan)

    def values(name):
        return arrays.get(name, missing)

    def total(array):
        # A missing household must not become a zero or a partial aggregate.
        return np.sum(array, axis=1)

    def mean(array):
        valid = np.isfinite(array)
        return np.divide(
            np.where(valid, array, 0).sum(axis=1),
            valid.sum(axis=1),
            out=np.full(n_periods, np.nan),
            where=valid.sum(axis=1) > 0,
        )

    for name, array in arrays.items():
        if name in MEANS:
            frame[name] = mean(array)
        elif name in PERCENT:
            frame[name] = 100 * mean(array)
        else:
            frame[name] = total(array) / 1e9
    lfa = values("liquid_financial_assets")
    frame["gross_positive_lfa"] = total(np.maximum(lfa, 0)) / 1e9
    frame["overdrafts"] = total(np.maximum(-lfa, 0)) / 1e9
    frame["overdraft_share"] = 100 * mean(np.where(np.isfinite(lfa), lfa < -1e-4, np.nan))
    new_entries = (lfa[1:] < -1e-4) & (lfa[:-1] >= -1e-4)
    frame["new_overdraft_share"] = np.r_[np.nan, 100 * new_entries.mean(axis=1)]
    frame["change_lfa"] = frame["liquid_financial_assets"].diff()
    # Negative deposits can make IFA/(LFA+IFA) exceed one. Do not present
    # that raw ratio as a bounded allocation share or silently clip it.
    gross_ifa = total(values("illiquid_financial_assets"))
    gross_assets = gross_ifa + total(np.maximum(lfa, 0))
    frame["gross_ifa_share"] = np.divide(
        100 * gross_ifa, gross_assets, out=np.full(n_periods, np.nan), where=gross_assets > 0
    )
    raw_share = values("portfolio_actual_illiquid_share")
    frame["invalid_raw_portfolio_share"] = 100 * mean(
        np.where(np.isfinite(raw_share), (raw_share < 0) | (raw_share > 1), np.nan)
    )
    frame["pre_portfolio_overdrafts"] = total(np.maximum(-values("portfolio_post_return_lfa"), 0)) / 1e9
    frame["total_credit_exposure"] = (
        total(values("mortgage_debt")) + total(values("consumption_loan_debt"))
    ) / 1e9 + frame["overdrafts"]
    frame["net_wealth"] = (total(values("wealth")) - total(values("debt"))) / 1e9
    for q in [10, 50, 90]:
        frame[f"lfa_p{q}"] = np.percentile(lfa, q, axis=1) / 1e6
    frame["income_forecast_shortfall"] = (
        total(values("expected_income") + values("stage5_subsistence_support") - values("income")) / 1e9
    )
    frame["employee_income_forecast_shortfall"] = (
        total(values("expected_income_employee") - values("income_employee")) / 1e9
    )
    # Stored income_growth is already the coefficient-weighted contribution.
    frame["income_growth_contribution"] = mean(values("target_consumption_income_growth"))
    request = values("target_consumption_loans")
    frame["credit_requested"] = total(request) / 1e9
    frame["credit_rationing_gap"] = total(np.maximum(request - values("received_consumption_loans"), 0)) / 1e9
    service = values("debt_installments") + values("interest_paid_on_loans")
    cash_income = values("income")
    ratio = np.divide(100 * service, cash_income, out=missing.copy(), where=cash_income > 0)
    for q in [50, 90]:
        frame[f"realised_service_income_p{q}"] = [
            np.percentile(row[np.isfinite(row)], q) if np.isfinite(row).any() else np.nan for row in ratio
        ]
    frame["cash_ledger_max_abs_residual"] = np.max(np.abs(values("stage5_cash_ledger_residual")), axis=1)
    # These histories contain initialization placeholders, not t0 decisions.
    decisions = (
        set(MEANS)
        | PERCENT
        | EXTRA
        | {
            "target_consumption_calibrated_total",
            "consumption_before_floor",
            "consumption_after_floor",
            "consumption_cut_amount",
            "funded_from_liquid_assets",
            "liquidity_shortfall_before_repair",
            "liquidation_planned",
            "remaining_subsistence_shortfall",
            "cash_saving_before_financing",
            "credit_requested",
            "credit_rationing_gap",
            "pre_portfolio_overdrafts",
            "cash_ledger_max_abs_residual",
            "realised_service_income_p50",
            "realised_service_income_p90",
            "income_forecast_shortfall",
            "employee_income_forecast_shortfall",
            "income_growth_contribution",
            "invalid_raw_portfolio_share",
        }
    )
    for name in decisions & set(frame.columns):
        frame.iloc[0, frame.columns.get_loc(name)] = np.nan
    frame.attrs.update(
        source=str(path),
        country_code=country_code,
        n_households=n_households,
        money_units="nominal LCU billions; flows per model period",
        shares="unweighted households",
    )
    return frame, pd.DataFrame(coverage).set_index("dataset")


def plot_household_diagnostics(runs, *, section):
    """Compare runs on six panels, with missing-series notices and explicit units."""
    panels = PANELS[section]
    fig = make_subplots(rows=3, cols=2, subplot_titles=[panel[0] for panel in panels], vertical_spacing=0.13)
    metric_names = list(dict.fromkeys(name for _, _, names in panels for name in names))
    colors = {name: qualitative.Dark24[i % len(qualitative.Dark24)] for i, name in enumerate(metric_names)}
    seen = set()
    for i, (_, unit, series) in enumerate(panels):
        row, col = divmod(i, 2)
        plotted = False
        for run_i, (run_name, frame) in enumerate(runs.items()):
            x = frame.index.astype(str) if isinstance(frame.index, pd.PeriodIndex) else frame.index
            for j, name in enumerate(series):
                if name not in frame or not frame[name].notna().any():
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=frame[name],
                        name=name.replace("_", " "),
                        legendgroup=name,
                        showlegend=(name not in seen),
                        connectgaps=False,
                        line=dict(color=colors[name], dash="solid" if run_i == 0 else "dash"),
                        hovertemplate="%{x}<br>%{y:.4g}<extra>"
                        + escape(run_name)
                        + ": "
                        + name.replace("_", " ")
                        + "</extra>",
                    ),
                    row=row + 1,
                    col=col + 1,
                )
                seen.add(name)
                plotted = True
        fig.update_yaxes(title_text=unit, zeroline=True, row=row + 1, col=col + 1)
        fig.update_xaxes(title_text="Period", row=row + 1, col=col + 1)
        if not plotted:
            fig.add_annotation(
                text="Unavailable in saved outputs — see coverage table", showarrow=False, row=row + 1, col=col + 1
            )
    run_key = "; ".join(f"{'Solid' if i == 0 else 'Dashed'}: {escape(name)}" for i, name in enumerate(runs))
    fig.update_layout(
        title=f"{section}<br><sup>{run_key}</sup>",
        height=1650,
        width=1450,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=-0.10, font=dict(size=10)),
        margin=dict(b=430, t=90),
        hovermode="x unified",
    )
    return fig


def household_diagnostic_summary(runs):
    """Compact initialization/final table; units follow the time-series columns."""
    metrics = [
        "consumption_including_housing",
        "expected_income",
        "income",
        "income_forecast_shortfall",
        "liquid_financial_assets",
        "overdrafts",
        "overdraft_share",
        "total_credit_exposure",
        "credit_rationing_gap",
        "consumption_cut_amount",
        "stage5_subsistence_support",
        "cash_ledger_max_abs_residual",
    ]
    records = []
    for run_name, frame in runs.items():
        for label, position in [("initial", 0), ("first period", min(1, len(frame) - 1)), ("final", len(frame) - 1)]:
            records.append(
                {
                    "run": run_name,
                    "observation": label,
                    **{key: frame[key].iloc[position] for key in metrics if key in frame},
                }
            )
    return pd.DataFrame(records).set_index(["run", "observation"])
