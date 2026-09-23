"""Firm-level diagnostics for policy-rate borrowing and investment experiments."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# These are all two-dimensional (period, firm) fields written by Firms.save_to_h5.
FIRM_POLICY_RATE_SERIES: tuple[str, ...] = (
    "deposits",
    "capital_inputs_stock_value",
    "credit_budget_hard_obligations",
    "credit_budget_cash_after_hard_obligations",
    "credit_budget_remaining_internal_finance_after_working_capital",
    "credit_budget_capital_costs",
    "credit_budget_technical_investment_costs",
    "credit_budget_tfp_costs",
    "credit_budget_investment_budget",
    "target_short_term_credit",
    "target_debt_rollover_credit",
    "target_overdraft_refinance_credit",
    "target_operating_refinance_credit",
    "ordinary_target_short_term_credit",
    "target_long_term_credit",
    "received_short_term_credit",
    "received_debt_rollover_credit",
    "received_overdraft_refinance_credit",
    "received_operating_refinance_credit",
    "received_ordinary_short_term_credit",
    "received_long_term_credit",
    "received_credit",
    "scheduled_debt_service",
    "debt_installments",
    "debt",
    "short_term_loan_debt",
    "long_term_loan_debt",
    "interest_paid",
    "interest_paid_on_loans",
    "firm_settlement_debt_rollover_shortfall",
    "firm_settlement_opening_principal_arrears",
    "firm_settlement_closing_principal_arrears",
    "tfp_investment_effective_cost_rate",
    "planned_tfp_investment",
    "executed_tfp_investment",
    "planned_productivity_investment",
    "planned_tfp_investment_expected_costs",
    "feasible_tfp_investment_expected_costs",
    "tfp_investment_cash_binding",
    "tfp_investment_cap_binding",
    "tfp_investment_finance_scale",
    "tfp_investment_target_intensity",
    "tfp_investment_desired_intensity",
    "tfp_investment_planned_intensity",
    "tfp_multiplier",
    "production",
    "production_nominal",
    "credit_market_firm_lt_capacity",
    "credit_market_firm_lt_collateral_cap",
    "credit_market_firm_lt_dscr_cap",
    "credit_market_firm_lt_binding_reason",
    "credit_market_firm_st_capacity",
    "credit_market_firm_st_collateral_cap",
    "credit_market_firm_st_dscr_cap",
    "credit_market_firm_st_binding_reason",
)

FIRM_POLICY_RATE_MEAN_SERIES = frozenset(
    {
        "tfp_investment_effective_cost_rate",
        "tfp_investment_cash_binding",
        "tfp_investment_cap_binding",
        "tfp_investment_finance_scale",
        "tfp_investment_target_intensity",
        "tfp_investment_desired_intensity",
        "tfp_investment_planned_intensity",
        "tfp_multiplier",
        "credit_market_firm_lt_binding_reason",
        "credit_market_firm_st_binding_reason",
    }
)

MACRO_POLICY_RATE_VARIABLES = (
    ("policy_rate", "{country}/central_Bank/policy_rate", "first"),
    ("firm_short_term_loan_rate", "{country}/banks/average_interest_rates_on_short_term_firm_loans", "mean"),
    ("firm_long_term_loan_rate", "{country}/banks/average_interest_rates_on_long_term_firm_loans", "mean"),
    ("firm_overdraft_rate", "{country}/banks/average_overdraft_rate_on_firm_deposits", "mean"),
    ("ppi", "{country}/economy/ppi", "first"),
    ("ppi_inflation", "{country}/economy/ppi_inflation", "first"),
    ("cpi", "{country}/economy/cpi_fixed_basket", "first"),
    ("gdp_output", "{country}/economy/gdp_output", "first"),
    ("gdp_expenditure", "{country}/economy/gdp_expenditure", "first"),
    ("gdp_income", "{country}/economy/gdp_income", "first"),
    ("unemployment_rate", "{country}/economy/unemployment_rate", "first"),
    ("firm_production", "{country}/firms/production", "sum"),
    ("firm_target_production", "{country}/firms/target_production", "sum"),
    ("average_tfp_multiplier", "{country}/firms/tfp_multiplier", "mean"),
    ("planned_tfp_investment", "{country}/firms/planned_tfp_investment", "sum"),
    ("executed_tfp_investment", "{country}/firms/executed_tfp_investment", "sum"),
    ("target_long_term_credit", "{country}/firms/target_long_term_credit", "sum"),
    ("received_long_term_credit", "{country}/firms/received_long_term_credit", "sum"),
    ("target_short_term_credit", "{country}/firms/target_short_term_credit", "sum"),
    ("received_short_term_credit", "{country}/firms/received_short_term_credit", "sum"),
    ("firm_total_credit_exposure", "{country}/firms/total_credit_exposure", "first"),
)

_BINDING_REASON_LABELS = {
    0.0: "not_binding",
    1.0: "no_demand",
    2.0: "collateral",
    3.0: "dscr",
    4.0: "roa",
    5.0: "roe",
    6.0: "bank_car",
    7.0: "bank_preference",
    8.0: "other",
}


def _read_firm_series(handle: h5py.File, country_code: str, name: str) -> np.ndarray:
    dataset_path = f"{country_code}/firms/{name}"
    if dataset_path not in handle:
        raise KeyError(f"Required firm policy-rate diagnostic is missing: {dataset_path}")
    values = np.asarray(handle[dataset_path], dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Firm diagnostic {dataset_path} must have shape (period, firm), got {values.shape}")
    return values


def load_firm_policy_rate_panel(
    h5_path: str | Path,
    *,
    country_code: str = "FRA",
) -> pd.DataFrame:
    """Load firm time series and derive liquidity, funding, and binding indicators."""
    with h5py.File(h5_path, "r") as handle:
        values = {name: _read_firm_series(handle, country_code, name) for name in FIRM_POLICY_RATE_SERIES}

    shape = values["deposits"].shape
    mismatched = {name: value.shape for name, value in values.items() if value.shape != shape}
    if mismatched:
        raise ValueError(f"Firm diagnostic arrays must share shape {shape}; mismatches: {mismatched}")

    n_periods, n_firms = shape
    frame = pd.DataFrame(
        {
            "period": np.repeat(np.arange(n_periods), n_firms),
            "firm": np.tile(np.arange(n_firms), n_periods),
        }
    )
    for name, value in values.items():
        frame[name] = value.reshape(-1)

    frame["positive_deposits"] = frame["deposits"].clip(lower=0.0)
    frame["capital_internal_finance"] = np.minimum(
        frame["credit_budget_remaining_internal_finance_after_working_capital"].clip(lower=0.0),
        frame["credit_budget_capital_costs"].clip(lower=0.0),
    )
    residual_after_capital = (
        frame["credit_budget_remaining_internal_finance_after_working_capital"].clip(lower=0.0)
        - frame["capital_internal_finance"]
    )
    productivity_costs = frame["credit_budget_technical_investment_costs"] + frame["credit_budget_tfp_costs"]
    frame["capital_funding_gap"] = (
        frame["credit_budget_capital_costs"] - frame["capital_internal_finance"]
    ).clip(lower=0.0)
    frame["productivity_funding_gap"] = (
        productivity_costs - np.minimum(residual_after_capital, productivity_costs)
    ).clip(lower=0.0)
    frame["tfp_execution_gap"] = (
        frame["planned_tfp_investment"] - frame["executed_tfp_investment"]
    ).clip(lower=0.0)

    frame["cash_constrained"] = frame["credit_budget_cash_after_hard_obligations"] < 0.0
    frame["debt_stressed"] = (
        (frame["target_debt_rollover_credit"] > 1e-12)
        | (frame["target_overdraft_refinance_credit"] > 1e-12)
        | (frame["firm_settlement_debt_rollover_shortfall"] > 1e-12)
        | (frame["firm_settlement_opening_principal_arrears"] > 1e-12)
        | (frame["firm_settlement_closing_principal_arrears"] > 1e-12)
    )
    frame["liquid"] = (
        (frame["deposits"] > 0.0)
        & ~frame["cash_constrained"]
    )
    frame["liquidity_group"] = np.select(
        [
            frame["deposits"] < 0.0,
            frame["liquid"],
            frame["cash_constrained"],
        ],
        ["overdraft", "liquid", "cash_constrained"],
        default="neutral",
    )
    frame["lt_binding_reason_label"] = frame["credit_market_firm_lt_binding_reason"].map(_BINDING_REASON_LABELS)
    frame["st_binding_reason_label"] = frame["credit_market_firm_st_binding_reason"].map(_BINDING_REASON_LABELS)
    return frame


def build_firm_policy_rate_irf_panel(
    baseline_h5: str | Path,
    shock_h5: str | Path,
    *,
    seed: int,
    shock_name: str,
    shock_kind: str,
    shock_period: int,
    shock_magnitude: float,
    shock_duration: int,
    shock_mode: str = "additive",
    horizon_periods: int = 50,
    country_code: str = "FRA",
    dscr_enabled: bool | None = None,
    variables: Sequence[str] = FIRM_POLICY_RATE_SERIES,
) -> pd.DataFrame:
    """Build a firm-period paired IRF panel classified by baseline liquidity."""
    if shock_period < 0 or horizon_periods <= 0:
        raise ValueError("shock_period must be non-negative and horizon_periods must be positive")

    baseline = load_firm_policy_rate_panel(baseline_h5, country_code=country_code)
    shock = load_firm_policy_rate_panel(shock_h5, country_code=country_code)
    keys = ["period", "firm"]
    baseline = baseline.set_index(keys)
    shock = shock.set_index(keys)
    if not baseline.index.equals(shock.index):
        raise ValueError("Baseline and shock firm panels do not have identical period/firm indexes")

    stop = min(shock_period + horizon_periods, int(baseline.index.get_level_values("period").max()) + 1)
    selected_index = baseline.index[
        (baseline.index.get_level_values("period") >= shock_period)
        & (baseline.index.get_level_values("period") < stop)
    ]
    base_selected = baseline.loc[selected_index]
    shock_selected = shock.loc[selected_index]

    rows: list[pd.DataFrame] = []
    for variable in variables:
        if variable not in baseline.columns:
            raise KeyError(f"Unknown firm policy-rate diagnostic variable: {variable}")
        values = pd.DataFrame(
            {
                "baseline": base_selected[variable].to_numpy(dtype=float),
                "shock": shock_selected[variable].to_numpy(dtype=float),
            },
            index=selected_index,
        )
        values["delta"] = values["shock"] - values["baseline"]
        values["pct_delta"] = np.divide(
            values["delta"],
            values["baseline"],
            out=np.full(len(values), np.nan),
            where=values["baseline"].to_numpy(dtype=float) != 0.0,
        )
        values["variable"] = variable
        rows.append(values.reset_index())

    panel = pd.concat(rows, ignore_index=True)
    classification = base_selected[["liquidity_group", "debt_stressed"]].reset_index()
    panel = panel.merge(classification, on=keys, how="left", validate="many_to_one")
    panel["horizon"] = panel["period"] - int(shock_period)
    panel["seed"] = int(seed)
    panel["shock_name"] = shock_name
    panel["shock_kind"] = shock_kind
    panel["shock_period"] = int(shock_period)
    panel["shock_magnitude"] = float(shock_magnitude)
    panel["shock_duration"] = int(shock_duration)
    panel["shock_mode"] = shock_mode
    panel["dscr_enabled"] = dscr_enabled
    return panel


def _aggregate_grouped_values(panel: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    variable_position = group_columns.index("variable")
    for group_key, group in panel.groupby(group_columns, dropna=False, sort=False):
        variable = str(group_key[variable_position])
        reducer = "mean" if variable in FIRM_POLICY_RATE_MEAN_SERIES else "sum"
        baseline = float(getattr(group["baseline"], reducer)())
        shock = float(getattr(group["shock"], reducer)())
        delta = shock - baseline
        rows.append(
            {
                **dict(zip(group_columns, group_key, strict=True)),
                "baseline": baseline,
                "shock": shock,
                "delta": delta,
                "pct_delta": np.nan if baseline == 0.0 else delta / baseline,
            }
        )
    return pd.DataFrame(rows)


def aggregate_firm_policy_rate_irf(panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate firm-period IRFs by seed, liquidity group, and debt stress."""
    if panel.empty:
        return panel.copy()
    group_columns = [
        "seed",
        "shock_name",
        "shock_kind",
        "shock_period",
        "shock_magnitude",
        "shock_duration",
        "shock_mode",
        "dscr_enabled",
        "period",
        "horizon",
        "liquidity_group",
        "debt_stressed",
        "variable",
    ]
    grouped = _aggregate_grouped_values(panel, group_columns)

    overall_columns = [column for column in group_columns if column not in {"liquidity_group", "debt_stressed"}]
    overall = _aggregate_grouped_values(panel, overall_columns)
    overall["liquidity_group"] = "all"
    overall["debt_stressed"] = "all"
    return pd.concat([grouped, overall], ignore_index=True, sort=False)


def summarize_firm_policy_rate_irf(
    aggregate_panel: pd.DataFrame,
    *,
    n_bootstrap: int = 500,
    random_state: int = 20260910,
) -> pd.DataFrame:
    """Summarize paired seed responses and bootstrap confidence intervals."""
    if aggregate_panel.empty:
        return aggregate_panel.copy()
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")

    group_columns = [
        "shock_name",
        "shock_kind",
        "shock_period",
        "shock_magnitude",
        "shock_duration",
        "shock_mode",
        "dscr_enabled",
        "period",
        "horizon",
        "liquidity_group",
        "debt_stressed",
        "variable",
    ]
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    for group_key, group in aggregate_panel.groupby(group_columns, dropna=False, sort=False):
        values = group["delta"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        bootstrap_indices = rng.integers(0, values.size, size=(n_bootstrap, values.size))
        bootstrap_means = values[bootstrap_indices].mean(axis=1)
        row = dict(zip(group_columns, group_key, strict=True))
        row.update(
            {
                "n_seeds": int(values.size),
                "delta_mean": float(values.mean()),
                "delta_median": float(np.median(values)),
                "delta_p10": float(np.quantile(values, 0.10)),
                "delta_p90": float(np.quantile(values, 0.90)),
                "bootstrap_ci_low": float(np.quantile(bootstrap_means, 0.025)),
                "bootstrap_ci_high": float(np.quantile(bootstrap_means, 0.975)),
                "negative_seed_share": float(np.mean(values < 0.0)),
                "positive_seed_share": float(np.mean(values > 0.0)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def classify_tfp_dose_response(
    summary: pd.DataFrame,
    *,
    variable: str = "planned_tfp_investment",
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Check monotonicity and sign consistency across policy-rate shock sizes."""
    if summary.empty:
        return summary.copy()
    data = summary[summary["variable"] == variable].copy()
    if data.empty:
        return data
    grouping = ["shock_kind", "shock_duration", "dscr_enabled", "horizon", "liquidity_group", "debt_stressed"]
    rows: list[dict[str, object]] = []
    for group_key, group in data.groupby(grouping, dropna=False, sort=False):
        group = group.sort_values("shock_magnitude")
        deltas = group["delta_mean"].to_numpy(dtype=float)
        magnitudes = group["shock_magnitude"].to_numpy(dtype=float)
        if len(deltas) < 2:
            continue
        differences = np.diff(deltas)
        rows.append(
            {
                **dict(zip(grouping, group_key, strict=True)),
                "shock_magnitudes": ",".join(str(float(value)) for value in magnitudes),
                "delta_means": ",".join(str(float(value)) for value in deltas),
                "monotone_nonincreasing": bool(np.all(differences <= tolerance)),
                "all_nonpositive": bool(np.all(deltas <= tolerance)),
                "max_step_increase": float(np.max(differences)),
            }
        )
    return pd.DataFrame(rows)


def macro_policy_rate_variables():
    """Return IRFVariable objects for the extended macro verification panel."""
    from src.irf_analysis import IRFVariable

    return tuple(IRFVariable(name, path, transform) for name, path, transform in MACRO_POLICY_RATE_VARIABLES)
