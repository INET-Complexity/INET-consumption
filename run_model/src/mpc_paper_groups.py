"""Shared paper grouping definitions for figures and tabular estimates."""

import numpy as np
import pandas as pd


def prepare_paper_groups(panel):
    """Group the displayed sample once; exclude arrears-only cases from consumption constraints."""
    from src.mpc_analysis import (
        MPC_REGIME_RELATIVE_TOLERANCE,
        assign_quantile_bin,
        reconcile_owner_mortgage_tenure,
    )

    panel = panel.copy()
    specs = []
    for field, title in [
        ("income", "Household income"),
        ("labor_income", "Labour income"),
        ("non_property_income", "Non-property income"),
        ("property_income", "Property income"),
        ("liquid_financial_assets", "Liquid financial assets"),
        ("net_liquid_financial_assets_all_debt", "Net liquid assets"),
        ("illiquid_financial_assets", "Illiquid financial assets"),
        ("housing_wealth", "Housing wealth"),
    ]:
        if field not in panel:
            continue
        group = field + "_paper_group"
        panel[group] = pd.Series(pd.NA, index=panel.index, dtype="string")
        if field in (
            "illiquid_financial_assets",
            "housing_wealth",
            "labor_income",
            "non_property_income",
            "property_income",
        ):
            positive = panel[field] > 0
            panel.loc[panel[field] == 0, group] = "Zero"
            panel.loc[panel[field] < 0, group] = "Negative"
            panel.loc[positive, group] = assign_quantile_bin(panel.loc[positive, field]).astype("string")
        else:
            panel[group] = assign_quantile_bin(panel[field]).astype("string")
        specs.append((group, title, ["Negative", "Zero", "Q1", "Q2", "Q3", "Q4", "Q5", "all"]))
    if "housing_tenure" in panel:
        owner_rows = panel.housing_tenure.isin(["owner_outright", "owner_mortgage"])
        if owner_rows.any():
            if "mortgage_debt" not in panel:
                raise ValueError("Housing tenure requires mortgage_debt; rebuild the panel from saved metadata.")
            panel["housing_tenure"] = reconcile_owner_mortgage_tenure(panel.housing_tenure, panel.mortgage_debt)
        panel["tenure"] = panel.housing_tenure.map(
            {
                "renter": "Renters",
                "owner_mortgage": "Mortgagors",
                "owner_outright": "Outright\nowners",
                "owner_other": "Other tenure",
                "unknown": "Other tenure",
            }
        ).fillna("Other tenure")
        specs.append(("tenure", "Housing tenure", ["Renters", "Mortgagors", "Outright\nowners", "Other tenure"]))
    if "activity_bracket_t0" in panel:
        panel["labour"] = panel.activity_bracket_t0.map({1: "Employed", 2: "Unemployed", 3: "Inactive", 4: "Retired"})
        specs.append(("labour", "Labour-market status", ["Employed", "Unemployed", "Inactive", "Retired"]))
    if {"consumption_loan_debt", "mortgage_debt"}.issubset(panel.columns):
        consumer = panel.consumption_loan_debt > 1e-8
        mortgage = panel.mortgage_debt > 1e-8
        panel["borrowing"] = np.select(
            [consumer & mortgage, consumer, mortgage],
            ["Both", "Consumer\ncredit", "Mortgage\nonly"],
            default="No borrowing",
        )
        specs.append(("borrowing", "Borrowing", ["No borrowing", "Consumer\ncredit", "Mortgage\nonly", "Both"]))
    if "mpc_regime" in panel:
        panel["constraint"] = pd.Series(pd.NA, index=panel.index, dtype="string")
        panel.loc[panel.mpc_regime.eq("unconstrained"), "constraint"] = "Unconstrained"
        candidates = panel.mpc_regime.eq("liquidity_constrained")
        if candidates.any():
            cut_column = next(
                (
                    name
                    for name in ("baseline_consumption_cut_amount", "diagnostic_consumption_cut_amount_t1")
                    if name in panel
                ),
                None,
            )
            if cut_column is None or "baseline_subsistence_floor" not in panel:
                raise ValueError(
                    "Consumption-constraint groups require baseline consumption cuts and floor; rebuild the panel from saved baseline windows."
                )
            tolerance = MPC_REGIME_RELATIVE_TOLERANCE * np.maximum(panel.baseline_subsistence_floor.abs(), 1.0)
            binds = panel[cut_column] > tolerance
            panel.loc[candidates & binds, "constraint"] = "Constrained"
        specs.append(("constraint", "Consumption liquidity constraint", ["Constrained", "Unconstrained"]))
    return panel, specs
