"""Hand-to-mouth household classes for MPC reporting.

Established (Kaplan, Violante and Weidner 2014, KVW): a household is
hand-to-mouth (HtM) when net liquid wealth m sits at a kink of the budget set,

    0 <= m <= tau                 (zero-liquidity kink), or
    m < 0 and m <= -q + tau       (credit-limit kink),

with tau = half a pay period's income and q the borrowing limit. HtM households
with net illiquid wealth a <= 0 are poor HtM (P-HtM), the rest wealthy HtM
(here R-HtM).

Adaptations to the model panel (proposed, not from KVW):
- m = liquid financial assets - consumer-loan debt (mortgages are illiquid debt).
- a = illiquid financial assets + housing wealth - mortgage debt.
- Pay is monthly in France: tau = (quarterly income / 3) / 2.
- No credit limit is observed (HFCS France has none either). Proxy B: q = 0 for
  households without consumer credit, otherwise one month of income.
- Doubly wealthy (DW): non-HtM with m and a both in the top two quintiles
  (within seed). Everyone else is "Other". Classes are mutually exclusive.

Classes are ex-ante balance-sheet bins measured in the baseline run before the
shock; they do not determine MPCs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HTM_CLASSES = ("P-HtM", "R-HtM", "DW", "Other")

# Paper classification (consumption-notebooks/src/consumption_calibration.py, group_label_map;
# paper table tab:household_groups): H/L on NLA/y, IFA/y, HA/y, H = strictly above the median.
GROUP_LABELS = {
    "LLL": "Poor hand-to-mouth",
    "LLH": "Housing-rich hand-to-mouth",
    "LHL": "Financially wealthy hand-to-mouth",
    "LHH": "Doubly wealthy hand-to-mouth",
    "HLL": "Liquid-wealth households",
    "HLH": "Liquid homeowners",
    "HHL": "Financially wealthy unconstrained",
    "HHH": "Fully diversified wealthy",
}
# Collapse for the main figure (KVW logic: low liquidity = hand-to-mouth).
# HLL (liquid wealth only) has no illiquid wealth; it is grouped with DW as non-hand-to-mouth.
THREE_GROUPS = ("Poor hand-to-mouth", "Wealthy hand-to-mouth", "Doubly wealthy")
_THREE_GROUP_OF_CODE = {
    "LLL": THREE_GROUPS[0],
    "LLH": THREE_GROUPS[1],
    "LHL": THREE_GROUPS[1],
    "LHH": THREE_GROUPS[1],
    "HLL": THREE_GROUPS[2],
    "HLH": THREE_GROUPS[2],
    "HHL": THREE_GROUPS[2],
    "HHH": THREE_GROUPS[2],
}


def classify_balance_sheet_groups(panel: pd.DataFrame, *, group_column: str = "seed") -> pd.DataFrame:
    """Return the paper's 8-group code/label and the 3-group collapse per row.

    Ratios use the household's pre-shock income: NLA = liquid financial assets
    minus all debt (consumer and mortgage), IFA = illiquid financial assets,
    HA = housing wealth. Medians are taken within seed over households with
    positive income, as in the calibration notebook (unweighted: model households
    carry equal weight). A median split is invariant to the income period, so
    quarterly model income gives the same groups as annual income. Rows with
    non-positive or missing income are left unclassified (NaN).
    """
    income = panel["income"].astype(float)
    valid = income.gt(0) & np.isfinite(income)
    nla = (panel["liquid_financial_assets"] - panel["consumption_loan_debt"] - panel["mortgage_debt"]) / income
    ratios = {
        "nla_y": nla,
        "ifa_y": panel["illiquid_financial_assets"] / income,
        "ha_y": panel["housing_wealth"] / income,
    }
    groups = panel[group_column] if group_column in panel else pd.Series(0, index=panel.index)
    flags = []
    for values in ratios.values():
        values = values.where(valid & np.isfinite(values))
        median = values.groupby(groups).transform("median")
        flags.append(np.where(values > median, "H", "L"))
    code = pd.Series(np.char.add(np.char.add(flags[0], flags[1]), flags[2]), index=panel.index).astype(object)
    known = valid & pd.concat(ratios, axis=1).apply(np.isfinite).all(axis=1)
    code[~known] = np.nan
    return pd.DataFrame(
        {
            "group_code": code,
            "group_label": code.map(GROUP_LABELS),
            "group_3": code.map(_THREE_GROUP_OF_CODE),
            **{name: values.where(known) for name, values in ratios.items()},
        },
        index=panel.index,
    )


def classify_households(
    panel: pd.DataFrame,
    *,
    pay_periods_per_quarter: float = 3.0,
    credit_limit_months: float = 1.0,
    top_quintiles: int = 2,
    group_column: str = "seed",
) -> pd.Series:
    """Return the HtM class per row (NaN where an input is missing)."""
    income = panel["income"].to_numpy(dtype=float)
    m = (panel["liquid_financial_assets"] - panel["consumption_loan_debt"]).to_numpy(dtype=float)
    a = (panel["illiquid_financial_assets"] + panel["housing_wealth"] - panel["mortgage_debt"]).to_numpy(dtype=float)
    pay = np.maximum(income, 0.0) / pay_periods_per_quarter
    tau = pay / 2.0
    has_credit = panel["consumption_loan_debt"].to_numpy(dtype=float) > 0.0
    q = np.where(has_credit, credit_limit_months * np.maximum(income, 0.0) / 3.0, 0.0)
    htm = ((m >= 0.0) & (m <= tau)) | ((m < 0.0) & (m <= -q + tau))

    cutoff = 1.0 - top_quintiles / 5.0
    groups = panel[group_column] if group_column in panel else pd.Series(0, index=panel.index)
    m_s = pd.Series(m, index=panel.index)
    a_s = pd.Series(a, index=panel.index)
    m_cut = m_s.groupby(groups).transform(lambda s: s.quantile(cutoff)).to_numpy()
    a_cut = a_s.groupby(groups).transform(lambda s: s.quantile(cutoff)).to_numpy()
    dw = ~htm & (m >= m_cut) & (a >= a_cut)

    labels = np.select([htm & (a <= 0.0), htm, dw], ["P-HtM", "R-HtM", "DW"], default="Other").astype(object)
    missing = ~(np.isfinite(income) & np.isfinite(m) & np.isfinite(a))
    labels[missing] = np.nan
    return pd.Series(labels, index=panel.index, name="htm_class")
