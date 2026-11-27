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


# Liquidity-constraint groups (classify_liquidity_groups). HtM = liquidity constrained in the model's own
# mechanics; poor/wealthy split as in KVW; the three Crawley-Kuchler types as summarised by Muellbauer (2020,
# pp. 538-539). Warm = constrained, cool = unconstrained.
LIQUIDITY_GROUP_LABELS = {
    "PHTM": "Poor hand-to-mouth",
    "WHTM": "Wealthy hand-to-mouth",
    "APU": "Asset-poor, unconstrained",
    "ILU": "Illiquid-wealthy, unconstrained",
    "DW": "Doubly wealthy",
}
LIQUIDITY_GROUP_SHORT = {
    "PHTM": "Poor\nHtM",
    "WHTM": "Wealthy\nHtM",
    "APU": "Asset-poor\nunconstrained",
    "ILU": "Illiquid-wealthy\nunconstrained",
    "DW": "Doubly\nwealthy",
}
LIQUIDITY_GROUP_COLORS = {
    "PHTM": "#8c3b1f",
    "WHTM": "#c0602b",
    "APU": "#9fbcd6",
    "ILU": "#6f97bd",
    "DW": "#1f3b5c",
}
LIQUIDITY_DSTI_LIMIT = 0.35  # consumer_credit.dsti_limit (consumption_paper_parameters.yaml)
_CONSUMER_PAYMENT_COLUMN = "diagnostic_scheduled_consumer_payment_t0"
_MORTGAGE_PAYMENT_COLUMNS = ("diagnostic_actual_mortgage_payment_t0", "diagnostic_unpaid_mortgage_payment_t0")


def consumer_credit_headroom(
    panel: pd.DataFrame,
    *,
    dsti_limit: float = LIQUIDITY_DSTI_LIMIT,
    consumer_annuity_factor: float | None = None,
    group_column: str = "seed",
) -> pd.Series:
    """Additional consumer loan the DSTI cap still permits, per household (pre-shock quarter).

    Mirrors CreditMarket._cap_consumer_origination_dsti and the Stage 5 fallback:

        K = max( max(dsti * y - S_m, 0) / phi - D_c, 0 ),

    with y quarterly income, S_m the scheduled mortgage service (paid + unpaid, t0), D_c the consumer
    balance and phi the consumer annuity payment factor. ``consumer_annuity_factor=None`` estimates phi
    within seed as the median of scheduled consumer service / consumer balance over borrowers.
    Approximations: y is the panel's income, not the gross cash underwriting income (K is understated
    when gross exceeds net); overdraft interest in "other service" and the FICP block are omitted.
    """
    income = panel["income"].astype(float)
    mortgage_service = sum(panel[column].astype(float) for column in _MORTGAGE_PAYMENT_COLUMNS)
    balance = panel["consumption_loan_debt"].astype(float)
    if consumer_annuity_factor is None:
        if _CONSUMER_PAYMENT_COLUMN not in panel:
            raise ValueError(f"Pass consumer_annuity_factor or provide {_CONSUMER_PAYMENT_COLUMN}.")
        payment = panel[_CONSUMER_PAYMENT_COLUMN].astype(float)
        groups = panel[group_column] if group_column in panel else pd.Series(0, index=panel.index)
        borrowers = (balance > 0.0) & (payment > 0.0)
        ratio = (payment / balance).where(borrowers)
        factor = ratio.groupby(groups).transform("median")
        if factor.isna().any():
            raise ValueError("Cannot estimate the consumer annuity factor: a seed has no serviced consumer loans.")
    else:
        if not consumer_annuity_factor > 0.0:
            raise ValueError("consumer_annuity_factor must be positive.")
        factor = pd.Series(float(consumer_annuity_factor), index=panel.index)
    service_room = np.maximum(dsti_limit * income - mortgage_service, 0.0)
    return np.maximum(service_room / factor - balance, 0.0).rename("credit_headroom")


def classify_liquidity_groups(
    panel: pd.DataFrame,
    *,
    cut_column: str = "diagnostic_consumption_cut_amount_t0",
    pay_periods_per_quarter: float = 3.0,
    dsti_limit: float = LIQUIDITY_DSTI_LIMIT,
    consumer_annuity_factor: float | None = None,
    group_column: str = "seed",
) -> pd.DataFrame:
    """Return the liquidity-constraint group per row, measured before the shock.

    All inputs are the state at ``pre_shock_row`` (end of the quarter before the transfer), common
    to the baseline and shock arms, so no outcome of the shock enters. With quarterly income y,

        tau = y / pay_periods_per_quarter / 2            (KVW: half a monthly pay period)
        K   = remaining consumer credit under the DSTI cap  (consumer_credit_headroom)
        constrained = consumption cut binding  or  LFA + K <= tau
        a   = illiquid financial assets + housing wealth - mortgage debt   (KVW net illiquid wealth)

    A cut binds when ``cut_column`` exceeds the MPC-regime tolerance. LFA is negative in overdraft;
    an overdraft is repaid before goods and cannot be refinanced with new credit, so it reduces the
    liquidity available this quarter. Groups (mutually exclusive):

        PHTM  constrained, a <= 0
        WHTM  constrained, a > 0
        APU   unconstrained, a <= 0
        ILU   unconstrained, a > 0, LFA <= tau   (no liquid buffer, smooths with credit)
        DW    unconstrained, a > 0, LFA > tau    (liquid and illiquid wealth)

    Rows with non-positive or missing income or inputs are NaN.
    """
    from src.mpc_analysis import MPC_REGIME_RELATIVE_TOLERANCE

    income = panel["income"].astype(float)
    lfa = panel["liquid_financial_assets"].astype(float)
    a = (panel["illiquid_financial_assets"] + panel["housing_wealth"] - panel["mortgage_debt"]).astype(float)
    tau = income / pay_periods_per_quarter / 2.0
    headroom = consumer_credit_headroom(
        panel, dsti_limit=dsti_limit, consumer_annuity_factor=consumer_annuity_factor, group_column=group_column
    )
    floor = panel["baseline_subsistence_floor"].abs() if "baseline_subsistence_floor" in panel else 0.0
    cut = panel[cut_column].astype(float)
    cut_binding = cut > MPC_REGIME_RELATIVE_TOLERANCE * np.maximum(floor, 1.0)
    available = lfa + headroom
    constrained = cut_binding | (available <= tau)
    code = pd.Series(
        np.select(
            [constrained & (a <= 0.0), constrained, a <= 0.0, lfa <= tau],
            ["PHTM", "WHTM", "APU", "ILU"],
            default="DW",
        ),
        index=panel.index,
        dtype=object,
    )
    inputs = pd.concat([income, lfa, a, headroom, cut], axis=1)
    known = income.gt(0) & np.isfinite(inputs).all(axis=1)
    code[~known] = np.nan
    return pd.DataFrame(
        {
            "liquidity_group": code,
            "liquidity_group_label": code.map(LIQUIDITY_GROUP_LABELS),
            "liquidity_constrained": constrained.where(known),
            "consumption_cut_binding": cut_binding.where(known),
            "credit_headroom": headroom.where(known),
            "available_liquidity": available.where(known),
            "liquidity_threshold": tau.where(known),
            "net_illiquid_wealth": a.where(known),
        },
        index=panel.index,
    )
