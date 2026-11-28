"""Calibration support of the HFCS consumption source, built from the HFCS microdata the model reads.

The consumption calibration estimates on households that survive a joint percentile trim of
c_y, ifa_y, nla_y, ha_y and r_y (``consumption_calibration.prepare_household_frame``). The default
sampler draws every hard-valid record instead. This module rebuilds the four stock/income variables
from the survey records already in the model, computes the same weighted p1/p99 thresholds on the
hard-valid pool of the wave being initialised, and flags the records inside all five thresholds.
Nothing is read from outside the model except the registered c_y (whose own derivation is not
rebuilt here). No country constants appear: taxes, the imputed-rent share and the pool come from the
country's own data.

Definitions (per HFCS record, implicate as read by the model, i.e. the first):
    Y    = disp_geom_avg_income - DI1412 + imputed rent of the main residence
    lfa  = DA2101 + HD1320c;  ifa = DA2102+DA2103+DA2105+DA2106+DA2107+DA2108+DA2109 - HD1320c
    ha   = DA1110;            debt = DL1100 + DL1200
    ifa_y, ha_y = ifa/Y, ha/Y;  nla_y = (lfa - debt)/Y;  r_y = (lfa - DO)/Y
    DO   = annual debt service, repaired as in the calibration's ``construct_debt_service``.
The calibration's deflator cancels in every ratio (numerator and Y are divided by the same price index)
and enters only through a validity check, so no price input is needed.

Deviations from the calibration, all caused by inputs the model does not hold:
    * disp_geom_avg_income uses the model's runtime tax rates (``hfcs_net_income``), not the
      calibration's tax simulator, and omits DI1700/DI1800 (private transfers, other income).
    * Imputed rent keeps the calibration's rule (rate x DA1110, total pinned to a share of non-durable
      spending) but the share is imputed rent / household final consumption from the country's
      input-output table. The calibration uses imputed rent / (consumption - durables) from national
      accounts, so durables are not netted. Missing HI0220 is not imputed (those households leave the
      imputed-rent totals; the calibration imputes it by PMM).
    * The model reads implicate 1; the calibration takes the median over the five implicates.
"""

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
import pandas as pd

from macro_data.processing.synthetic_population.balance_sheet_scaling import (
    HOUSEHOLD_STOCK_COLUMNS,
    hfcs_net_income,
)
from macro_data.util.hfcs_consumption_source import HFCSConsumptionSource

# Calibration hard bounds (``calibration_bounds`` of the calibration settings), applied before trimming.
CALIBRATION_BOUNDS = {
    "c_y": (0.0, 3.0),
    "nla_y": (-10.0, 10.0),
    "r_y": (-10.0, 10.0),
    "ifa_y": (0.0, 20.0),
    "ha_y": (0.0, 30.0),
}
TRIM_VARIABLES = ("c_y", "ifa_y", "nla_y", "ha_y", "r_y")
SUPPORT_INPUT_COLUMNS = (
    "Interest Payments",
    "Mortgage Payments (flow)",
    "HMR Mortgage Payments (flow)",
    "Monthly Leasing Payments",
    "Money Market Funds",
)
IFA_COMPONENTS = (
    "Mutual Funds",
    "Bonds",
    "Shares",
    "Managed Accounts",
    "Money owed to Households",
    "Other Assets",
    "Voluntary Pension",
)


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """Survey-weighted quantile, linear interpolation on cumulative-weight midpoints (calibration rule)."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    order = np.argsort(x)
    x_sorted, w_sorted = x[order], w[order]
    cum_w = np.cumsum(w_sorted) - 0.5 * w_sorted
    cum_w /= w_sorted.sum()
    return float(np.interp(q, cum_w, x_sorted))


def _weighted_median_step(x: pd.Series, w: pd.Series) -> float:
    """Weighted median without interpolation, as in the calibration's debt-service repairs."""
    if x.empty:
        return float("nan")
    order = np.argsort(x.to_numpy())
    xs, ws = x.to_numpy()[order], w.to_numpy()[order]
    cum = np.cumsum(ws)
    return float(xs[np.searchsorted(cum, 0.5 * cum[-1])])


def _column(households: pd.DataFrame, name: str) -> pd.Series:
    if name not in households:
        raise ValueError(f"Calibration support needs the HFCS column {name!r}.")
    return pd.to_numeric(households[name], errors="coerce")


def _zero_filled(households: pd.DataFrame, name: str) -> pd.Series:
    return _column(households, name).fillna(0.0)


def hfcs_imputed_rent(households: pd.DataFrame, imputed_rent_share: float) -> pd.Series:
    """Imputed rent of the main residence: a common rate per euro of DA1110.

    The weighted total is ``s / (1 - s)`` times the weighted survey spending on goods and actual rent
    (12 x (HI0220 + HB2300 + HB0410)), over households with observed HI0220.
    """
    if not 0.0 < imputed_rent_share < 1.0:
        raise ValueError("The imputed-rent share of household consumption must lie in (0, 1).")
    weight = _column(households, "Weight")
    goods = 12.0 * _column(households, "Amount spent on Consumption of Goods and Services")
    rent = 12.0 * (
        _zero_filled(households, "Rent Paid") + _zero_filled(households, "Rent Paid for Partially Owned Dwelling")
    )
    non_durables_less_imputed = goods + rent
    observed = non_durables_less_imputed.notna()
    main_residence = _zero_filled(households, "Value of the Main Residence")
    main_total = float((weight * main_residence)[observed].sum())
    total = (
        imputed_rent_share
        / (1.0 - imputed_rent_share)
        * float((weight * non_durables_less_imputed.fillna(0.0))[observed].sum())
    )
    rate = total / main_total if main_total > 0 else 0.0
    return rate * main_residence


def debt_service(households: pd.DataFrame, total_debt_mortgage: pd.Series, nonmortgage_debt: pd.Series) -> pd.DataFrame:
    """Annual debt service DO, with the calibration's repairs (``src/debt_service.py``).

    Mortgage: 12 x max(DL2100, DL2110); where the monthly payment is at least the balance (principal keyed as
    payment) it is replaced by balance x the weighted median annual-payment/balance of plausible payers.
    Non-mortgage: 12 x (HC0110 + HC1200); not-reported (zero) payments on positive debt are replaced by debt x the
    weighted median annual-payment/debt of households reporting both. Medians are over the whole survey wave.
    """
    weight = _column(households, "Weight")
    monthly_mortgage = pd.concat(
        [_column(households, "Mortgage Payments (flow)"), _column(households, "HMR Mortgage Payments (flow)")], axis=1
    ).max(axis=1)
    balance = total_debt_mortgage
    principal_as_payment = monthly_mortgage.gt(0) & monthly_mortgage.ge(balance)
    plausible = monthly_mortgage.gt(0) & balance.gt(0) & ~principal_as_payment & weight.gt(0)
    mortgage_rate = _weighted_median_step((12.0 * monthly_mortgage / balance)[plausible], weight[plausible])
    mortgage_service = (12.0 * monthly_mortgage).where(~principal_as_payment, mortgage_rate * balance).fillna(0.0)

    monthly_nonmortgage = pd.concat(
        [_column(households, "Monthly Leasing Payments"), _column(households, "Additional Consumer Loan Payments")],
        axis=1,
    ).sum(axis=1, skipna=True)
    debt = nonmortgage_debt.fillna(0.0)
    reported = monthly_nonmortgage.gt(0) & debt.gt(0) & weight.gt(0)
    nonmortgage_rate = _weighted_median_step((12.0 * monthly_nonmortgage / debt)[reported], weight[reported])
    imputed = monthly_nonmortgage.eq(0) & debt.gt(0)
    nonmortgage_service = (12.0 * monthly_nonmortgage).where(~imputed, nonmortgage_rate * debt)
    return pd.DataFrame(
        {
            "debt_service": mortgage_service + nonmortgage_service,
            "mortgage_service_exceeds_balance": mortgage_service.gt(balance.fillna(0.0)),
        },
        index=households.index,
    )


def _survey_gross_income_frame(households: pd.DataFrame) -> pd.DataFrame:
    """The six gross income columns ``hfcs_net_income`` expects, from the reader's names."""
    names = {
        "HFCS Gross Employee Income": "HFCS Gross Employee Income",
        "HFCS Gross Self-Employment Income": "HFCS Gross Self-Employment Income",
        "HFCS Gross Pension Income": "Income from Pensions",
        "HFCS Gross Social Transfers": "HFCS Gross Social Transfers",
        "HFCS Gross Rental Income": "Rental Income from Real Estate",
        "HFCS Gross Financial Income": "Income from Financial Assets",
    }
    return pd.DataFrame(
        {target: _column(households, source) for target, source in names.items()}, index=households.index
    )


def survey_net_income(
    households: pd.DataFrame, concept: str, *, income_tax: float, employee_social_insurance_tax: float
):
    """Annual HFCS net income per survey household on the balance-sheet scaling concept."""
    return hfcs_net_income(
        _survey_gross_income_frame(households),
        concept,
        income_tax=income_tax,
        employee_social_insurance_tax=employee_social_insurance_tax,
    )


def build_support_variables(
    households: pd.DataFrame, *, income_tax: float, employee_social_insurance_tax: float, imputed_rent_share: float
) -> pd.DataFrame:
    """ifa_y, nla_y, ha_y, r_y, income and the model-side validity flag, indexed like ``households``."""
    missing = [c for c in SUPPORT_INPUT_COLUMNS if c not in households]
    if missing:
        raise ValueError(f"Calibration support needs HFCS columns read with support_inputs=True: {missing}")
    weight = _column(households, "Weight")
    disp_geom = pd.Series(
        survey_net_income(
            households,
            "disp_geom_avg_income",
            income_tax=income_tax,
            employee_social_insurance_tax=employee_social_insurance_tax,
        ),
        index=households.index,
    )
    income = (
        disp_geom - _zero_filled(households, "Interest Payments") + hfcs_imputed_rent(households, imputed_rent_share)
    )

    money_market = _zero_filled(households, "Money Market Funds")
    lfa = _zero_filled(households, "Wealth in Deposits") + money_market
    ifa = pd.concat([_zero_filled(households, c) for c in IFA_COMPONENTS], axis=1).sum(axis=1) - money_market
    ha = _zero_filled(households, "Value of the Main Residence")
    mortgage_debt = _column(households, "Outstanding Balance of Mortgage Debt")
    nonmortgage_debt = _column(households, "Outstanding Balance of Non-Mortgage Debt")
    total_debt = pd.concat([mortgage_debt, nonmortgage_debt], axis=1).sum(axis=1, skipna=True)
    service = debt_service(households, mortgage_debt, nonmortgage_debt)

    out = pd.DataFrame(index=households.index)
    out["income"] = income
    out["ifa_y"] = ifa / income
    out["nla_y"] = (lfa - total_debt) / income
    out["ha_y"] = ha / income
    out["r_y"] = lfa / income - service["debt_service"] / income
    valid = weight.notna() & np.isfinite(weight) & weight.gt(0) & disp_geom.gt(0) & income.gt(0)
    valid &= (
        np.isfinite(out[["ifa_y", "nla_y", "ha_y", "r_y"]]).all(axis=1) & ~service["mortgage_service_exceeds_balance"]
    )
    for name in ("ifa_y", "nla_y", "ha_y", "r_y"):
        low, high = CALIBRATION_BOUNDS[name]
        valid &= out[name].between(low, high, inclusive="both")
    out["valid"] = valid
    return out


@dataclass(frozen=True)
class CalibrationSupport:
    """Support flags aligned to the source rows, with the thresholds that define them."""

    support: np.ndarray
    hard_valid: np.ndarray
    thresholds: dict[str, tuple[float, float]]
    variables: pd.DataFrame


def calibration_support(
    source: HFCSConsumptionSource,
    variables: pd.DataFrame,
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> CalibrationSupport:
    """Joint percentile trim on the hard-valid pool of the source wave.

    ``variables`` holds ifa_y, nla_y, ha_y, r_y, ``valid`` aligned to ``source.source_ids``. The hard-valid pool is
    the registered hard-valid mask intersected with the model-side validity (finite ratios, hard bounds, debt-service
    check). Thresholds are survey-weighted quantiles on that pool restricted to the registered threshold pool. A record
    is in the support when it is hard-valid and every variable lies inside its thresholds.
    """
    c_y = source.c_y
    weights = source.survey_weights
    hard_valid = source.winsorised.valid & variables["valid"].to_numpy(dtype=bool) & np.isfinite(c_y)
    lo_c, hi_c = CALIBRATION_BOUNDS["c_y"]
    hard_valid &= (c_y >= lo_c) & (c_y <= hi_c)
    pool = hard_valid if source.threshold_pool is None else hard_valid & source.threshold_pool
    if not pool.any():
        raise ValueError("No hard-valid household remains in the threshold pool of the calibration support.")
    values = {"c_y": c_y, **{name: variables[name].to_numpy(dtype=float) for name in TRIM_VARIABLES if name != "c_y"}}
    thresholds = {}
    trimmed = np.zeros(len(weights), dtype=bool)
    for name in TRIM_VARIABLES:
        lower = weighted_quantile(values[name][pool], weights[pool], lower_quantile)
        upper = weighted_quantile(values[name][pool], weights[pool], upper_quantile)
        thresholds[name] = (lower, upper)
        with np.errstate(invalid="ignore"):
            trimmed |= (values[name] < lower) | (values[name] > upper)
    support = hard_valid & ~trimmed
    return CalibrationSupport(support, hard_valid, thresholds, variables)


def survey_stocks(households: pd.DataFrame) -> dict[str, np.ndarray]:
    """Survey-level value of every stock ``balance_sheet_scaling`` multiplies by k_i, with the model's definitions."""
    components = pd.concat([_zero_filled(households, c).clip(lower=0.0) for c in IFA_COMPONENTS], axis=1).sum(axis=1)
    other_real = (
        _zero_filled(households, "Value of Household Vehicles")
        + _zero_filled(households, "Value of Household Valuables")
        + _zero_filled(households, "Value of Self-Employment Businesses")
    )
    stocks = {
        "Wealth in Deposits": _zero_filled(households, "Wealth in Deposits"),
        "Wealth in Other Financial Assets": components,
        "Outstanding Balance of HMR Mortgages": _zero_filled(households, "Outstanding Balance of HMR Mortgages"),
        "Outstanding Balance of Mortgages on other Properties": _zero_filled(
            households, "Outstanding Balance of Mortgages on other Properties"
        ),
        "Outstanding Balance of other Non-Mortgage Loans": _zero_filled(
            households, "Outstanding Balance of other Non-Mortgage Loans"
        ),
        "Value of the Main Residence": _zero_filled(households, "Value of the Main Residence"),
        "Value of other Properties": _zero_filled(households, "Value of other Properties"),
        "Wealth Other Real Assets": other_real,
    }
    if set(stocks) != set(HOUSEHOLD_STOCK_COLUMNS):
        raise ValueError("Survey stock definitions are out of step with the balance-sheet scaling stock list.")
    return {name: stocks[name].to_numpy(dtype=float) for name in HOUSEHOLD_STOCK_COLUMNS}


def stock_anchor_factors(
    stocks: Mapping[str, np.ndarray],
    income: np.ndarray,
    weights: np.ndarray,
    all_valid: np.ndarray,
    support: np.ndarray,
) -> dict[str, float]:
    """k_S = (aggregate S/y on the default pool) / (aggregate S/y on the support), per stock.

    Aggregates are survey-weighted sums on records with finite positive income, so
    ``k_S x (sum_support w S / sum_support w y) = sum_all w S / sum_all w y`` holds exactly.
    ``all_valid`` is the pool the default sampler draws from; ``support`` must lie inside it.
    """
    income = np.asarray(income, dtype=float)
    weights = np.asarray(weights, dtype=float)
    usable = np.isfinite(income) & (income > 0)
    default_pool, support_pool = all_valid & usable, support & usable
    if not support_pool.any():
        raise ValueError("The calibration support has no household with positive income.")
    factors = {}
    for name, stock in stocks.items():
        default_ratio = (weights * stock)[default_pool].sum() / (weights * income)[default_pool].sum()
        support_ratio = (weights * stock)[support_pool].sum() / (weights * income)[support_pool].sum()
        if support_ratio == 0.0:
            if default_ratio != 0.0:
                raise ValueError(f"Cannot re-anchor {name!r}: the support holds none of it.")
            factors[name] = 1.0
        else:
            factors[name] = float(default_ratio / support_ratio)
    return factors


def imputed_rent_share_of_household_consumption(icio, exchange_rates, country, year: int) -> float:
    """Imputed rent over household final consumption (period flows, national currency), from the input-output table."""
    imputed = icio.imputed_rents.get(country)
    if imputed is None:
        raise ValueError(f"No imputed rent for {country}; the calibration support needs it.")
    actual = icio.actual_rents.get(country) or 0.0
    remaining = float(np.sum(icio.get_hh_consumption(country))) * exchange_rates.from_usd_to_lcu(country, year)
    return float(imputed / (remaining + imputed + actual))


def attach_calibration_support(
    source: HFCSConsumptionSource,
    households: pd.DataFrame,
    *,
    income_tax: float,
    employee_social_insurance_tax: float,
    imputed_rent_share: float,
    scaling_income_concept: str,
) -> HFCSConsumptionSource:
    """Return ``source`` with the support mask and the stock re-anchoring factors, built from ``households``.

    ``households`` are the survey records of the registered source (any order); they are matched by HFCS Source ID.
    """
    if "HFCS Source ID" not in households or "Weight" not in households:
        raise ValueError("Calibration support needs HFCS Source ID and Weight on the survey households.")
    ids = pd.Index(households["HFCS Source ID"].to_numpy())
    positions = ids.get_indexer(source.source_ids)
    if np.any(positions < 0) or ids.has_duplicates or len(ids) != len(source.source_ids):
        raise ValueError("Survey households differ from the registered source ID set.")
    variables = build_support_variables(
        households,
        income_tax=income_tax,
        employee_social_insurance_tax=employee_social_insurance_tax,
        imputed_rent_share=imputed_rent_share,
    ).iloc[positions]
    result = calibration_support(
        source, variables, lower_quantile=source.quantiles[0], upper_quantile=source.quantiles[1]
    )
    income = survey_net_income(
        households,
        scaling_income_concept,
        income_tax=income_tax,
        employee_social_insurance_tax=employee_social_insurance_tax,
    )[positions]
    stocks = {name: values[positions] for name, values in survey_stocks(households).items()}
    # The default sampler draws from the registered hard-valid mask; k_S restores that pool's aggregate S/y.
    factors = stock_anchor_factors(stocks, income, source.survey_weights, source.winsorised.valid, result.support)
    return replace(source, support=result.support, stock_anchor_factors=factors, support_thresholds=result.thresholds)


def support_inputs_from_readers(readers, country, year: int) -> dict:
    """Tax rates and imputed-rent share from the country's own readers (no country constants)."""
    return {
        "income_tax": readers.oecd_econ.read_tau_income(country, year),
        "employee_social_insurance_tax": readers.oecd_econ.read_tau_siw(country, year),
        "imputed_rent_share": imputed_rent_share_of_household_consumption(
            readers.icio[year], readers.exchange_rates, country, year
        ),
    }
