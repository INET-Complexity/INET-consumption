"""Scale the opening household balance sheet by household-specific income factors.

Each household's stocks are multiplied by k_i = y_model,i / y_HFCS,i, both on the
same income concept, so every stock/income ratio equals its HFCS value at t0.
National-accounts stock levels are then no longer matched; their gaps are reported.

Steps (spec: vault ``experiments/specs/2026-11-24-balance-sheet-scaling``):
1. k_i from t0 income, computed once.
2. Deposits, IFA (survey level, without the national-accounts anchor), debt, other
   real assets, house values and imputed rents are multiplied by k_i. Rented dwellings:
   value by the landlord's k_i, rent unchanged (it stays anchored to actual rent).
3. Financial income = annual yield x IFA.
4. Banks and debt installments are built afterwards from these stocks.
"""

from typing import Optional

import numpy as np
import pandas as pd

# Income concepts: name of the HFCS calibration column -> model-side construction.
SCALING_INCOME_CONCEPTS = ("disp_geom_avg_income", "disposable_income", "disp_non_property_income")

HOUSEHOLD_STOCK_COLUMNS = (
    "Wealth in Deposits",
    "Wealth in Other Financial Assets",
    "Outstanding Balance of HMR Mortgages",
    "Outstanding Balance of Mortgages on other Properties",
    "Outstanding Balance of other Non-Mortgage Loans",
    "Value of the Main Residence",
    "Value of other Properties",
    "Wealth Other Real Assets",
    "Rent Imputed",
)


def hfcs_income_column(concept: str) -> str:
    """Household-data column holding the HFCS source income for ``concept`` (annual, per household)."""
    return f"HFCS Income {concept}"


def model_income(household_data: pd.DataFrame, concept: str, yearly_factor: float) -> np.ndarray:
    """Annual model income on ``concept``. Non-property income = employee income + social transfers."""
    if concept not in SCALING_INCOME_CONCEPTS:
        raise ValueError(f"Unknown scaling income concept {concept!r}.")
    total = household_data["Income"].to_numpy(dtype=float)
    non_property = (household_data["Employee Income"] + household_data["Regular Social Transfers"]).to_numpy(
        dtype=float
    )
    if concept == "disposable_income":
        income = total
    elif concept == "disp_non_property_income":
        income = non_property
    else:
        both_nonnegative = (total >= 0.0) & (non_property >= 0.0)
        income = np.where(both_nonnegative, np.sqrt(np.clip(total * non_property, 0.0, None)), np.nan)
    return income * yearly_factor


def household_income_factors(model_annual: np.ndarray, hfcs_annual_scaled: np.ndarray) -> tuple[np.ndarray, int]:
    """k_i = model / HFCS income. Households with non-positive income on either side get the common factor."""
    model_annual = np.asarray(model_annual, dtype=float)
    hfcs_annual_scaled = np.asarray(hfcs_annual_scaled, dtype=float)
    valid = np.isfinite(model_annual) & np.isfinite(hfcs_annual_scaled) & (model_annual > 0) & (hfcs_annual_scaled > 0)
    if not valid.any():
        raise ValueError("No household has positive model and HFCS income on the scaling concept.")
    common = model_annual[valid].sum() / hfcs_annual_scaled[valid].sum()
    factors = np.full(model_annual.shape, common)
    factors[valid] = model_annual[valid] / hfcs_annual_scaled[valid]
    return factors, int((~valid).sum())


def scale_housing_market(housing_df: pd.DataFrame, factors: np.ndarray) -> pd.DataFrame:
    """Values by the owner's k_i; imputed rents of owner-occupied dwellings by the owner's k_i.

    Rent of rented dwellings is unchanged. Dwellings without a household owner (social housing) are unchanged.
    """
    owners = pd.to_numeric(housing_df["Corresponding Owner Household ID"], errors="coerce").to_numpy()
    has_owner = np.isfinite(owners) & (owners >= 0) & (owners < len(factors))
    owner_factor = np.ones(len(housing_df))
    owner_factor[has_owner] = factors[owners[has_owner].astype(int)]
    housing_df = housing_df.copy()
    housing_df["Value"] = housing_df["Value"].to_numpy(dtype=float) * owner_factor
    owner_occupied = housing_df["Is Owner-Occupied"].to_numpy(dtype=bool)
    rent = housing_df["Rent"].to_numpy(dtype=float)
    housing_df["Rent"] = np.where(owner_occupied, rent * owner_factor, rent)
    return housing_df


def is_scaled(population) -> bool:
    """True once the stocks carry a scaling factor (e.g. on a cached population being reset)."""
    return "Balance Sheet Scaling Factor" in population.household_data


def apply_financial_asset_yield(population, financial_asset_yield: float, *, financial_income_tax: float) -> None:
    """Step 3: financial income = annual yield x IFA, replacing the D4 anchor; recompute total income."""
    data = population.household_data
    population.coefficient_fa_income = financial_asset_yield / population.yearly_factor
    data["Income from Financial Assets"] = population.coefficient_fa_income * data["Wealth in Other Financial Assets"]
    population.set_household_income(financial_income_tax=financial_income_tax)


def scale_household_balance_sheet(
    population,
    housing_df: Optional[pd.DataFrame],
    *,
    concept: str,
    financial_asset_yield: float,
    financial_income_tax: float,
) -> tuple[Optional[pd.DataFrame], dict]:
    """Apply steps 1-3 in place on ``population.household_data``; return the scaled housing table and diagnostics."""
    if is_scaled(population):
        raise ValueError("Household balance sheet is already scaled; re-apply only the financial-asset yield.")
    data = population.household_data
    column = hfcs_income_column(concept)
    if column not in data:
        raise ValueError(f"Balance-sheet scaling needs {column!r}; export it in the HFCS source table.")
    if not financial_asset_yield > 0.0:
        raise ValueError("Balance-sheet scaling needs a positive annual financial_asset_yield.")

    # Step 1. HFCS income is per survey household; model stocks and flows represent `scale` households.
    model_annual = model_income(data, concept, population.yearly_factor)
    hfcs_annual = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float) * population.scale
    factors, n_fallback = household_income_factors(model_annual, hfcs_annual)

    # Step 2. IFA restarts from the survey level, i.e. without the national-accounts anchor.
    data["Wealth in Other Financial Assets"] /= getattr(population, "ifa_anchor_factor", 1.0)
    for name in HOUSEHOLD_STOCK_COLUMNS:
        data[name] = data[name].to_numpy(dtype=float) * factors
    population.set_household_total_real_assets()
    population.set_household_financial_assets()
    population.set_household_wealth()
    population.set_household_debt()
    population.set_household_net_wealth()
    scaled_housing = None if housing_df is None else scale_housing_market(housing_df, factors)

    data["Balance Sheet Scaling Factor"] = factors
    apply_financial_asset_yield(population, financial_asset_yield, financial_income_tax=financial_income_tax)

    diagnostics = {
        "concept": concept,
        "common_factor": float(np.nansum(model_annual) / np.nansum(hfcs_annual)),
        "factor_p10_p50_p90": [float(v) for v in np.percentile(factors, [10, 50, 90])],
        "n_common_factor_fallback": n_fallback,
        "financial_asset_yield": financial_asset_yield,
    }
    return scaled_housing, diagnostics
