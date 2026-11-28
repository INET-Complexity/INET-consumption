"""Scale the opening household balance sheet by household-specific income factors.

Each household's stocks are multiplied by k_i = y_model,i / y_HFCS,i, both on the
same income concept, so every stock/income ratio equals its HFCS value at t0.
National-accounts stock levels are then no longer matched; their gaps are reported.

Steps (spec: vault ``experiments/specs/2026-11-24-balance-sheet-scaling``):
1. k_i from t0 income, computed once. HFCS net income = survey gross income netted with
   the model's runtime tax rates (no external tax simulator, no extra data file).
2. Survey stocks are multiplied by k_i: deposits, IFA (survey level, without the
   national-accounts anchor), debt, other real assets and house values (owner-occupied
   dwellings by the owner's k_i, rented dwellings by the landlord's k_i). Rents are flows
   anchored to the national accounts (imputed rent to CPA_L68A, actual rent to CPA_L68B)
   and are not scaled.
3. The rent/value floor is re-applied to the scaled values. It only raises rents; an owner's
   "Rent Imputed" stays equal to the rent of the dwelling it lives in.
4. Financial income = annual yield x IFA.
5. Banks and debt installments are built afterwards from these stocks.
"""

from typing import Mapping, Optional

import numpy as np
import pandas as pd

from macro_data.processing.synthetic_matching.matching_households_with_houses import apply_rent_yield_floor

# Income concepts, named as in the consumption calibration.
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
)


HFCS_GROSS_COLUMNS = (
    "HFCS Gross Employee Income",
    "HFCS Gross Self-Employment Income",
    "HFCS Gross Pension Income",
    "HFCS Gross Social Transfers",
    "HFCS Gross Rental Income",
    "HFCS Gross Financial Income",
)


def hfcs_net_income(
    household_data: pd.DataFrame, concept: str, *, income_tax: float, employee_social_insurance_tax: float
) -> np.ndarray:
    """Annual HFCS income per survey household on ``concept``, netted with the model's runtime tax rates.

    Labour income x (1 - employee SI)(1 - income tax); rental and financial income x (1 - income tax);
    pensions and social transfers untaxed, as in model income. Matches the calibration's
    ``disp_geom_avg_income`` to 1.008 on aggregate (household p10/p90 0.91/1.15; vault, 2026-11-24).
    """
    missing = [c for c in HFCS_GROSS_COLUMNS if c not in household_data]
    if missing:
        raise ValueError(f"Balance-sheet scaling needs HFCS gross income columns: {missing}")
    gross = {
        c: pd.to_numeric(household_data[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        for c in HFCS_GROSS_COLUMNS
    }
    labour = (
        (gross["HFCS Gross Employee Income"] + gross["HFCS Gross Self-Employment Income"])
        * (1.0 - employee_social_insurance_tax)
        * (1.0 - income_tax)
    )
    non_property = labour + gross["HFCS Gross Pension Income"] + gross["HFCS Gross Social Transfers"]
    total = non_property + (gross["HFCS Gross Rental Income"] + gross["HFCS Gross Financial Income"]) * (
        1.0 - income_tax
    )
    return _concept(concept, total, non_property)


def _concept(concept: str, total: np.ndarray, non_property: np.ndarray) -> np.ndarray:
    if concept == "disposable_income":
        return total
    if concept == "disp_non_property_income":
        return non_property
    if concept == "disp_geom_avg_income":
        both_nonnegative = (total >= 0.0) & (non_property >= 0.0)
        return np.where(both_nonnegative, np.sqrt(np.clip(total * non_property, 0.0, None)), np.nan)
    raise ValueError(f"Unknown scaling income concept {concept!r}.")


def model_income(household_data: pd.DataFrame, concept: str, yearly_factor: float) -> np.ndarray:
    """Annual model income on ``concept``. Non-property income = employee income + social transfers."""
    total = household_data["Income"].to_numpy(dtype=float)
    non_property = (household_data["Employee Income"] + household_data["Regular Social Transfers"]).to_numpy(
        dtype=float
    )
    return _concept(concept, total, non_property) * yearly_factor


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


def scale_housing_market(
    housing_df: pd.DataFrame, factors: np.ndarray, anchor: Optional[Mapping[str, float]] = None
) -> pd.DataFrame:
    """Dwelling values by the owner's k_i. Rents are flows anchored to the national accounts and stay unchanged.

    Dwellings without a household owner (social housing) are unchanged. ``anchor`` (opt-in calibration support)
    multiplies owner-occupied values by k_S of the main residence and rented dwellings by k_S of other properties.
    """
    owners = pd.to_numeric(housing_df["Corresponding Owner Household ID"], errors="coerce").to_numpy()
    has_owner = np.isfinite(owners) & (owners >= 0) & (owners < len(factors))
    owner_factor = np.ones(len(housing_df))
    owner_factor[has_owner] = factors[owners[has_owner].astype(int)]
    if anchor is not None:
        owner_occupied = housing_df["Is Owner-Occupied"].to_numpy(dtype=bool)
        owner_factor = owner_factor * np.where(
            has_owner,
            np.where(owner_occupied, anchor["Value of the Main Residence"], anchor["Value of other Properties"]),
            1.0,
        )
    housing_df = housing_df.copy()
    housing_df["Value"] = housing_df["Value"].to_numpy(dtype=float) * owner_factor
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
    employee_social_insurance_tax: float,
    minimum_annual_rental_yield: float = 0.0,
    rental_income_taxes: float = 0.0,
    stock_anchor_factors: Optional[Mapping[str, float]] = None,
) -> tuple[Optional[pd.DataFrame], dict]:
    """Apply steps 1-4 in place on ``population.household_data``; return the scaled housing table and diagnostics.

    ``minimum_annual_rental_yield`` and ``rental_income_taxes`` are those used when the housing table was built.
    ``stock_anchor_factors`` (opt-in calibration support) multiplies each stock by its k_S after the k_i scaling.
    """
    if is_scaled(population):
        raise ValueError("Household balance sheet is already scaled; re-apply only the financial-asset yield.")
    data = population.household_data
    if not financial_asset_yield > 0.0:
        raise ValueError("Balance-sheet scaling needs a positive annual financial_asset_yield.")

    # Step 1. HFCS income is per survey household; model stocks and flows represent `scale` households.
    model_annual = model_income(data, concept, population.yearly_factor)
    hfcs_annual = (
        hfcs_net_income(
            data,
            concept,
            income_tax=financial_income_tax,
            employee_social_insurance_tax=employee_social_insurance_tax,
        )
        * population.scale
    )
    factors, n_fallback = household_income_factors(model_annual, hfcs_annual)

    # Step 2. IFA restarts from the survey level, i.e. without the national-accounts anchor.
    data["Wealth in Other Financial Assets"] /= getattr(population, "ifa_anchor_factor", 1.0)
    for name in HOUSEHOLD_STOCK_COLUMNS:
        stock_factors = factors if stock_anchor_factors is None else factors * stock_anchor_factors[name]
        data[name] = data[name].to_numpy(dtype=float) * stock_factors
    population.set_household_total_real_assets()
    population.set_household_financial_assets()
    population.set_household_wealth()
    population.set_household_debt()
    population.set_household_net_wealth()
    scaled_housing = None if housing_df is None else scale_housing_market(housing_df, factors, stock_anchor_factors)

    # Step 3. Scaled values can put rents below the rent/value floor; raise them to it.
    floor_diagnostics = {}
    if scaled_housing is not None:
        rent_before = scaled_housing["Rent"].to_numpy(dtype=float).copy()
        apply_rent_yield_floor(
            household_data=data,
            housing_df=scaled_housing,
            minimum_annual_rental_yield=minimum_annual_rental_yield,
            time_unit=round(12 / population.yearly_factor),
            rental_income_taxes=rental_income_taxes,
        )
        raise_by_dwelling = scaled_housing["Rent"].to_numpy(dtype=float) - rent_before
        owner_occupied = scaled_housing["Is Owner-Occupied"].to_numpy(dtype=bool)
        owner = pd.to_numeric(scaled_housing["Corresponding Owner Household ID"], errors="coerce").to_numpy()
        social = ~(owner >= 0)  # no household owner (NaN or -1)
        groups = (
            ("owner_occupied", owner_occupied),
            ("rented", ~owner_occupied & ~social),
            ("social", ~owner_occupied & social),
        )
        for label, rows in groups:
            floor_diagnostics[f"rent_floor_binding_{label}"] = int(np.count_nonzero(raise_by_dwelling[rows] > 0.0))
            floor_diagnostics[f"rent_floor_uplift_{label}"] = float(raise_by_dwelling[rows].sum())

    data["Balance Sheet Scaling Factor"] = factors
    apply_financial_asset_yield(population, financial_asset_yield, financial_income_tax=financial_income_tax)

    diagnostics = {
        "concept": concept,
        "common_factor": float(np.nansum(model_annual) / np.nansum(hfcs_annual)),
        "factor_p10_p50_p90": [float(v) for v in np.percentile(factors, [10, 50, 90])],
        "n_common_factor_fallback": n_fallback,
        "financial_asset_yield": financial_asset_yield,
        **({} if stock_anchor_factors is None else {"stock_anchor_factors": dict(stock_anchor_factors)}),
        **floor_diagnostics,
    }
    return scaled_housing, diagnostics
