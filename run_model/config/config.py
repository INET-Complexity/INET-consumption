from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import environs


@dataclass
class Config:
    data_dir: Path = Path("data/hfcs")
    config_dir: Path = Path("config/")
    api_dotenv_path: Path = Path("/Users/andone/.ssh/macro_ts_api_keys.env")
    model_dotenv_path: Path = Path(__file__).with_name(".env")
    country_name: str = "France"
    country_iso3: str = "FRA"
    country_iso2: str = "FR"
    seed: int = 45
    t_max: int = 100
    raw_data_path: Path = Path("data/raw_data")
    output_path: Path = Path("data/output_data")
    fred_api_key: str | None = None
    webstat_api_key: str | None = None

    @classmethod
    def from_env(
        cls,
        api_dotenv_path: str | Path = "/Users/andone/.ssh/macro_ts_api_keys.env",
        model_dotenv_path: str | Path | None = None,
        **overrides,
    ) -> "Config":
        run_model_dir = Path(__file__).resolve().parents[1]
        model_env_path = Path(model_dotenv_path) if model_dotenv_path is not None else Path(__file__).with_name(".env")

        model_env = environs.Env()
        model_env.read_env(str(model_env_path), recurse=False)

        api_env = environs.Env()
        api_env.read_env(str(api_dotenv_path), recurse=False)

        raw_data_path = Path(model_env("RAW_DATA_PATH"))
        output_path = Path(model_env("OUTPUT_DATA_PATH"))

        if not raw_data_path.is_absolute():
            raw_data_path = run_model_dir / raw_data_path
        if not output_path.is_absolute():
            output_path = run_model_dir / output_path

        return cls(
            api_dotenv_path=Path(api_dotenv_path),
            model_dotenv_path=model_env_path,
            raw_data_path=raw_data_path,
            output_path=output_path,
            fred_api_key=api_env("FRED_API_KEY", default=None),
            webstat_api_key=api_env("WEBSTAT_API_KEY", default=None),
            **overrides,
        )


# Shared consumption settings are authoritative in
# consumption_paper_parameters.yaml -> desired_consumption.credit_augmented.
# Keep this public preset name for notebook compatibility; it adds no hidden
# intercept, smoothing or residual-mode override to the country baseline.
CALIBRATED_CONSUMPTION_OVERRIDES: dict[str, Any] = {
    "households.functions.consumption.parameters['long_run_intercept']": 0.15,
}

# Formerly commented one-line alternatives in run_model.ipynb. They remain
# separate so selecting one cannot silently alter the calibrated baseline.
EXPERIMENTAL_OVERRIDE_PRESETS: dict[str, dict[str, Any]] = {
    # Paired income-definition comparison: both inherit exactly the same
    # calibrated overrides. Only the behavioural-income input switch differs.
    "investment_adjustment": {
        "firms.functions.target_capital_inputs.parameters['target_capital_inputs_fraction']": 0.05,
    },
    "tfp_investment_adjustment_cost": {
        "firms.functions.productivity_investment_planner.parameters['adjustment_cost_lambda']": 0.75,
    },
    "calibrated_income_legacy": {
        **CALIBRATED_CONSUMPTION_OVERRIDES,
        "households.functions.consumption.parameters['uses_aligned_behavioural_income']": False,
    },
    "calibrated_income_reconciled": {
        **CALIBRATED_CONSUMPTION_OVERRIDES,
        "households.functions.consumption.parameters['uses_aligned_behavioural_income']": True,
    },
    "no_target_inventory_buffer": {
        "firms.functions.target_production.parameters['target_inventory_to_demand_fraction']": 0.0,
    },
    "disable_income_belief_learning": {
        "households.functions.consumption.parameters['uses_income_belief_learning']": False,
    },
    "zero_consumption_intercept": {
        "households.functions.consumption.parameters['long_run_intercept']": 0,
    },
    "income_growth_propensity_0_15": {
        "households.functions.consumption.parameters['income_growth_propensity']": 0.15,
    },
    "disable_portfolio_choice": {
        "households.functions.wealth.parameters['uses_portfolio_choice']": False,
        "households.functions.wealth.parameters['settles_portfolio_choice']": False,
    },
    "disable_feasibility_resolver": {
        "households.parameters.uses_feasibility_resolver": False,
    },
    "inventory_adjustment_speed_0_75": {
        "firms.functions.target_production.parameters['inventory_adjustment_speed']": 0.75,
    },
    "income_growth_propensity_0_5_legacy_path": {
        "households.functions.consumption.parameters['income_growth_propensity']": 0.5,
    },
    "income_growth_propensity_0_5": {
        "households.functions.consumption.parameters.['income_growth_propensity']": 0.5,
    },
    "change_sectoral_weights": {
        "government_entities.functions.consumption.parameters['sectoral_weights']": "initial_price_normalized",
    },
    "default_labour_cleaner": {
        "labour_market.functions.clearing.name": "ReservationWageBindingDefaultLabourMarketClearer",
    },
    "contract_wage_setter": {
        "firms.functions.wage_setter.name": "ContractWageSetter",
        "firms.functions.wage_setter.parameters['initial_rate_source']": "firm_anchor",
        "firms.functions.wage_setter.parameters['indexation_base']": "realised_productivity",
        "firms.functions.wage_setter.parameters['realised_productivity_window']": 4,
    },
    "realised_productivity_frozen_markup": {
        "firms.functions.wage_setter.name": "ContractWageSetter",
        "firms.functions.wage_setter.parameters['initial_rate_source']": "individual",
        "firms.functions.wage_setter.parameters['indexation_base']": "realised_productivity",
        "firms.functions.wage_setter.parameters['realised_productivity_window']": 4,
        "firms.functions.wage_setter.parameters['incumbent_indexation_pass_through']": 0.0,
        "firms.functions.prices.parameters['demand_pull_speed']": 0.0,
        "labour_market.functions.clearing.name": "ReservationWageBindingDefaultLabourMarketClearer",
        "households.functions.consumption.parameters['long_run_intercept']": -0.3,
    },
    # France's country configuration is the alpha=0 benchmark. These presets
    # change only the wage-indexation coefficient, keeping all other France
    # settings—including active demand-pull markup adjustment—identical.
    "wage_indexation_current_alpha_1": {
        "firms.functions.wage_setter.parameters['incumbent_indexation_pass_through']": 1.0,
    },
    "wage_indexation_fixed_alpha_0": {
        "firms.functions.wage_setter.parameters['incumbent_indexation_pass_through']": 0.0,
    },
    "labour_market_tightness": {
        "firms.functions.wage_setter.parameters['labour_market_tightness_markup_scale']": 0.5,
    },
    "debt_service_to_income_ratio": {
        "banks.parameters.['firm_loans_min_dscr']": 2,
    },
    "production_tax_vector_scale": {
        "central_government.tax_overrides['production_tax_vector_scale']": 4.0,
    },
    "firm_dividend_payout_ratio": {
        "households.functions.wealth.parameters['dividend_fund_firm_payout_ratio']": 0.75,
    },
    "hiring_speed_1_0": {
        "labour_market.functions.clearing.parameters['hiring_speed']": 1.0,
    },
}


SCENARIO_PRESETS: dict[str, dict[str, Any]] = {
    "country_config": {},
    "calibrated_consumption": CALIBRATED_CONSUMPTION_OVERRIDES,
    **EXPERIMENTAL_OVERRIDE_PRESETS,
}

FIGURE_SIZES = {
    "wide": {"base_height": 360, "base_width": 600},
    "dense": {"base_height": 300, "base_width": 520},
    "benchmark": {"base_height": 220, "base_width": 380},
    "sensitivity": {"base_height": 500, "base_width": 600},
}

MACRO_COLUMNS = [
    # Activity and demand
    "real_gdp",
    "real_gdp_yoy_growth",
    "total_consumption_to_gdp",
    "household_consumption_to_gdp",
    "government_consumption_to_gdp",
    "investment_to_gdp",
    "net_exports_to_gdp",
    "wages",
    "profits",
    # Prices, labour, and monetary transmission
    "cpi_fixed_basket_yoy_change",
    "ppi_yoy_change",
    "unemployment_rate",
    "participation_rate",
    "vacancy_rate",
    "central_bank_policy_rate",
    "long_term_firm_borrowing_rate",
    "household_consumption_borrowing_rate",
    "mortgage_borrowing_rate",
    # Public finances and fragility
    "debt_to_gdp",
    "deficit_to_gdp",
    "fiscal_revenue_to_gdp",
    "fiscal_expenditure_to_gdp",
    "interest_payments_on_debt",
    "bank_insolvency_rate",
    # Productivity and accounting diagnostics
    "total_productivity_investment_to_gdp",
    # "avg_tfp_multiplier",
    # "avg_tfp_multiplier_growth",
    "avg_tfp_multiplier_yoy_growth",
    "employed_workers",
    # "estimated_hours_worked",
    "labour_productivity_proxy",
    "household_balance_sheet_identity_residual",
    "firm_balance_sheet_identity_residual",
    "bank_balance_sheet_identity_residual_proxy",
    # Credit demand and receipts, by borrower and maturity/product
    "firm_credit_demand_short_term",
    "firm_credit_received_short_term",
    "firm_credit_demand_long_term",
    "firm_credit_received_long_term",
    "household_credit_demand_consumption",
    "household_credit_received_consumption",
    "household_credit_demand_mortgage",
    "household_credit_received_mortgage",
]

FISCAL_COLUMNS = (
    "unemployment_benefits",
    "household_social_transfers",
    "government_consumption",
    "interest_payments_on_debt",
    "fiscal_revenue_to_gdp",
    "fiscal_expenditure_to_gdp",
    "deficit_to_gdp",
    "unemployment_benefits_to_expenditure",
    "household_social_transfers_to_expenditure",
    "government_consumption_to_expenditure",
    "interest_payments_on_debt_to_expenditure",
    "unemployment_rate",
    "cpi_transaction",
    "debt_to_gdp",
    "central_bank_policy_rate",
    "fiscal_expenditure",
    "real_gdp_yoy_growth",
    "taxes_paid_on_production",
    "taxes_on_products",
    "real_demand",
)
POLICY_COLUMNS = (
    "unemployment_rate",
    "cpi_fixed_basket_yoy_change",
    "ppi_yoy_change",
    "cpi_transaction",
    "ppi",
    "central_bank_policy_rate",
    "output_gap",
    "short_term_firm_borrowing_rate",
    "long_term_firm_borrowing_rate",
    "household_consumption_borrowing_rate",
    "mortgage_borrowing_rate",
    "short_term_loans_to_firms",
    "long_term_loans_to_firms",
    "consumption_loans_to_households",
    "mortgages_to_households",
)
LABOUR_COLUMNS = (
    "wages",
    "economy_wage_rate",
    "participation_rate",
    "labour_input_shortfall_rate",
    "unfilled_jobs",
    "unemployment_rate",
    "vacancy_rate",
    "job_reallocation_rate",
    "job_reallocation_rate_growth",
)
BALANCE_SHEET_COLUMNS = (
    "household_assets",
    "household_liabilities",
    "household_net_worth",
    "household_balance_sheet_identity_residual",
    "firm_assets",
    "firm_liabilities",
    "firm_equity",
    "firm_balance_sheet_identity_residual",
    "bank_assets_proxy",
    "bank_liabilities",
    "bank_equity",
    "bank_balance_sheet_identity_residual_proxy",
)

# Housing prices, rental flows, and mortgage-market activity. HPI is the
# model's only aggregate house-price index; no separate RPI series is emitted.
HOUSING_COLUMNS = (
    "hpi",
    # "hpi_inflation",
    # "estimated_hpi_inflation",
    "total_real_rent_paid",
    "total_real_rent_rec",
    "total_imp_rent_paid",
    "total_number_of_bought_houses",
    "total_number_of_newly_rented_houses",
    "total_number_of_houses_rented",
    "total_number_of_houses_owner_occupied",
    "total_number_of_houses_unoccupied",
    "mortgages_to_households",
    "mortgage_borrowing_rate",
    "household_credit_demand_mortgage",
    "household_credit_received_mortgage",
    "new_credit_granted_mortgages",
)


# Human-readable names for plots and shared result tables. Machine-readable
# dataframe columns retain their original keys.
_SERIES_LABEL_OVERRIDES = {
    "real_gdp": "Real GDP",
    "gdp_growth": "Nominal GDP growth (per period)",
    "real_gdp_yoy_growth": "Real GDP growth (y/y)",
    "total_consumption_to_gdp": "Consumption/GDP",
    "household_consumption_to_gdp": "Household consumption/GDP",
    "government_consumption_to_gdp": "Government consumption/GDP",
    "investment_to_gdp": "Investment/GDP",
    "net_exports_to_gdp": "Net exports/GDP",
    "debt_to_gdp": "Public debt/annual GDP",
    "fiscal_revenue_to_gdp": "Revenue/GDP",
    "fiscal_expenditure_to_gdp": "Expenditure/GDP",
    "deficit_to_gdp": "Deficit/GDP",
    "unemployment_benefits_to_expenditure": "Unemployment benefits/expenditure",
    "household_social_transfers_to_expenditure": "Social transfers/expenditure",
    "government_consumption_to_expenditure": "Government consumption/expenditure",
    "interest_payments_on_debt_to_expenditure": "Debt interest/expenditure",
    "cpi_fixed_basket": "Fixed-basket consumer price index",
    "cpi_transaction": "Transaction-based consumer price index",
    "ppi": "Producer price index",
    "cpi_fixed_basket_yoy_change": "Fixed-basket CPI inflation (y/y)",
    "cpi_transaction_yoy_change": "Transaction CPI inflation (y/y)",
    "cpi_chained_basket_yoy_change": "Chained-basket CPI inflation (y/y)",
    "ppi_yoy_change": "PPI inflation (y/y)",
    "central_bank_policy_rate": "Policy rate (annualised)",
    "short_term_firm_borrowing_rate": "Short-term firm loan rate (annualised)",
    "long_term_firm_borrowing_rate": "Long-term firm loan rate (annualised)",
    "household_consumption_borrowing_rate": "Household loan rate (annualised)",
    "mortgage_borrowing_rate": "Mortgage rate (annualised)",
    "participation_rate": "Labour force participation rate",
    "labour_input_shortfall_rate": "Labour input shortfall rate",
    "unfilled_jobs": "Unfilled job vacancies",
    "vacancy_rate": "Job vacancy rate",
    "job_reallocation_rate": "Job reallocation rate",
    "job_reallocation_rate_growth": "Job reallocation growth (per period)",
    "wage_rate_yoy_change": "Wage growth (y/y)",
    "total_productivity_investment_to_gdp": "Productivity investment/GDP",
    "avg_tfp_multiplier": "Mean firm TFP multiplier (index)",
    "avg_tfp_multiplier_growth": "Mean firm TFP growth (per period)",
    "avg_tfp_multiplier_yoy_growth": "Mean firm TFP growth (y/y)",
    "employed_workers": "Employed workers (headcount)",
    "estimated_hours_worked": "Estimated hours worked (1,607 hours per worker per year)",
    "labour_productivity_proxy": "Labour productivity proxy",
    "bank_insolvency_rate": "Bank insolvency rate",
    "firm_insolvency_rate": "Firm insolvency rate",
    "household_insolvency_rate": "Household insolvency rate",
    "output_gap": "Output gap (relative to model trend)",
    "economy_wage_rate": "Economy-wide wage rate",
    "household_balance_sheet_identity_residual": "Household balance-sheet identity residual",
    "firm_balance_sheet_identity_residual": "Firm balance-sheet identity residual",
    "bank_balance_sheet_identity_residual_proxy": "Bank balance-sheet residual (proxy)",
    "firm_credit_demand_short_term": "Firm short-term credit demand",
    "firm_credit_received_short_term": "Firm short-term credit received",
    "firm_credit_demand_long_term": "Firm long-term credit demand",
    "firm_credit_received_long_term": "Firm long-term credit received",
    "household_credit_demand_consumption": "Household consumption-credit demand",
    "household_credit_received_consumption": "Household consumption-credit received",
    "household_credit_demand_mortgage": "Household mortgage-credit demand",
    "household_credit_received_mortgage": "Household mortgage-credit received",
    "hpi": "House price index (HPI)",
    "hpi_inflation": "House price index growth (per period)",
    "estimated_hpi_inflation": "Expected house price index growth (per period)",
    "total_real_rent_paid": "Rent paid (total, nominal LCU)",
    "total_imp_rent_paid": "Imputed rent (total, nominal LCU)",
    "total_real_rent_rec": "Rental income received (total, nominal LCU)",
    "mortgages_to_households": "Mortgage debt (stock, nominal LCU)",
    "new_credit_granted_mortgages": "New mortgages granted (flow, nominal LCU)",
    "total_number_of_bought_houses": "Homes purchased (completed sales)",
    "total_number_of_newly_rented_houses": "New rental agreements",
    "total_number_of_houses_rented": "Rented homes (stock)",
    "total_number_of_houses_owner_occupied": "Owner-occupied homes (stock)",
    "total_number_of_houses_unoccupied": "Unoccupied homes (stock)",
    "inventory": "Firm inventories (quantity)",
    "profits": "Firm profits",
    "wages": "Wage bill",
    "interest_payments_on_debt": "Public debt interest payments",
    "household_assets": "Household assets",
    "household_liabilities": "Household liabilities",
    "household_net_worth": "Household net worth",
    "firm_assets": "Firm assets",
    "firm_liabilities": "Firm liabilities",
    "firm_equity": "Firm equity",
    "bank_assets_proxy": "Bank assets (proxy)",
    "bank_liabilities": "Bank liabilities",
    "bank_equity": "Bank equity",
    "unemployment_rate": "Unemployment rate",
    "fiscal_revenue": "Fiscal revenue",
    "fiscal_expenditure": "Fiscal expenditure",
    "real_demand": "Real demand",
    "taxes_paid_on_production": "Taxes on production",
    "taxes_on_products": "Taxes on products",
}


def _humanize_series_name(series_name: str) -> str:
    """Turn an unlisted snake-case output key into a readable display name."""
    if series_name.endswith("_to_gdp"):
        subject = _humanize_series_name(series_name[: -len("_to_gdp")])
        return f"{subject} as a share of GDP"
    if series_name.endswith("_to_expenditure"):
        subject = _humanize_series_name(series_name[: -len("_to_expenditure")])
        return f"{subject} as a share of expenditure"
    if series_name.endswith("_yoy_change"):
        subject = _humanize_series_name(series_name[: -len("_yoy_change")])
        return f"{subject} change, yoy"
    words = series_name.replace("_", " ").split()
    acronyms = {"cpi": "CPI", "gdp": "GDP", "ppi": "PPI", "tfp": "TFP", "yoy": "yoy"}
    return " ".join(acronyms.get(word.lower(), word.capitalize()) for word in words)


_CONFIGURED_SERIES = tuple(
    dict.fromkeys(MACRO_COLUMNS + list(FISCAL_COLUMNS + POLICY_COLUMNS + LABOUR_COLUMNS + BALANCE_SHEET_COLUMNS))
)
SERIES_LABELS = {name: _humanize_series_name(name) for name in _CONFIGURED_SERIES}
SERIES_LABELS.update(_SERIES_LABEL_OVERRIDES)


def series_label(series_name: str) -> str:
    """Return the external-facing label for a result series."""
    return SERIES_LABELS.get(series_name, _humanize_series_name(str(series_name)))


def label_result_columns(frame, *, include_code: bool = False):
    """Return a copy of a result table with readable column names.

    Set ``include_code=True`` to retain the machine-readable key in
    parentheses for reproducible handoffs.
    """
    labels = {
        column: f"{series_label(column)} ({column})" if include_code else series_label(column)
        for column in frame.columns
    }
    return frame.rename(columns=labels).copy()
