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


CALIBRATED_CONSUMPTION_OVERRIDES: dict[str, Any] = {
    # "households.functions.consumption.parameters.['paper_parameter_ref']": "desired_consumption.credit_augmented_v1",
    "households.functions.consumption.parameters.['long_run_intercept']": -0.3,
    # "households.functions.consumption.parameters.['permanent_income_propensity']": 0.55,
    # "households.functions.consumption.parameters.['liquid_wealth_propensity']": 0.14,
    # "households.functions.consumption.parameters.['illiquid_wealth_propensity']": 0.022,
    # "households.functions.consumption.parameters.['housing_wealth_propensity']": 0.013,
    # "households.functions.consumption.parameters.['income_growth_propensity']": 0.15,
    # "households.functions.consumption.parameters.['interest_rate_cashflow_propensity']": -0.003,
    # "households.functions.consumption.parameters.['uncertainty_propensity']": -0.005,
    "households.functions.consumption.parameters.['partial_adjustment_speed']": 0.7,
    # "households.functions.consumption.parameters.['long_run_mpc_lower_bound']": 0,
    # "households.functions.consumption.parameters.['long_run_mpc_upper_bound']": 2,
    # "firms.functions.wage_setter.parameters['labour_market_tightness_markup_scale']": 0.01,
    # "firms.functions.wage_setter.parameters['markup_time_span']": 1,
    # "central_bank.taylor_rule_overrides['rho']": 0.25,
    # "central_government.tax_overrides['production_tax_vector_scale']": 1,
}

# Formerly commented one-line alternatives in run_model.ipynb. They remain
# separate so selecting one cannot silently alter the calibrated baseline.
EXPERIMENTAL_OVERRIDE_PRESETS: dict[str, dict[str, Any]] = {
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
        "households.parameters.desired_consumption.credit_augmented_v1.['income_growth_propensity']": 0.5,
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

MACRO_COLUMNS = (
    "total_consumption_to_gdp",
    "investment_to_gdp",
    "net_exports_to_gdp",
    "debt_to_gdp",
    "fiscal_revenue_to_gdp",
    "fiscal_expenditure_to_gdp",
    "deficit_to_gdp",
    "household_consumption_to_gdp",
    "government_consumption_to_gdp",
    "real_gdp",
    "gdp_growth",
    "unemployment_rate",
    "central_bank_policy_rate",
    "inventory",
    "profits",
    "wages",
    "wage_rate_yoy_change",
    "cpi_transaction_yoy_change",
    # "cpi_transaction",
    # "ppi",
    "bank_insolvency_rate",
    "avg_tfp_multiplier",
)
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
    "gdp_growth",
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
