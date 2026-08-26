"""Household time series data management.

This module implements time series tracking for household economic variables
through:
- Consumption and investment tracking
- Income source monitoring
- Wealth composition tracking
- Property market participation
- Debt level monitoring

The implementation handles:
- Target and actual consumption
- Investment allocations
- Income components
- Property transactions
- Wealth composition
- Debt positions
- Financial flows
"""

from typing import Optional

import numpy as np
import pandas as pd

from macromodel.timeseries import TimeSeries
from macromodel.util.get_histogram import get_histogram


def create_households_timeseries(
    data: pd.DataFrame,
    initial_consumption_by_industry: np.ndarray,
    initial_hh_investment: np.ndarray,
    initial_investment_by_industry: np.ndarray,
    initial_hh_consumption: np.ndarray,
    scale: int,
    vat: float,
    tau_cf: float,
    consumption_emissions: Optional[np.ndarray] = None,
    investment_emissions: Optional[np.ndarray] = None,
    consumption_emissions_by_good: Optional[np.ndarray] = None,
    investment_emissions_by_good: Optional[np.ndarray] = None,
    consumption_emissions_ch4_by_good: Optional[np.ndarray] = None,
    investment_emissions_ch4_by_good: Optional[np.ndarray] = None,
    coal_consumption_emissions: Optional[np.ndarray] = None,
    gas_consumption_emissions: Optional[np.ndarray] = None,
    oil_consumption_emissions: Optional[np.ndarray] = None,
    refined_products_consumption_emissions: Optional[np.ndarray] = None,
    coal_investment_emissions: Optional[np.ndarray] = None,
    gas_investment_emissions: Optional[np.ndarray] = None,
    oil_investment_emissions: Optional[np.ndarray] = None,
    refined_products_investment_emissions: Optional[np.ndarray] = None,
) -> TimeSeries:
    """Create time series for tracking household economic variables.

    Initializes tracking for:
    - Consumption patterns and targets
    - Investment decisions
    - Income sources and levels
    - Property market activity
    - Wealth composition
    - Debt positions
    - Emissions data

    Args:
        data (pd.DataFrame): Initial household economic data
        initial_consumption_by_industry (np.ndarray): Starting industry consumption
        initial_hh_investment (np.ndarray): Starting household investment
        initial_investment_by_industry (np.ndarray): Starting industry investment
        initial_hh_consumption (np.ndarray): Starting household consumption
        scale (int): Scaling factor for histograms
        vat (float): Value added tax rate
        tau_cf (float): Capital formation tax rate
        consumption_emissions (Optional[np.ndarray]): Initial consumption emissions
        investment_emissions (Optional[np.ndarray]): Initial investment emissions
        coal_consumption_emissions (Optional[np.ndarray]): Coal consumption emissions
        gas_consumption_emissions (Optional[np.ndarray]): Gas consumption emissions
        oil_consumption_emissions (Optional[np.ndarray]): Oil consumption emissions
        refined_products_consumption_emissions (Optional[np.ndarray]):
            Refined products consumption emissions
        coal_investment_emissions (Optional[np.ndarray]): Coal investment emissions
        gas_investment_emissions (Optional[np.ndarray]): Gas investment emissions
        oil_investment_emissions (Optional[np.ndarray]): Oil investment emissions
        refined_products_investment_emissions (Optional[np.ndarray]):
            Refined products investment emissions

    Returns:
        TimeSeries: Initialized time series for household variables
    """
    inhabited_house_id = pd.to_numeric(data["Corresponding Inhabited House ID"], errors="coerce").fillna(-1).values
    owns_main_residence = np.isin(data["Tenure Status of the Main Residence"].values, [1, 2, 4]) & (
        inhabited_house_id >= 0
    )
    initial_wealth_main_residence = np.where(
        owns_main_residence,
        data["Value of the Main Residence"].values,
        0.0,
    )
    initial_wealth_other_properties = data["Value of other Properties"].values
    initial_wealth_other_real_assets = data["Wealth Other Real Assets"].values
    initial_wealth_real_assets = (
        initial_wealth_main_residence + initial_wealth_other_properties + initial_wealth_other_real_assets
    )
    initial_wealth_financial_assets = data["Wealth in Financial Assets"].values
    initial_mortgage_debt = (
        data["Outstanding Balance of HMR Mortgages"].values
        + data["Outstanding Balance of Mortgages on other Properties"].values
    )
    initial_consumption_loan_debt = data["Outstanding Balance of other Non-Mortgage Loans"].values
    initial_debt = initial_mortgage_debt + initial_consumption_loan_debt
    initial_wealth = initial_wealth_real_assets + initial_wealth_financial_assets

    return TimeSeries(
        n_households=len(data),
        #
        target_consumption_loans=np.full(len(data), np.nan),
        total_target_consumption_loans=[0.0],
        target_consumption=initial_hh_consumption,
        target_consumption_lagged_consumption=np.zeros(len(data)),
        target_consumption_real_income=np.zeros(len(data)),
        target_consumption_lagged_real_income=np.zeros(len(data)),
        target_consumption_real_lagged_consumption=np.zeros(len(data)),
        target_consumption_long_run=np.zeros(len(data)),
        target_consumption_log_long_run=np.zeros(len(data)),
        target_consumption_permanent_income=np.zeros(len(data)),
        target_consumption_liquid_wealth=np.zeros(len(data)),
        target_consumption_illiquid_wealth=np.zeros(len(data)),
        target_consumption_housing_wealth=np.zeros(len(data)),
        target_consumption_real_net_liquid_assets=np.zeros(len(data)),
        target_consumption_real_illiquid_financial_assets=np.zeros(len(data)),
        target_consumption_real_housing_wealth=np.zeros(len(data)),
        target_consumption_real_lagged_housing_wealth=np.zeros(len(data)),
        target_consumption_real_consumer_debt=np.zeros(len(data)),
        target_consumption_rent=np.zeros(len(data)),
        target_consumption_mortgage_debt=np.zeros(len(data)),
        target_consumption_mortgage_payment=np.zeros(len(data)),
        target_consumption_rent_diagnostic=np.zeros(len(data)),
        target_consumption_mortgage_debt_diagnostic=np.zeros(len(data)),
        target_consumption_mortgage_payment_diagnostic=np.zeros(len(data)),
        target_consumption_house_price=np.zeros(len(data)),
        target_consumption_interest_rate_cashflow=np.zeros(len(data)),
        target_consumption_uncertainty=np.zeros(len(data)),
        target_consumption_partial_adjustment_gap=np.zeros(len(data)),
        target_consumption_income_growth=np.zeros(len(data)),
        target_consumption_house_price_index=np.zeros(len(data)),
        target_consumption_lagged_house_price_index=np.zeros(len(data)),
        target_consumption_real_lagged_house_price=np.zeros(len(data)),
        target_consumption_real_borrowing_rate=np.zeros(len(data)),
        target_consumption_permanent_income_log_ratio=np.zeros(len(data)),
        target_consumption_permanent_income_log_ratio_individual=np.zeros(len(data)),
        target_consumption_permanent_income_log_ratio_common=np.zeros(len(data)),
        target_consumption_consumer_debt_rate_delta=np.zeros(len(data)),
        target_consumption_interest_rate_cashflow_index=np.zeros(len(data)),
        target_consumption_uncertainty_delta=np.zeros(len(data)),
        target_consumption_owner_occupied=np.zeros(len(data)),
        target_consumption_mortgagor=np.zeros(len(data)),
        target_consumption_delta_log_consumption=np.zeros(len(data)),
        target_consumption_growth_clipped=np.zeros(len(data)),
        target_consumption_alpha_2=np.zeros(len(data)),
        target_consumption_gamma_1=np.zeros(len(data)),
        target_consumption_wealth_drag_clipped=np.zeros(len(data)),
        target_consumption_cash_rent=np.zeros(len(data)),
        target_consumption_imputed_rent=np.zeros(len(data)),
        target_consumption_non_goods_housing=np.zeros(len(data)),
        target_consumption_calibrated_total=np.zeros(len(data)),
        target_consumption_goods_total=np.zeros(len(data)),
        # ECM state variable for CreditAugmentedConsumption: the real consumption
        # budget it produced. Seeded from initial consumption, which is already
        # real at t=0 since the CPI deflator is 1 there.
        cacf_real_consumption_budget=data["Consumption"].values,
        income_belief_floor_used=np.zeros(len(data)),
        income_belief_posterior_fallback_used=np.zeros(len(data)),
        income_belief_growth_clipped=np.zeros(len(data)),
        formula_implied_mpc=np.zeros(len(data)),
        amount_bought=np.full(len(data), np.nan),
        consumption=data["Consumption"].values,
        total_consumption=[(1 + vat) * initial_consumption_by_industry.sum()],
        total_consumption_before_vat=[initial_consumption_by_industry.sum()],
        industry_consumption=initial_consumption_by_industry,
        #
        target_investment=initial_hh_investment,
        investment=initial_hh_investment,
        total_investment=[(1 + tau_cf) * initial_hh_investment.sum()],
        total_investment_before_vat=[initial_hh_investment.sum()],
        industry_investment=initial_investment_by_industry,
        #
        # HFCS source components are preserved as initialization diagnostics;
        # they are not used as an additional live income flow.
        hfcs_pension_income=scale * data.get("Pension Income", pd.Series(0.0, index=data.index)).values,
        hfcs_public_pension_income=scale * data.get("Public Pension Income", pd.Series(0.0, index=data.index)).values,
        hfcs_occupational_private_pension_income=scale
        * data.get("Occupational and Private Pension Income", pd.Series(0.0, index=data.index)).values,
        hfcs_unemployment_benefits=scale * data.get("Unemployment Benefits", pd.Series(0.0, index=data.index)).values,
        hfcs_social_transfer_income=scale * data.get("Social Transfer Income", pd.Series(0.0, index=data.index)).values,
        fiscal_public_pension_benefits=data.get(
            "Allocated Public Pension Benefits", pd.Series(0.0, index=data.index)
        ).values,
        fiscal_other_social_transfers=data.get(
            "Allocated Other Social Transfers", pd.Series(0.0, index=data.index)
        ).values,
        income=data["Income"].values,
        income_histogram=get_histogram(data["Income"].values, scale),
        expected_income=data["Income"].values,
        non_property_income=data["Employee Income"].values
        + data["Regular Social Transfers"].values
        + data["Rental Income from Real Estate"].values,
        income_employee=data["Employee Income"].values,
        total_income_employee=[data["Employee Income"].values.sum()],
        expected_income_employee=data["Employee Income"].values,
        income_social_transfers=data["Regular Social Transfers"].values,
        total_income_social_transfers=[data["Regular Social Transfers"].values.sum()],
        stage5_subsistence_support=np.zeros(len(data)),
        expected_income_social_transfers=data["Regular Social Transfers"].values,
        income_rental=data["Rental Income from Real Estate"].values,
        total_income_rental=[data["Rental Income from Real Estate"].values.sum()],
        income_financial_assets=data["Income from Financial Assets"].values,
        total_income_financial_assets=[data["Income from Financial Assets"].values.sum()],
        expected_income_financial_assets=data["Income from Financial Assets"].values,
        income_dividend_distributions=np.zeros(len(data)),
        total_income_dividend_distributions=[0.0],
        dividend_fund_income_tax_withheld=np.zeros(len(data)),
        total_dividend_fund_income_tax_withheld=[0.0],
        expected_income_dividend_distributions=np.zeros(len(data)),
        total_expected_income_dividend_distributions=[0.0],
        illiquid_financial_asset_capital_gains=np.zeros(len(data)),
        total_illiquid_financial_asset_capital_gains=[0.0],
        dividend_fund_settled_firm_distribution=np.zeros(len(data)),
        dividend_fund_settled_bank_distribution=np.zeros(len(data)),
        dividend_fund_declared_firm_distribution=np.zeros(len(data)),
        dividend_fund_declared_bank_distribution=np.zeros(len(data)),
        dividend_fund_total_settled_distribution=[0.0],
        dividend_fund_total_firm_distributable_profit_candidate=[0.0],
        dividend_fund_total_bank_distributable_profit_candidate=[0.0],
        dividend_fund_total_firm_declared_distribution=[0.0],
        dividend_fund_total_bank_declared_distribution=[0.0],
        dividend_fund_total_firm_retained_capacity=[0.0],
        dividend_fund_total_bank_retained_capacity=[0.0],
        dividend_fund_quota_sum=[0.0],
        dividend_fund_firm_settlement_identity_error=[0.0],
        dividend_fund_bank_settlement_identity_error=[0.0],
        dividend_fund_settlement_identity_error=[0.0],
        dividend_fund_firm_retained_capacity_identity_error=[0.0],
        dividend_fund_bank_retained_capacity_identity_error=[0.0],
        dividend_fund_fund_identity_error=[0.0],
        #
        price_paid_for_property=np.full(len(data), np.nan),
        rent=data["Rent Paid"].values,
        rent_imputed=data["Rent Imputed"].values,
        max_price_willing_to_pay=np.full(len(data), np.nan),
        max_rent_willing_to_pay=np.full(len(data), np.nan),
        rent_div_income_histogram=get_histogram(data["Rent Paid"].values / data["Income"].values, None),
        #
        wealth=initial_wealth,
        wealth_histogram=get_histogram(initial_wealth, scale),
        wealth_real_assets=initial_wealth_real_assets,
        wealth_main_residence=initial_wealth_main_residence,
        total_wealth_main_residence=[np.sum(initial_wealth_main_residence)],
        wealth_other_properties=initial_wealth_other_properties,
        total_wealth_other_properties=[np.sum(initial_wealth_other_properties)],
        wealth_other_real_assets=initial_wealth_other_real_assets,
        total_wealth_other_real_assets=[np.sum(initial_wealth_other_real_assets)],
        liquid_financial_assets=data["Wealth in Deposits"].values,
        total_liquid_financial_assets=[np.sum(data["Wealth in Deposits"].values)],
        illiquid_financial_assets=data["Wealth in Other Financial Assets"].values,
        total_illiquid_financial_assets=[np.sum(data["Wealth in Other Financial Assets"].values)],
        wealth_financial_assets=data["Wealth in Financial Assets"].values,
        #
        mortgage_debt=initial_mortgage_debt,
        total_mortgage_debt=[np.sum(initial_mortgage_debt)],
        consumption_loan_debt=initial_consumption_loan_debt,
        received_consumption_loans=np.full(len(data), np.nan),
        total_consumption_loan_debt=[np.sum(initial_consumption_loan_debt)],
        debt=initial_debt,
        total_received_consumption_loans=[0.0],
        debt_histogram=get_histogram(initial_debt, scale),
        #
        net_wealth=initial_wealth - initial_debt,
        #
        target_mortgage=np.full(len(data), np.nan),
        total_target_mortgage=[0.0],
        received_mortgages=np.full(len(data), np.nan),
        total_received_mortgages=[0.0],
        #
        debt_installments=data["Debt Installments"].values,
        total_debt_installments=[data["Debt Installments"].values.sum()],
        #
        interest_paid_on_deposits=np.full(len(data), np.nan),
        interest_paid_on_loans=np.full(len(data), np.nan),
        interest_paid=np.full(len(data), np.nan),
        consumption_emissions=consumption_emissions,
        investment_emissions=investment_emissions,
        consumption_emissions_by_good=consumption_emissions_by_good,
        investment_emissions_by_good=investment_emissions_by_good,
        consumption_emissions_ch4_by_good=consumption_emissions_ch4_by_good,
        investment_emissions_ch4_by_good=investment_emissions_ch4_by_good,
        coal_consumption_emissions=coal_consumption_emissions,
        gas_consumption_emissions=gas_consumption_emissions,
        oil_consumption_emissions=oil_consumption_emissions,
        refined_products_consumption_emissions=refined_products_consumption_emissions,
        coal_investment_emissions=coal_investment_emissions,
        gas_investment_emissions=gas_investment_emissions,
        oil_investment_emissions=oil_investment_emissions,
        refined_products_investment_emissions=refined_products_investment_emissions,
    )
