"""Time series management for Central Government agent.

This module handles the creation and management of time series data
for the central government agent, including:
- Fiscal variables (revenue, deficit, debt)
- Tax collections by type
- Social benefits and transfers
- Public housing income

The time series provide historical tracking of:
- Government financial position
- Tax revenue streams
- Social benefit payments
- Public sector operations
"""

import numpy as np
import pandas as pd

from macromodel.timeseries import TimeSeries


def create_central_government_timeseries(
    data: pd.DataFrame,
    number_of_unemployed_individuals: int,
    initial_unemployment_benefit: float,
) -> TimeSeries:
    """Create time series objects for central government variables.

    Initializes time series tracking for:
    - Fiscal position (debt, deficit, revenue)
    - Tax collections by type
    - Social benefits and transfers
    - Public housing income

    Args:
        data (pd.DataFrame): Initial government data including historical
            values for all tracked variables
        number_of_unemployed_individuals (int): Count of unemployed people
            for per-person benefit calculation
        initial_unemployment_benefit (float): Calibrated model subsidy per
            unemployed individual

    Returns:
        TimeSeries: Initialized time series containing all government
            variables with their initial values
    """
    if number_of_unemployed_individuals < 0:
        raise ValueError("Number of unemployed individuals must be non-negative.")
    initial_unemployment_benefit = float(initial_unemployment_benefit)
    if not np.isfinite(initial_unemployment_benefit) or initial_unemployment_benefit < 0.0:
        raise ValueError("Initial unemployment benefit must be a finite non-negative number.")
    settled_initial_unemployment_benefits = initial_unemployment_benefit * number_of_unemployed_individuals
    public_pension_benefits = data.get("Public Pension Benefits", pd.Series([0.0])).values[0]
    other_social_benefits = data["Other Social Benefits"].values[0]
    return TimeSeries(
        debt=np.array([float(data["Debt"].iloc[0])]),
        unemployment_benefits_by_individual=[initial_unemployment_benefit],
        # Planned real component controls.
        public_pension_benefits=[public_pension_benefits],
        total_other_benefits=[other_social_benefits],
        reader_non_unemployment_social_benefits=[
            data.get("Reader Non-Unemployment Social Benefits", pd.Series([other_social_benefits])).values[0]
        ],
        # Settled nominal fiscal components.
        total_unemployment_benefits=[settled_initial_unemployment_benefits],
        total_public_pension_benefits=[public_pension_benefits],
        total_other_social_transfers=[other_social_benefits],
        total_necessity_support=[0.0],
        total_household_social_transfers=[
            settled_initial_unemployment_benefits + public_pension_benefits + other_social_benefits
        ],
        interest_payments_on_debt=[0.0],
        debt_interest_rate=[np.nan],
        #
        taxes_production=[data["Taxes on Production"].values[0]],
        taxes_vat=[data["VAT"].values[0]],
        taxes_cf=[data["Capital Formation Taxes"].values[0]],
        taxes_corporate_income=[data["Corporate Taxes"].values[0]],
        taxes_exports=[data["Export Taxes"].values[0]],
        taxes_income=[data["Income Taxes"].values[0]],
        taxes_rental_income=[data["Rental Income Taxes"].values[0]],
        taxes_employee_si=[data["Employee SI Tax"].values[0]],
        taxes_employer_si=[data["Employer SI Tax"].values[0]],
        taxes_on_products=[data["Taxes on Products"].values[0]],
        total_rent_received=[data["Total Social Housing Rent"].values[0]],
        #
        revenue=[data["Revenue"].values[0]],
        deficit=np.array([np.nan]),
        #
        bank_equity_injection=[data["Bank Equity Injection"].values[0]],
    )
