"""Module for preprocessing loan-specific credit relationship data.

This module provides dataclasses for organizing different types of loan data that will be
used to initialize behavioral models. Key preprocessing includes:

1. Loan Parameter Processing:
   - Principal amount calculations
   - Interest rate applications
   - Annuity payment computations

2. Bank-Borrower Relationships:
   - Firm-bank loan mappings
   - Household-bank loan mappings
   - Initial loan state organization

3. Loan Type Specialization:
   - Long-term firm loans
   - Short-term firm loans
   - Consumer loans
   - Payday loans
   - Mortgages

Note:
    This module is NOT used for simulating loan behavior. It only handles
    the preprocessing and organization of loan-specific data that will later
    be used to initialize behavioral models in the simulation package.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LoanData:
    """Base class for preprocessing loan-specific credit data.

    This class provides a framework for organizing loan parameters that will be used
    to initialize behavioral models. It is NOT used for simulating loan behavior -
    it only handles data preprocessing.

    The preprocessed data includes:
    - Principal amounts by bank-borrower pair
    - Period interest rates by bank-borrower pair
    - Scheduled annuity payments by bank-borrower pair

    Note:
        This is a data container class. The actual loan behavior (repayment,
        default, etc.) is implemented in the simulation package, which uses
        this preprocessed data for initialization.

    Attributes:
        principal (np.ndarray): Initial loan principal amounts
        interest (np.ndarray): Initial loan period rates
        installments (np.ndarray): Initial scheduled annuity payments
    """

    principal: np.ndarray
    interest: np.ndarray
    installments: np.ndarray

    def stack(self):
        """Stack loan parameters for preprocessing.

        Returns:
            np.ndarray: Stacked array of [principal, period rate, scheduled payment]
        """
        return np.stack([self.principal, self.interest, self.installments])


def annuity_payment(principal: float, period_rate: float, maturity: int) -> float:
    """Return the fixed per-period annuity payment for a principal."""
    if principal <= 0.0 or maturity <= 0:
        return 0.0
    if not np.isfinite(period_rate) or period_rate <= 0.0:
        return principal / maturity
    return principal * period_rate / (1.0 - (1.0 + period_rate) ** (-maturity))


@dataclass
class LongtermLoans(LoanData):
    """Container for preprocessed long-term firm loan data.

    This class organizes initial state data for long-term loans to firms. It processes:
    - Principal amounts from firm debt data
    - Period rates from bank long-term rates
    - Scheduled annuity payments based on maturity

    Note:
        This is a data container class. The actual loan behavior is implemented
        in the simulation package.
    """

    @classmethod
    def from_agent_data(
        cls, bank_data: pd.DataFrame, firm_data: pd.DataFrame, firm_loan_maturity: int = 60
    ) -> "LongtermLoans":
        """Create a preprocessed long-term loan data container.

        This method:
        1. Extracts firm debt data
        2. Matches with bank rate data
        3. Calculates initial parameters

        Args:
            bank_data (pd.DataFrame): Bank data with rates
            firm_data (pd.DataFrame): Firm data with debt
            firm_loan_maturity (int, optional): Initial maturity. Defaults to 60.

        Returns:
            LongtermLoans: Container with preprocessed loan data
        """
        firm_debt = firm_data["Debt"].values
        firms_corresponding_bank = firm_data["Corresponding Bank ID"].values

        principal = np.zeros((bank_data.shape[0], firm_debt.shape[0]))
        interest = np.zeros((bank_data.shape[0], firm_debt.shape[0]))
        discount = np.zeros((bank_data.shape[0], firm_debt.shape[0]))

        for firm_id in range(firm_debt.shape[0]):
            principal[firms_corresponding_bank[firm_id], firm_id] = firm_debt[firm_id]
            rate = bank_data["Long-Term Interest Rates on Firm Loans"].values[firms_corresponding_bank[firm_id]]
            interest[firms_corresponding_bank[firm_id], firm_id] = rate
            discount[firms_corresponding_bank[firm_id], firm_id] = annuity_payment(
                firm_debt[firm_id],
                rate,
                firm_loan_maturity,
            )

        return cls(principal, interest, discount)


@dataclass
class ShorttermLoans(LoanData):
    """Container for preprocessed short-term firm loan data.

    This class organizes initial state data for short-term loans to firms. It processes:
    - Principal amounts from firm debt data
    - Period rates from bank short-term rates
    - Scheduled annuity payments based on maturity

    Note:
        This is a data container class. The actual loan behavior is implemented
        in the simulation package.
    """

    @classmethod
    def from_agent_data(
        cls, bank_data: pd.DataFrame, firm_data: pd.DataFrame, firm_loan_maturity: int = 60
    ) -> "ShorttermLoans":
        """Create a preprocessed short-term loan data container.

        This method:
        1. Extracts firm debt data
        2. Matches with bank rate data
        3. Calculates initial parameters

        Args:
            bank_data (pd.DataFrame): Bank data with rates
            firm_data (pd.DataFrame): Firm data with debt
            firm_loan_maturity (int, optional): Initial maturity. Defaults to 60.

        Returns:
            ShorttermLoans: Container with preprocessed loan data
        """
        firm_debt = firm_data["Debt"].values

        principal = np.zeros((bank_data.shape[0], firm_debt.shape[0]))
        interest = np.zeros((bank_data.shape[0], firm_debt.shape[0]))
        discount = np.zeros((bank_data.shape[0], firm_debt.shape[0]))

        return cls(principal, interest, discount)


@dataclass
class ConsumptionExpansionLoans(LoanData):
    """Container for preprocessed consumer loan data.

    This class organizes initial state data for household consumption loans. It processes:
    - Principal amounts from household debt data
    - Period rates from bank consumer rates
    - Scheduled annuity payments based on maturity

    Note:
        This is a data container class. The actual loan behavior is implemented
        in the simulation package.
    """

    @classmethod
    def from_agent_data(
        cls, bank_data: pd.DataFrame, household_data: pd.DataFrame, consumption_loan_maturity: int = 1
    ) -> "ConsumptionExpansionLoans":
        """Create a preprocessed consumer loan data container.

        This method:
        1. Extracts household debt data
        2. Matches with bank rate data
        3. Calculates initial parameters

        Args:
            bank_data (pd.DataFrame): Bank data with rates
            household_data (pd.DataFrame): Household data with debt
            consumption_loan_maturity (int, optional): Initial maturity. Defaults to 1.

        Returns:
            ConsumptionExpansionLoans: Container with preprocessed loan data
        """
        household_other_debt = household_data["Outstanding Balance of other Non-Mortgage Loans"].values
        households_corresponding_bank = household_data["Corresponding Bank ID"].values

        principal = np.zeros((bank_data.shape[0], household_other_debt.shape[0]))
        interest = np.zeros((bank_data.shape[0], household_other_debt.shape[0]))
        discount = np.zeros((bank_data.shape[0], household_other_debt.shape[0]))

        for household_id in range(household_other_debt.shape[0]):
            principal[households_corresponding_bank[household_id], household_id] = household_other_debt[household_id]
            rate = bank_data["Interest Rates on Household Consumption Loans"].values[
                households_corresponding_bank[household_id]
            ]
            interest[households_corresponding_bank[household_id], household_id] = rate
            discount[households_corresponding_bank[household_id], household_id] = annuity_payment(
                household_other_debt[household_id],
                rate,
                consumption_loan_maturity,
            )

        return cls(principal, interest, discount)


@dataclass
class PaydayLoans(LoanData):
    """Container for preprocessed payday loan data.

    This class organizes initial state data for household payday loans. It processes:
    - Principal amounts from household data
    - Period rates from bank payday rates
    - Scheduled annuity payments based on maturity

    Note:
        This is a data container class. The actual loan behavior is implemented
        in the simulation package.
    """

    @classmethod
    def from_agent_data(
        cls, bank_data: pd.DataFrame, household_data: pd.DataFrame, payday_loan_maturity: int = 1
    ) -> "PaydayLoans":
        """Create a preprocessed payday loan data container.

        This method:
        1. Extracts household data
        2. Matches with bank rate data
        3. Calculates initial parameters

        Args:
            bank_data (pd.DataFrame): Bank data with rates
            household_data (pd.DataFrame): Household data
            payday_loan_maturity (int, optional): Initial maturity. Defaults to 1.

        Returns:
            PaydayLoans: Container with preprocessed loan data
        """
        principal = np.zeros((bank_data.shape[0], household_data.shape[0]))
        interest = np.zeros((bank_data.shape[0], household_data.shape[0]))
        discount = np.zeros((bank_data.shape[0], household_data.shape[0]))

        return cls(principal, interest, discount)


@dataclass
class MortgageLoans(LoanData):
    """Container for preprocessed mortgage loan data.

    This class organizes initial state data for household mortgages. It processes:
    - Principal amounts from household mortgage data
    - Period rates from bank mortgage rates
    - Scheduled annuity payments based on maturity

    Note:
        This is a data container class. The actual loan behavior is implemented
        in the simulation package.
    """

    @classmethod
    def from_agent_data(cls, bank_data: pd.DataFrame, household_data: pd.DataFrame, mortgage_maturity: int = 120):
        """Create a preprocessed mortgage loan data container.

        This method:
        1. Extracts household mortgage data
        2. Matches with bank rate data
        3. Calculates initial parameters

        Args:
            bank_data (pd.DataFrame): Bank data with rates
            household_data (pd.DataFrame): Household data with mortgages
            mortgage_maturity (int, optional): Initial maturity. Defaults to 120.

        Returns:
            MortgageLoans: Container with preprocessed loan data
        """
        household_mortgage_debt = household_data["Outstanding Balance of HMR Mortgages"].values
        households_corresponding_bank = household_data["Corresponding Bank ID"].values

        principal = np.zeros((bank_data.shape[0], household_mortgage_debt.shape[0]))
        interest = np.zeros((bank_data.shape[0], household_mortgage_debt.shape[0]))
        discount = np.zeros((bank_data.shape[0], household_mortgage_debt.shape[0]))

        for household_id in range(household_mortgage_debt.shape[0]):
            principal[households_corresponding_bank[household_id], household_id] = household_mortgage_debt[household_id]
            rate = bank_data["Interest Rates on Mortgages"].values[households_corresponding_bank[household_id]]
            interest[households_corresponding_bank[household_id], household_id] = rate
            discount[households_corresponding_bank[household_id], household_id] = annuity_payment(
                household_mortgage_debt[household_id],
                rate,
                mortgage_maturity,
            )

        return cls(principal, interest, discount)
