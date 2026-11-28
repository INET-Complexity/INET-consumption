"""Household insolvency management implementation.

This module implements household insolvency handling through:
- Default detection and processing
- Debt restructuring mechanisms
- Asset liquidation procedures
- Bank interaction management

The implementation handles:
- Insolvency identification
- Credit market impacts
- Non-performing loan tracking
- Default rate calculations
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from macromodel.agents.banks.banks import Banks
from macromodel.agents.households.households import Households
from macromodel.markets.credit_market.credit_market import CreditMarket


class HouseholdInsolvencyHandler(ABC):
    """Abstract base class for household insolvency management.

    Defines interface for handling household defaults through:
    - Insolvency detection
    - Default processing
    - Credit market updates
    - Bank interactions
    """

    @abstractmethod
    def handle_insolvency(
        self,
        households: Households,
        banks: Banks,
        credit_market: CreditMarket,
        consumer_terminal_removal_exclusion: np.ndarray | None = None,
    ) -> Tuple[float, float, float]:
        """Process household insolvency cases.

        Args:
            households (Households): Household sector agent
            banks (Banks): Banking sector agent
            credit_market (CreditMarket): Credit market interface

        Returns:
            Tuple[float, float, float]: Default rate, NPL ratios for consumption
                loans and mortgages
        """
        pass


class DefaultHouseholdInsolvencyHandler(HouseholdInsolvencyHandler):
    """Default implementation of household insolvency management.

    Implements insolvency handling through:
    - Net wealth and deposit checks
    - Loan removal processing
    - NPL ratio calculations

    With ``consumer_insolvency_opens_ficp`` an insolvent household keeps its
    consumer loans and enters FICP in the next period (20-quarter ban,
    rescheduled repayment, residual forgiven with a bank loss at the end).
    Without it, consumer loans are removed with no bank loss (legacy).
    Mortgages are removed and booked as bank losses in both cases.
    """

    def __init__(self, consumer_insolvency_opens_ficp: bool = False):
        self.consumer_insolvency_opens_ficp = bool(consumer_insolvency_opens_ficp)
        self.last_ficp_trigger: np.ndarray | None = None

    def handle_insolvency(
        self,
        households: Households,
        banks: Banks,
        credit_market: CreditMarket,
        consumer_terminal_removal_exclusion: np.ndarray | None = None,
    ) -> Tuple[float, float, float]:
        """Process household defaults using default behavior.

        Handles insolvency through:
        - Identifying insolvent households
        - Processing loan removals
        - Calculating default metrics

        Args:
            households (Households): Household sector agent
            banks (Banks): Banking sector agent
            credit_market (CreditMarket): Credit market interface

        Returns:
            Tuple[float, float, float]: Default rate, NPL ratios for consumption
                loans and mortgages
        """
        insolvent_mask = np.logical_and(
            households.ts.current("net_wealth") < 0,
            households.ts.current("liquid_financial_assets") < 0,
        )
        insolvent_households = np.where(insolvent_mask)[0]
        if self.consumer_insolvency_opens_ficp:
            # Consumer loans stay with the household; FICP settles them.
            consumer_terminal_removal_exclusion = np.ones_like(credit_market.states["cons_loans"][0], dtype=bool)
            self.last_ficp_trigger = insolvent_mask & ~households.current_ficp_active()
        else:
            self.last_ficp_trigger = None
        bad_hh_cons_loans, bad_mortgages = credit_market.remove_loans_to_households(
            insolvent_households,
            consumer_exclusion=consumer_terminal_removal_exclusion,
        )

        # Calculate NPL ratios
        total_cons_loans = credit_market.ts.current("total_outstanding_loans_granted_households_consumption")[0]
        if total_cons_loans == 0.0:
            npl_hh_cons_loans = 0.0
        else:
            npl_hh_cons_loans = bad_hh_cons_loans / total_cons_loans
        if credit_market.ts.current("total_outstanding_loans_granted_mortgages")[0] == 0.0:
            npl_mortgages = 0.0
        else:
            npl_mortgages = bad_mortgages / credit_market.ts.current("total_outstanding_loans_granted_mortgages")[0]

        return (
            len(insolvent_households) / households.ts.current("n_households"),
            npl_hh_cons_loans,
            npl_mortgages,
        )
