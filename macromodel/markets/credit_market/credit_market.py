"""Credit market implementation for macroeconomic agent-based model.

This module implements a sophisticated credit market system that manages lending relationships
between banks, firms, and households. It handles multiple types of loans including:

1. Firm Loans:
   - Short-term loans: Working capital, operational expenses
   - Long-term loans: Capital investment, expansion

2. Household Loans:
   - Consumption loans: Personal loans, credit lines
   - Mortgage loans: Home purchases, refinancing
   - Payday loans: Short-term emergency credit

The market clearing mechanism considers:
- Bank lending capacity and risk appetite
- Borrower creditworthiness and collateral
- Interest rate determination
- Non-performing loan (NPL) dynamics
- Regulatory constraints

Key Features:
- Multi-agent lending relationships
- Dynamic interest rate adjustment
- Risk-based credit allocation
- Loan lifecycle management
- Default handling
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Tuple

import h5py
import numpy as np

from macro_data import SyntheticCreditMarket
from macromodel.configurations import CreditMarketConfiguration
from macromodel.markets.credit_market.credit_market_ts import (
    create_credit_market_timeseries,
)
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import functions_from_model, update_functions

if TYPE_CHECKING:
    from macromodel.agents.banks.banks import Banks
    from macromodel.agents.firms import Firms
    from macromodel.agents.households.households import Households


_LOAN_KEYS = ("st_loans", "lt_loans", "cons_loans", "mort_loans")


def _zero_like_loan_states(states: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.zeros_like(states[key]) for key in _LOAN_KEYS}


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    copy = np.asarray(values, dtype=float).copy()
    copy.setflags(write=False)
    return copy


def _zero_like_firm_service_schedule(states: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.zeros_like(states[key][0]) for key in ("st_loans", "lt_loans")}


def _copy_firm_service_schedule(schedule: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: values.copy() for key, values in schedule.items()}


def _scheduled_service_components(
    loans: np.ndarray,
    opening_principal_arrears: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return scheduled interest and principal due for a loan state."""
    interest_due = loans[0] * loans[1]
    raw_principal_due = np.minimum(
        loans[0],
        np.maximum(loans[2] - interest_due, 0.0),
    )
    if opening_principal_arrears is None:
        principal_due = raw_principal_due
    else:
        principal_due = np.minimum(
            raw_principal_due,
            np.maximum(loans[0] - np.maximum(opening_principal_arrears, 0.0), 0.0),
        )
    return interest_due, principal_due


@dataclass(frozen=True)
class HouseholdServiceSnapshot:
    """Immutable opening-period household loan service due."""

    consumer_interest_due: np.ndarray
    consumer_principal_due: np.ndarray
    consumer_total_due: np.ndarray
    mortgage_interest_due: np.ndarray
    mortgage_principal_due: np.ndarray
    mortgage_total_due: np.ndarray
    opening_consumer_interest_arrears: np.ndarray
    opening_consumer_principal_arrears: np.ndarray
    newly_granted_consumer_loans: np.ndarray


@dataclass(frozen=True)
class ConsumerServiceArrears:
    """Opening and closing consumer service arrears by bank and household."""

    opening_interest: np.ndarray
    closing_interest: np.ndarray
    opening_principal: np.ndarray
    closing_principal: np.ndarray


@dataclass(frozen=True)
class ConsumerPaymentSettlement:
    """Authoritative current-period consumer payment outcome."""

    scheduled_service: np.ndarray
    actual_payment: np.ndarray
    unpaid_service: np.ndarray
    interest_paid: np.ndarray
    principal_paid: np.ndarray
    interest_paid_by_cell: np.ndarray
    principal_paid_by_cell: np.ndarray
    unpaid_interest_by_cell: np.ndarray
    unpaid_principal_by_cell: np.ndarray
    arrears: ConsumerServiceArrears


def _compute_credit_supply_caps_by_type(
    banks: "Banks",
    current_npl_firm_loans: float,
    current_npl_hh_cons_loans: float,
    current_npl_mortgages: float,
    credit_supply_temperature: float,
    total_target_short_term_credit: float,
    total_target_long_term_credit: float,
    total_ordinary_target_short_term_credit: float | None = None,
) -> dict[str, np.ndarray]:
    """Compute CAR-based bank credit supply caps (total + split by type).

    Total cap is lending headroom implied by the capital adequacy requirement:
        max_car = max(0, equity / capital_adequacy_ratio - total_outstanding_loans)

    Type caps follow the same preference-weighted split as credit clearing:
    initial `new_loans_fraction_*` shares adjusted by `credit_supply_temperature`
    and the economy-wide NPL rates.

    Firm short-term vs long-term caps are split using ordinary short-term firm
    demand when it is provided. Emergency overdraft-refinance demand remains part
    of total short-term borrower demand, but it is excluded from this cap split so
    old overdraft repair cannot mechanically starve long-term investment capacity.
    """
    max_car = np.maximum(
        0.0,
        banks.ts.current("equity") / banks.parameters.capital_adequacy_ratio
        - banks.ts.current("total_outstanding_loans"),
    )

    # Import locally to keep CreditMarket module lightweight and avoid circular-import risk.
    from macromodel.markets.credit_market.func.clearing import _compute_loan_type_preference_caps

    firm_caps, hh_cons_caps, mortgage_caps = _compute_loan_type_preference_caps(
        banks=banks,
        current_npl_firm_loans=current_npl_firm_loans,
        current_npl_hh_cons_loans=current_npl_hh_cons_loans,
        current_npl_mortgages=current_npl_mortgages,
        credit_supply_temperature=credit_supply_temperature,
    )

    if total_ordinary_target_short_term_credit is None:
        ordinary_target_short_term_credit = float(total_target_short_term_credit)
    else:
        ordinary_target_short_term_credit = max(0.0, float(total_ordinary_target_short_term_credit))

    denom = float(ordinary_target_short_term_credit + total_target_long_term_credit)
    if not np.isfinite(denom) or denom <= 0.0:
        short_term_share = 0.5
    else:
        short_term_share = float(ordinary_target_short_term_credit) / denom
        short_term_share = float(np.clip(short_term_share, 0.0, 1.0))

    firm_caps_short_term = firm_caps * short_term_share
    firm_caps_long_term = firm_caps * (1.0 - short_term_share)
    return {
        "total": max_car,
        "firms": firm_caps,
        "firms_short_term": firm_caps_short_term,
        "firms_long_term": firm_caps_long_term,
        "households_consumption": hh_cons_caps,
        "mortgages": mortgage_caps,
    }


def _ordinary_short_term_target_for_cap_split(
    target_short_term_credit: np.ndarray,
    ordinary_target_short_term_credit: np.ndarray,
    target_debt_rollover_credit: np.ndarray,
    target_overdraft_refinance_credit: np.ndarray,
) -> np.ndarray:
    """Return ordinary ST demand after excluding emergency ST buckets."""
    target_short_term_credit = np.asarray(target_short_term_credit, dtype=float)
    target_debt_rollover_credit = np.minimum(
        np.maximum(0.0, np.asarray(target_debt_rollover_credit, dtype=float)),
        np.maximum(0.0, target_short_term_credit),
    )
    target_overdraft_refinance_credit = np.minimum(
        np.maximum(0.0, np.asarray(target_overdraft_refinance_credit, dtype=float)),
        np.maximum(0.0, target_short_term_credit - target_debt_rollover_credit),
    )
    ordinary_target_short_term_credit = np.asarray(ordinary_target_short_term_credit, dtype=float)
    if (
        np.nansum(ordinary_target_short_term_credit) == 0.0
        and np.nansum(target_debt_rollover_credit) == 0.0
        and np.nansum(target_overdraft_refinance_credit) == 0.0
    ):
        ordinary_target_short_term_credit = target_short_term_credit.copy()
    return np.maximum(
        0.0,
        np.minimum(
            ordinary_target_short_term_credit,
            target_short_term_credit - target_debt_rollover_credit - target_overdraft_refinance_credit,
        ),
    )


def _append_credit_supply_caps_to_banks_ts(
    banks: "Banks",
    current_npl_firm_loans: float,
    current_npl_hh_cons_loans: float,
    current_npl_mortgages: float,
    credit_supply_temperature: float,
    total_target_short_term_credit: float,
    total_target_long_term_credit: float,
    total_ordinary_target_short_term_credit: float | None = None,
) -> None:
    """Append credit supply caps to `banks.ts`.

    The recorded firm short-term/long-term cap split reflects ordinary
    short-term demand versus long-term demand. Overdraft-refinance demand is
    tracked separately on firms and clears from residual capacity in the
    WaterBucket clearer.
    """
    caps = _compute_credit_supply_caps_by_type(
        banks=banks,
        current_npl_firm_loans=current_npl_firm_loans,
        current_npl_hh_cons_loans=current_npl_hh_cons_loans,
        current_npl_mortgages=current_npl_mortgages,
        credit_supply_temperature=credit_supply_temperature,
        total_target_short_term_credit=total_target_short_term_credit,
        total_target_long_term_credit=total_target_long_term_credit,
        total_ordinary_target_short_term_credit=total_ordinary_target_short_term_credit,
    )

    banks.ts.credit_supply_cap_total.append(caps["total"])
    banks.ts.credit_supply_cap_firms.append(caps["firms"])
    banks.ts.credit_supply_cap_firms_short_term.append(caps["firms_short_term"])
    banks.ts.credit_supply_cap_firms_long_term.append(caps["firms_long_term"])
    banks.ts.credit_supply_cap_households_consumption.append(caps["households_consumption"])
    banks.ts.credit_supply_cap_mortgages.append(caps["mortgages"])

    banks.ts.total_credit_supply_cap_total.append([float(np.nansum(caps["total"]))])
    banks.ts.total_credit_supply_cap_firms.append([float(np.nansum(caps["firms"]))])
    banks.ts.total_credit_supply_cap_firms_short_term.append([float(np.nansum(caps["firms_short_term"]))])
    banks.ts.total_credit_supply_cap_firms_long_term.append([float(np.nansum(caps["firms_long_term"]))])
    banks.ts.total_credit_supply_cap_households_consumption.append([float(np.nansum(caps["households_consumption"]))])
    banks.ts.total_credit_supply_cap_mortgages.append([float(np.nansum(caps["mortgages"]))])


class CreditMarket:
    """Credit market implementation managing lending relationships and loan lifecycles.

    This class implements the core credit market functionality, managing the interactions
    between financial institutions (banks) and borrowers (firms and households). It handles
    loan origination, servicing, repayment, and default processes.

    The market maintains state information about all outstanding loans including:
    - Principal amounts
    - Interest rates
    - Payment schedules
    - Default status

    Loan types are tracked in separate arrays with dimensions [3, n_banks, n_borrowers]:
    - Index 0: Outstanding principal
    - Index 1: Period interest rate
    - Index 2: Scheduled annuity payment

    Attributes:
        country_name (str): Name of the country this market operates in
        functions (dict[str, Any]): Market functions (clearing, pricing, etc.)
        ts (TimeSeries): Time series tracking market metrics
        states (dict[str, np.ndarray]): Current state of all loans
        initial_states (dict[str, np.ndarray]): Initial state snapshot for resets

    Example:
        >>> market = CreditMarket.from_data(
        ...     country_name="USA",
        ...     st_loans=short_term_loan_data,
        ...     lt_loans=long_term_loan_data,
        ...     cons_loans=consumer_loan_data,
        ...     mort_loans=mortgage_loan_data
        ... )
        >>> market.clear(banks, firms, households, npl_firm=0.02, npl_cons=0.03, npl_mort=0.01)
    """

    def __init__(
        self,
        country_name: str,
        functions: dict[str, Any],
        ts: TimeSeries,
        states: dict[str, np.ndarray],
        initial_states: dict[str, np.ndarray],
    ):
        """Initialize a new credit market instance.

        Args:
            country_name (str): Name of the country this market operates in
            functions (dict[str, Any]): Dictionary of market functions (clearing, etc.)
            ts (TimeSeries): Time series object for tracking market metrics
            states (dict[str, np.ndarray]): Current state of all loans
            initial_states (dict[str, np.ndarray]): Initial state snapshot for resets
        """
        self.country_name = country_name
        self.functions = functions
        self.ts = ts
        self.states = states
        self.initial_states = initial_states
        self._new_loans_this_period = _zero_like_loan_states(self.states)
        self._serviceable_loans_this_period = {key: self.states[key].copy() for key in _LOAN_KEYS}
        self._pending_consumer_loans_this_period: np.ndarray | None = None
        self._consumer_loan_remodulation_maturity: int | None = None
        self._consumer_interest_arrears_by_cell = np.zeros_like(self.states["cons_loans"][0])
        self._consumer_principal_arrears_by_cell = np.zeros_like(self.states["cons_loans"][0])
        self._household_service_snapshot: HouseholdServiceSnapshot | None = None
        self._last_consumer_settlement: ConsumerPaymentSettlement | None = None
        self._last_mortgage_principal_paid = np.zeros(self.states["mort_loans"].shape[2])
        self._last_mortgage_interest_paid = np.zeros(self.states["mort_loans"].shape[2])
        self._last_mortgage_interest_by_bank = np.zeros(self.states["mort_loans"].shape[1])
        self._last_consumer_opening_interest_collection_by_bank = np.zeros(self.states["cons_loans"].shape[1])
        self._last_consumer_accrued_interest_by_bank = np.zeros(self.states["cons_loans"].shape[1])
        self._firm_interest_arrears_by_cell = _zero_like_firm_service_schedule(self.states)
        self._firm_principal_arrears_by_cell = _zero_like_firm_service_schedule(self.states)
        self._reset_firm_service_period_tracking()

    def _reset_firm_service_period_tracking(self) -> None:
        self._last_interest_by_firm = np.zeros(self.states["st_loans"].shape[2])
        self._last_interest_by_household = np.zeros(self.states["cons_loans"].shape[2])
        self._last_household_interest_by_bank = np.zeros(self.states["cons_loans"].shape[1])
        self._last_firm_interest_by_bank = np.zeros(self.states["st_loans"].shape[1])
        self._last_interest_by_bank = np.zeros(self.states["st_loans"].shape[1])
        self._scheduled_firm_opening_interest_arrears_by_cell = _zero_like_firm_service_schedule(self.states)
        self._scheduled_firm_opening_principal_arrears_by_cell = _zero_like_firm_service_schedule(self.states)
        self._scheduled_firm_contractual_interest_due_by_cell = _zero_like_firm_service_schedule(self.states)
        self._scheduled_firm_contractual_principal_due_by_cell = _zero_like_firm_service_schedule(self.states)
        self._scheduled_firm_interest_due_by_cell = _zero_like_firm_service_schedule(self.states)
        self._scheduled_firm_principal_due_by_cell = _zero_like_firm_service_schedule(self.states)

    @classmethod
    def from_pickled_market(
        cls,
        synthetic_credit_market: SyntheticCreditMarket,
        credit_market_configuration: CreditMarketConfiguration,
        country_name: str,
    ) -> "CreditMarket":
        """Create a credit market instance from a pickled synthetic market.

        This factory method initializes a credit market from preprocessed synthetic data,
        which includes historical loan data and market configuration parameters.

        Args:
            synthetic_credit_market (SyntheticCreditMarket): Preprocessed market data
            credit_market_configuration (CreditMarketConfiguration): Market parameters
            country_name (str): Name of the country this market operates in

        Returns:
            CreditMarket: Initialized credit market instance with historical data

        Note:
            The synthetic market data includes:
            - Historical loan volumes by type
            - Default rates and loss history
            - Interest rate patterns
            - Bank-borrower relationships
        """
        functions = functions_from_model(
            credit_market_configuration.functions,
            loc="macromodel.markets.credit_market",
        )

        shortterm_loans = synthetic_credit_market.shortterm_loans.stack()
        longterm_loans = synthetic_credit_market.longterm_loans.stack()
        payday_loans = synthetic_credit_market.payday_loans.stack()
        consumption_expansion_loans = synthetic_credit_market.consumption_expansion_loans.stack()
        mortgage_loans = synthetic_credit_market.mortgage_loans.stack()

        ts = create_credit_market_timeseries(
            total_consumption_expansion_loans=consumption_expansion_loans[0].sum(),
            total_short_term_loans=shortterm_loans[0].sum(),
            total_long_term_loans=longterm_loans[0].sum(),
            total_mortgage_loans=mortgage_loans[0].sum(),
        )

        states = {
            "st_loans": shortterm_loans,
            "lt_loans": longterm_loans,
            "payday_loans": payday_loans,
            "cons_loans": consumption_expansion_loans,
            "mort_loans": mortgage_loans,
        }

        initial_states = deepcopy(states)

        return cls(
            country_name,
            functions,
            ts,
            states=states,
            initial_states=initial_states,
        )

    def reset(self, configuration: CreditMarketConfiguration) -> None:
        """Reset the credit market to its initial state.

        Restores all loan states to their initial values and updates market functions
        with the new configuration. This is useful for running multiple simulations
        or testing different scenarios.

        Args:
            configuration (CreditMarketConfiguration): New market configuration to use
        """
        self.states = deepcopy(self.initial_states)
        self.ts.reset()
        self._new_loans_this_period = _zero_like_loan_states(self.states)
        self._serviceable_loans_this_period = {key: self.states[key].copy() for key in _LOAN_KEYS}
        self._pending_consumer_loans_this_period = None
        self._consumer_loan_remodulation_maturity = None
        self._consumer_interest_arrears_by_cell = np.zeros_like(self.states["cons_loans"][0])
        self._consumer_principal_arrears_by_cell = np.zeros_like(self.states["cons_loans"][0])
        self._household_service_snapshot = None
        self._last_consumer_settlement = None
        self._last_mortgage_principal_paid = np.zeros(self.states["mort_loans"].shape[2])
        self._last_mortgage_interest_paid = np.zeros(self.states["mort_loans"].shape[2])
        self._last_mortgage_interest_by_bank = np.zeros(self.states["mort_loans"].shape[1])
        self._last_consumer_opening_interest_collection_by_bank = np.zeros(self.states["cons_loans"].shape[1])
        self._last_consumer_accrued_interest_by_bank = np.zeros(self.states["cons_loans"].shape[1])
        self._firm_interest_arrears_by_cell = _zero_like_firm_service_schedule(self.states)
        self._firm_principal_arrears_by_cell = _zero_like_firm_service_schedule(self.states)
        self._reset_firm_service_period_tracking()
        update_functions(model=configuration.functions, loc="macromodel.agents.credit_market", functions=self.functions)

    @classmethod
    def from_data(
        cls,
        country_name: str,
        st_loans: np.ndarray,
        lt_loans: np.ndarray,
        cons_loans: np.ndarray,
        mort_loans: np.ndarray,
    ) -> "CreditMarket":
        """Create a credit market instance directly from loan data arrays.

        This factory method provides a simpler way to initialize a credit market
        when you have direct access to loan data arrays rather than a synthetic market.

        Args:
            country_name (str): Name of the country this market operates in
            st_loans (np.ndarray): Short-term loan data [3, n_banks, n_firms]
            lt_loans (np.ndarray): Long-term loan data [3, n_banks, n_firms]
            cons_loans (np.ndarray): Consumer loan data [3, n_banks, n_households]
            mort_loans (np.ndarray): Mortgage loan data [3, n_banks, n_households]

        Returns:
            CreditMarket: Initialized credit market instance

        Note:
            Each loan array has shape [3, n_banks, n_borrowers] where:
            - Index 0: Outstanding principal
            - Index 1: Period interest rate
            - Index 2: Scheduled annuity payment
        """
        # Record the states of all loans
        states = {
            "st_loans": st_loans,
            "lt_loans": lt_loans,
            "cons_loans": cons_loans,
            "mort_loans": mort_loans,
        }

        # Create the corresponding time series object
        ts = create_credit_market_timeseries(
            total_short_term_loans=st_loans[0].sum(),
            total_long_term_loans=lt_loans[0].sum(),
            total_consumption_expansion_loans=cons_loans[0].sum(),
            total_mortgage_loans=mort_loans[0].sum(),
        )

        return cls(
            country_name=country_name,
            functions={},
            ts=ts,
            states=states,
            initial_states=deepcopy(states),
        )

    def clear(
        self,
        banks: Banks,
        firms: Firms,
        households: Households,
        current_npl_firm_loans: float,
        current_npl_hh_cons_loans: float,
        current_npl_mortgages: float,
    ) -> None:
        """Clear the credit market by matching loan supply with demand.

        This is the core market clearing function that:
        1. Evaluates new loan applications
        2. Determines credit allocation
        3. Updates loan states
        4. Records lending activity

        The clearing process considers:
        - Bank lending capacity and risk appetite
        - Borrower creditworthiness
        - Current NPL rates
        - Market conditions

        Args:
            banks (Banks): Banking sector agent
            firms (Firms): Corporate sector agents
            households (Households): Household sector agents
            current_npl_firm_loans (float): Current NPL rate for firm loans
            current_npl_hh_cons_loans (float): Current NPL rate for consumer loans
            current_npl_mortgages (float): Current NPL rate for mortgages

        Note:
            The function updates various time series metrics including:
            - New loan originations by type
            - Outstanding loan balances
            - Bank portfolio composition
        """
        self._new_loans_this_period = _zero_like_loan_states(self.states)
        self._serviceable_loans_this_period = {key: self.states[key].copy() for key in _LOAN_KEYS}
        self._pending_consumer_loans_this_period = None
        self._consumer_loan_remodulation_maturity = None
        self._household_service_snapshot = None
        self._last_consumer_settlement = None
        self._last_consumer_opening_interest_collection_by_bank = np.zeros(self.states["cons_loans"].shape[1])
        self._last_consumer_accrued_interest_by_bank = np.zeros(self.states["cons_loans"].shape[1])
        self._reset_firm_service_period_tracking()

        credit_supply_temperature = float(
            getattr(self.functions.get("clearing"), "credit_supply_temperature", 0.0) or 0.0
        )
        target_short_term_credit = firms.ts.current("target_short_term_credit")
        target_debt_rollover_credit = firms.ts.current("target_debt_rollover_credit")
        target_overdraft_refinance_credit = firms.ts.current("target_overdraft_refinance_credit")
        ordinary_target_short_term_credit = _ordinary_short_term_target_for_cap_split(
            target_short_term_credit=target_short_term_credit,
            ordinary_target_short_term_credit=firms.ts.current("ordinary_target_short_term_credit"),
            target_debt_rollover_credit=target_debt_rollover_credit,
            target_overdraft_refinance_credit=target_overdraft_refinance_credit,
        )
        total_target_short_term_credit = float(np.nansum(target_short_term_credit))
        total_ordinary_target_short_term_credit = float(np.nansum(ordinary_target_short_term_credit))
        total_target_long_term_credit = float(np.nansum(firms.ts.current("target_long_term_credit")))
        _append_credit_supply_caps_to_banks_ts(
            banks=banks,
            current_npl_firm_loans=current_npl_firm_loans,
            current_npl_hh_cons_loans=current_npl_hh_cons_loans,
            current_npl_mortgages=current_npl_mortgages,
            credit_supply_temperature=credit_supply_temperature,
            total_target_short_term_credit=total_target_short_term_credit,
            total_target_long_term_credit=total_target_long_term_credit,
            total_ordinary_target_short_term_credit=total_ordinary_target_short_term_credit,
        )

        # Clear the credit market
        (
            new_st_loans,
            new_lt_loans,
            new_cons_loans,
            new_mort_loans,
        ) = self.functions["clearing"].clear(
            banks=banks,
            firms=firms,
            households=households,
            current_npl_firm_loans=current_npl_firm_loans,
            current_npl_hh_cons_loans=current_npl_hh_cons_loans,
            current_npl_mortgages=current_npl_mortgages,
        )

        # Record the new loans. Slot 1 is a period rate, so it must be
        # principal-weighted instead of added as an interest cash-flow amount.
        self._add_new_loans("st_loans", new_st_loans)
        self._add_new_loans("lt_loans", new_lt_loans)
        if households.uses_feasibility_resolver:
            self._pending_consumer_loans_this_period = new_cons_loans.copy()
        else:
            self._add_new_loans("cons_loans", new_cons_loans)
        self._add_new_loans("mort_loans", new_mort_loans)

        # Calculate aggregates for firms
        firms.ts.received_short_term_credit.append(new_st_loans[0].sum(axis=0))
        firms.ts.total_received_short_term_credit.append([firms.ts.current("received_short_term_credit").sum()])
        received_debt_rollover_credit = getattr(
            self.functions["clearing"],
            "_last_received_debt_rollover_credit_by_firm",
            np.minimum(
                firms.ts.current("target_debt_rollover_credit"),
                firms.ts.current("received_short_term_credit"),
            ),
        )
        received_debt_rollover_credit = np.minimum(
            received_debt_rollover_credit,
            firms.ts.current("received_short_term_credit"),
        )
        received_overdraft_refinance_credit = getattr(
            self.functions["clearing"],
            "_last_received_overdraft_refinance_credit_by_firm",
            np.minimum(
                firms.ts.current("target_overdraft_refinance_credit"),
                np.maximum(0.0, firms.ts.current("received_short_term_credit") - received_debt_rollover_credit),
            ),
        )
        received_overdraft_refinance_credit = np.minimum(
            received_overdraft_refinance_credit,
            np.maximum(0.0, firms.ts.current("received_short_term_credit") - received_debt_rollover_credit),
        )
        received_ordinary_short_term_credit = np.maximum(
            0.0,
            firms.ts.current("received_short_term_credit")
            - received_debt_rollover_credit
            - received_overdraft_refinance_credit,
        )
        firms.ts.received_debt_rollover_credit.append(received_debt_rollover_credit)
        firms.ts.total_received_debt_rollover_credit.append([received_debt_rollover_credit.sum()])
        firms.ts.received_overdraft_refinance_credit.append(received_overdraft_refinance_credit)
        firms.ts.total_received_overdraft_refinance_credit.append([received_overdraft_refinance_credit.sum()])
        firms.ts.received_ordinary_short_term_credit.append(received_ordinary_short_term_credit)
        firms.ts.total_received_ordinary_short_term_credit.append([received_ordinary_short_term_credit.sum()])
        firms.ts.received_long_term_credit.append(new_lt_loans[0].sum(axis=0))
        firms.ts.total_received_long_term_credit.append([firms.ts.current("received_long_term_credit").sum()])
        firms.ts.received_credit.append(
            firms.ts.current("received_short_term_credit") + firms.ts.current("received_long_term_credit")
        )

        # Calculate aggregates for households
        households.ts.received_consumption_loans.append(new_cons_loans[0].sum(axis=0))
        households.ts.total_received_consumption_loans.append(
            [households.ts.current("received_consumption_loans").sum()]
        )
        households.ts.received_mortgages.append(new_mort_loans[0].sum(axis=0))
        households.ts.total_received_mortgages.append([households.ts.current("received_mortgages").sum()])

        # Update credit market aggregates
        self.ts.total_newly_loans_granted_firms_short_term.append(
            [firms.ts.current("received_short_term_credit").sum()]
        )
        self.ts.total_newly_loans_granted_firms_long_term.append([firms.ts.current("received_long_term_credit").sum()])
        self.ts.total_newly_loans_granted_households_consumption.append(
            [households.ts.current("received_consumption_loans").sum()]
        )
        self.ts.total_newly_loans_granted_mortgages.append([households.ts.current("received_mortgages").sum()])

        # Update fractions of types of loans granted by bank
        total_loans_by_bank = (
            self.states["st_loans"][0].sum(axis=1)
            + self.states["lt_loans"][0].sum(axis=1)
            + self.states["cons_loans"][0].sum(axis=1)
            + (new_cons_loans[0].sum(axis=1) if households.uses_feasibility_resolver else 0.0)
            + self.states["mort_loans"][0].sum(axis=1)
        )
        banks.ts.new_loans_fraction_firms.append(
            np.divide(
                self.states["st_loans"][0].sum(axis=1) + self.states["lt_loans"][0].sum(axis=1),
                total_loans_by_bank,
                out=np.zeros(banks.ts.current("n_banks")),
                where=total_loans_by_bank != 0.0,
            )
        )
        banks.ts.new_loans_fraction_hh_cons.append(
            np.divide(
                self.states["cons_loans"][0].sum(axis=1)
                + (new_cons_loans[0].sum(axis=1) if households.uses_feasibility_resolver else 0.0),
                total_loans_by_bank,
                out=np.zeros(banks.ts.current("n_banks")),
                where=total_loans_by_bank != 0.0,
            )
        )
        banks.ts.new_loans_fraction_mortgages.append(
            np.divide(
                self.states["mort_loans"][0].sum(axis=1),
                total_loans_by_bank,
                out=np.zeros(banks.ts.current("n_banks")),
                where=total_loans_by_bank != 0.0,
            )
        )

    def _add_new_loans(self, key: str, new_loans: np.ndarray) -> None:
        """Add new principal while preserving period-rate loan semantics."""
        self._new_loans_this_period[key] = new_loans.copy()
        loans = self.states[key]
        old_principal = loans[0].copy()
        old_rate_weighted_principal = old_principal * loans[1]
        new_principal = new_loans[0]
        total_principal = old_principal + new_principal

        loans[0] = total_principal
        loans[1] = np.divide(
            old_rate_weighted_principal + new_principal * new_loans[1],
            total_principal,
            out=np.zeros_like(total_principal),
            where=total_principal > 0.0,
        )
        loans[2] += new_loans[2]

    def pending_granted_consumption_loans(self) -> np.ndarray:
        """Return the unbooked bank-by-household consumer-credit grant matrix."""
        if self._pending_consumer_loans_this_period is None:
            raise RuntimeError("No unbooked consumer-credit settlement is available for this period.")
        return self._pending_consumer_loans_this_period[0].copy()

    def _consumer_service_components(
        self,
        loan_state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return opening consumer service components including arrears."""
        opening_interest_arrears = np.maximum(self._consumer_interest_arrears_by_cell, 0.0).copy()
        opening_principal_arrears = np.minimum(
            np.maximum(self._consumer_principal_arrears_by_cell, 0.0),
            np.maximum(loan_state[0], 0.0),
        )
        contractual_interest, contractual_principal = _scheduled_service_components(
            loan_state,
            opening_principal_arrears=opening_principal_arrears,
        )
        return (
            opening_interest_arrears,
            opening_principal_arrears,
            contractual_interest,
            contractual_principal,
        )

    def prepare_household_service_snapshot(self) -> HouseholdServiceSnapshot:
        """Capture immutable opening household service before deferred settlement."""
        if self._pending_consumer_loans_this_period is not None:
            raise RuntimeError("Consumer-credit settlement must be booked before preparing household service.")
        if self._household_service_snapshot is not None:
            return self._household_service_snapshot
        opening_consumer = self._serviceable_loans_this_period["cons_loans"].copy()
        opening_mortgage = self._serviceable_loans_this_period["mort_loans"].copy()
        for name, values in (
            ("consumer principal", opening_consumer[0]),
            ("consumer rate", opening_consumer[1]),
            ("consumer scheduled payment", opening_consumer[2]),
            ("mortgage principal", opening_mortgage[0]),
            ("mortgage rate", opening_mortgage[1]),
            ("mortgage scheduled payment", opening_mortgage[2]),
            ("consumer interest arrears", self._consumer_interest_arrears_by_cell),
            ("consumer principal arrears", self._consumer_principal_arrears_by_cell),
        ):
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise RuntimeError(f"Opening {name} must be finite and non-negative.")
        if np.any(self._consumer_principal_arrears_by_cell > opening_consumer[0]):
            raise RuntimeError("Opening consumer principal arrears cannot exceed contractual principal.")
        (
            opening_interest_arrears,
            opening_principal_arrears,
            contractual_interest,
            contractual_principal,
        ) = self._consumer_service_components(opening_consumer)
        mortgage_interest, mortgage_principal = _scheduled_service_components(opening_mortgage)
        for name, values in (
            ("consumer contractual interest", contractual_interest),
            ("consumer contractual principal", contractual_principal),
            ("mortgage interest", mortgage_interest),
            ("mortgage principal", mortgage_principal),
        ):
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise RuntimeError(f"Opening {name} must be finite and non-negative.")
        snapshot = HouseholdServiceSnapshot(
            consumer_interest_due=_readonly_copy(contractual_interest),
            consumer_principal_due=_readonly_copy(contractual_principal),
            consumer_total_due=_readonly_copy(
                (
                    opening_interest_arrears + opening_principal_arrears + contractual_interest + contractual_principal
                ).sum(axis=0)
            ),
            mortgage_interest_due=_readonly_copy(mortgage_interest),
            mortgage_principal_due=_readonly_copy(mortgage_principal),
            mortgage_total_due=_readonly_copy((mortgage_interest + mortgage_principal).sum(axis=0)),
            opening_consumer_interest_arrears=_readonly_copy(opening_interest_arrears),
            opening_consumer_principal_arrears=_readonly_copy(opening_principal_arrears),
            newly_granted_consumer_loans=_readonly_copy(self._new_loans_this_period["cons_loans"]),
        )
        self._household_service_snapshot = snapshot
        return snapshot

    def household_service_snapshot(self) -> HouseholdServiceSnapshot:
        """Return the current-period opening household service snapshot."""
        if self._household_service_snapshot is None:
            raise RuntimeError("Household service snapshot has not been prepared for this period.")
        return self._household_service_snapshot

    def consumer_payment_settlement(self) -> ConsumerPaymentSettlement:
        """Return the committed current-period consumer settlement carrier."""
        if self._last_consumer_settlement is None:
            raise RuntimeError("Consumer payment has not been settled for this period.")
        return self._last_consumer_settlement

    def settle_granted_consumption_loans(
        self,
        *,
        credit_granted: np.ndarray,
        granted_consumer_credit_by_bank_and_household: np.ndarray,
        consumer_loan_maturity: int,
    ) -> None:
        """Book and remodulate consumer debt from the settled Stage 6 carrier."""
        if self._pending_consumer_loans_this_period is None:
            raise RuntimeError("Consumer-credit settlement has already been booked or was not cleared this period.")
        pending_loans = self._pending_consumer_loans_this_period
        settlement = np.asarray(granted_consumer_credit_by_bank_and_household, dtype=float)
        granted = np.asarray(credit_granted, dtype=float)
        expected_households = self.states["cons_loans"].shape[2]
        if granted.shape != (expected_households,):
            raise ValueError(
                "credit_granted must contain exactly one value per household; "
                f"expected shape {(expected_households,)}, got {granted.shape}."
            )
        if settlement.shape != pending_loans[0].shape:
            raise ValueError(
                "granted_consumer_credit_by_bank_and_household must match the cleared bank-by-household "
                "consumer-credit shape."
            )
        if not np.all(np.isfinite(settlement)) or np.any(settlement < 0.0):
            raise RuntimeError("Consumer-credit settlement contains non-finite or negative granted principal.")
        if not np.allclose(settlement, pending_loans[0], rtol=1e-10, atol=1e-8):
            raise RuntimeError("Consumer-credit settlement differs from the cleared bank-by-household grant.")
        if not np.allclose(settlement.sum(axis=0), granted, rtol=1e-10, atol=1e-8):
            raise RuntimeError("Consumer-credit settlement does not reconcile with received_consumption_loans.")
        if (
            isinstance(consumer_loan_maturity, bool)
            or not isinstance(consumer_loan_maturity, (int, np.integer))
            or consumer_loan_maturity <= 0
        ):
            raise ValueError("consumer_loan_maturity must be positive for consumer-debt remodulation.")

        opening_principal = self._serviceable_loans_this_period["cons_loans"][0]
        if not np.allclose(self.states["cons_loans"][0], opening_principal, rtol=1e-10, atol=1e-8):
            raise RuntimeError("Consumer-credit principal was mutated before Stage 6 settlement.")
        self._add_new_loans("cons_loans", pending_loans)
        self._consumer_loan_remodulation_maturity = consumer_loan_maturity
        booked_principal = self.states["cons_loans"][0] - opening_principal
        if not np.allclose(booked_principal, settlement, rtol=1e-10, atol=1e-8):
            raise RuntimeError("Consumer-credit household liabilities and bank assets were not booked exactly once.")
        self._pending_consumer_loans_this_period = None

    def _remodulate_settled_consumer_loan_schedule(
        self,
        *,
        settled_loans: np.ndarray,
        consumer_loan_maturity: int,
    ) -> None:
        """Refinance newly borrowing households into one aggregate consumer-loan schedule."""
        from macromodel.markets.credit_market.func.clearing import _annuity_payment_factor

        loans = self.states["cons_loans"]
        settled_principal = settled_loans[0].sum(axis=0)
        remodulated = settled_principal > 0.0
        if not np.any(remodulated):
            return

        settled_rate = np.divide(
            (settled_loans[0] * settled_loans[1]).sum(axis=0),
            settled_principal,
            out=np.zeros_like(settled_principal),
            where=remodulated,
        )
        aggregate_principal = loans[0].sum(axis=0)
        aggregate_payment = aggregate_principal * _annuity_payment_factor(
            settled_rate,
            consumer_loan_maturity,
        )
        payment_shares = np.divide(
            loans[0],
            aggregate_principal[None, :],
            out=np.zeros_like(loans[0]),
            where=aggregate_principal[None, :] > 0.0,
        )

        loans[1][:, remodulated] = np.where(
            loans[0][:, remodulated] > 0.0,
            settled_rate[None, remodulated],
            0.0,
        )
        loans[2][:, remodulated] = payment_shares[:, remodulated] * aggregate_payment[None, remodulated]

    @staticmethod
    def _allocate_household_payment(
        due_by_cell: np.ndarray,
        payment_by_household: np.ndarray,
    ) -> np.ndarray:
        """Allocate a household payment pro rata across its bank cells."""
        due_by_cell = np.maximum(np.asarray(due_by_cell, dtype=float), 0.0)
        payment_by_household = np.maximum(np.asarray(payment_by_household, dtype=float), 0.0)
        total_due = due_by_cell.sum(axis=0)
        ratio = np.divide(
            payment_by_household,
            total_due,
            out=np.zeros_like(payment_by_household),
            where=total_due > 0.0,
        )
        return due_by_cell * ratio[None, :]

    def settle_deferred_consumer_service(
        self,
        remaining_subsistence_shortfall: np.ndarray,
    ) -> ConsumerPaymentSettlement:
        """Settle opening consumer service after post-labour affordability."""
        if self._last_consumer_settlement is not None:
            raise RuntimeError("Consumer service has already been settled for this period.")
        snapshot = self.household_service_snapshot()
        opening_interest = snapshot.opening_consumer_interest_arrears
        opening_principal = snapshot.opening_consumer_principal_arrears
        contractual_interest = snapshot.consumer_interest_due
        contractual_principal = snapshot.consumer_principal_due
        total_due = snapshot.consumer_total_due
        residual = np.asarray(remaining_subsistence_shortfall, dtype=float)
        if residual.shape != total_due.shape:
            raise ValueError(
                "remaining_subsistence_shortfall must match household service shape; "
                f"expected {total_due.shape}, got {residual.shape}."
            )
        if not np.all(np.isfinite(residual)) or np.any(residual < 0.0):
            raise ValueError("remaining_subsistence_shortfall must be finite and non-negative.")
        residual = residual.copy()
        unpaid = np.minimum(residual, total_due)
        actual = total_due - unpaid

        paid_opening_interest = self._allocate_household_payment(
            opening_interest,
            np.minimum(actual, opening_interest.sum(axis=0)),
        )
        remaining_payment = np.maximum(actual - paid_opening_interest.sum(axis=0), 0.0)
        paid_opening_principal = self._allocate_household_payment(
            opening_principal,
            np.minimum(remaining_payment, opening_principal.sum(axis=0)),
        )
        remaining_payment = np.maximum(remaining_payment - paid_opening_principal.sum(axis=0), 0.0)
        paid_contractual_interest = self._allocate_household_payment(
            contractual_interest,
            np.minimum(remaining_payment, contractual_interest.sum(axis=0)),
        )
        remaining_payment = np.maximum(remaining_payment - paid_contractual_interest.sum(axis=0), 0.0)
        paid_contractual_principal = self._allocate_household_payment(
            contractual_principal,
            np.minimum(remaining_payment, contractual_principal.sum(axis=0)),
        )

        interest_paid_by_cell = paid_opening_interest + paid_contractual_interest
        principal_paid_by_cell = paid_opening_principal + paid_contractual_principal
        unpaid_interest_by_cell = np.maximum(opening_interest - paid_opening_interest, 0.0) + np.maximum(
            contractual_interest - paid_contractual_interest, 0.0
        )
        unpaid_principal_by_cell = np.maximum(opening_principal - paid_opening_principal, 0.0) + np.maximum(
            contractual_principal - paid_contractual_principal, 0.0
        )

        loans = self.states["cons_loans"]
        opening_principal_state = self._serviceable_loans_this_period["cons_loans"][0]
        loans[0] = np.maximum(loans[0] - principal_paid_by_cell, 0.0)
        remaining_opening_principal = np.maximum(opening_principal_state - principal_paid_by_cell, 0.0)
        fully_repaid = np.isclose(remaining_opening_principal, 0.0, atol=1e-2)
        opening_schedule = self._serviceable_loans_this_period["cons_loans"][2]
        loans[2] = np.maximum(loans[2] - np.where(fully_repaid, opening_schedule, 0.0), 0.0)

        self._consumer_interest_arrears_by_cell = unpaid_interest_by_cell.copy()
        self._consumer_principal_arrears_by_cell = np.minimum(
            unpaid_principal_by_cell,
            np.maximum(loans[0], 0.0),
        )
        self._last_consumer_opening_interest_collection_by_bank = paid_opening_interest.sum(axis=1)
        self._last_consumer_accrued_interest_by_bank = np.maximum(
            contractual_interest - paid_contractual_interest,
            0.0,
        ).sum(axis=1)
        settlement = ConsumerPaymentSettlement(
            scheduled_service=total_due.copy(),
            actual_payment=actual.copy(),
            unpaid_service=unpaid.copy(),
            interest_paid=interest_paid_by_cell.sum(axis=0),
            principal_paid=principal_paid_by_cell.sum(axis=0),
            interest_paid_by_cell=interest_paid_by_cell.copy(),
            principal_paid_by_cell=principal_paid_by_cell.copy(),
            unpaid_interest_by_cell=unpaid_interest_by_cell.copy(),
            unpaid_principal_by_cell=unpaid_principal_by_cell.copy(),
            arrears=ConsumerServiceArrears(
                opening_interest=opening_interest.copy(),
                closing_interest=self._consumer_interest_arrears_by_cell.copy(),
                opening_principal=opening_principal.copy(),
                closing_principal=self._consumer_principal_arrears_by_cell.copy(),
            ),
        )
        self._last_consumer_settlement = settlement
        self._last_interest_by_household = self._last_mortgage_interest_paid + settlement.interest_paid
        self._last_household_interest_by_bank = self._last_mortgage_interest_by_bank + interest_paid_by_cell.sum(axis=1)
        self._last_interest_by_bank = self._last_firm_interest_by_bank + self._last_household_interest_by_bank
        if self._consumer_loan_remodulation_maturity is not None:
            self._remodulate_settled_consumer_loan_schedule(
                settled_loans=snapshot.newly_granted_consumer_loans,
                consumer_loan_maturity=self._consumer_loan_remodulation_maturity,
            )
            self._consumer_loan_remodulation_maturity = None
        return settlement

    def _service_loans(self, loan_keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Service loans that existed before current-quarter origination."""
        n_borrowers = self.states[loan_keys[0]].shape[2]
        n_banks = self.states[loan_keys[0]].shape[1]
        principal_paid_by_borrower = np.zeros(n_borrowers)
        interest_paid_by_borrower = np.zeros(n_borrowers)
        interest_paid_by_bank = np.zeros(n_banks)

        for key in loan_keys:
            loans = self.states[key]
            serviceable = self._serviceable_loans_this_period.get(key, loans).copy()
            current_new = self._new_loans_this_period.get(key, np.zeros_like(loans))

            serviceable_principal = np.minimum(serviceable[0], loans[0])
            interest_due = serviceable_principal * serviceable[1]
            principal_due = np.minimum(
                serviceable_principal,
                np.maximum(serviceable[2] - interest_due, 0.0),
            )

            loans[0] = np.maximum(loans[0] - principal_due, 0.0)

            remaining_serviceable_principal = np.maximum(serviceable_principal - principal_due, 0.0)
            fully_repaid = np.isclose(remaining_serviceable_principal, 0.0, atol=1e-2)
            loans[2] = np.maximum(loans[2] - np.where(fully_repaid, serviceable[2], 0.0), 0.0)

            rate_weighted_principal = remaining_serviceable_principal * serviceable[1] + current_new[0] * current_new[1]
            loans[1] = np.divide(
                rate_weighted_principal,
                loans[0],
                out=np.zeros_like(loans[0]),
                where=loans[0] > 0.0,
            )

            principal_paid_by_borrower += principal_due.sum(axis=0)
            interest_paid_by_borrower += interest_due.sum(axis=0)
            interest_paid_by_bank += interest_due.sum(axis=1)

            self._new_loans_this_period[key] = np.zeros_like(loans)
            self._serviceable_loans_this_period[key] = loans.copy()

        return principal_paid_by_borrower, interest_paid_by_borrower, interest_paid_by_bank

    def _scheduled_service_by_borrower_and_bank(
        self,
        loan_keys: tuple[str, ...],
        loan_states: dict[str, np.ndarray] | None = None,
        principal_arrears_by_key: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """Compute scheduled interest/principal due without mutating loan state."""
        if loan_states is None:
            loan_states = self.states

        n_borrowers = loan_states[loan_keys[0]].shape[2]
        n_banks = loan_states[loan_keys[0]].shape[1]
        interest_due_by_key: dict[str, np.ndarray] = {}
        principal_due_by_key: dict[str, np.ndarray] = {}
        principal_due_by_borrower = np.zeros(n_borrowers)
        interest_due_by_borrower = np.zeros(n_borrowers)
        interest_due_by_bank = np.zeros(n_banks)

        for key in loan_keys:
            opening_principal_arrears = None if principal_arrears_by_key is None else principal_arrears_by_key[key]
            interest_due, principal_due = _scheduled_service_components(
                loan_states[key],
                opening_principal_arrears=opening_principal_arrears,
            )
            interest_due_by_key[key] = interest_due
            principal_due_by_key[key] = principal_due
            principal_due_by_borrower += principal_due.sum(axis=0)
            interest_due_by_borrower += interest_due.sum(axis=0)
            interest_due_by_bank += interest_due.sum(axis=1)

        return (
            interest_due_by_key,
            principal_due_by_key,
            principal_due_by_borrower,
            interest_due_by_borrower,
            interest_due_by_bank,
        )

    def _scheduled_service_by_borrower(self, loan_keys: tuple[str, ...]) -> np.ndarray:
        """Compute next-period interest plus principal due on outstanding loans."""
        n_borrowers = self.states[loan_keys[0]].shape[2]
        service = np.zeros(n_borrowers)
        for key in loan_keys:
            loans = self.states[key]
            interest_due = loans[0] * loans[1]
            opening_interest_arrears = self._firm_interest_arrears_by_cell[key]
            opening_principal_arrears = self._firm_principal_arrears_by_cell[key]
            _, principal_due = _scheduled_service_components(
                loans,
                opening_principal_arrears=opening_principal_arrears,
            )
            service += (interest_due + opening_interest_arrears + principal_due + opening_principal_arrears).sum(axis=0)
        return service

    def compute_scheduled_debt_service_by_firm(self) -> np.ndarray:
        """Calculate next-period scheduled firm loan interest and principal service."""
        return self._scheduled_service_by_borrower(("st_loans", "lt_loans"))

    def _compute_firm_installment_buckets(
        self,
        loan_states: dict[str, np.ndarray] | None = None,
    ) -> dict[str, dict[str, np.ndarray] | np.ndarray]:
        """Return firm debt-service buckets with arrears-inclusive totals."""
        if loan_states is None:
            loan_states = self.states

        n_firms = loan_states["st_loans"].shape[2]
        (
            contractual_interest_due_by_key,
            contractual_principal_due_by_key,
            contractual_principal_due_by_firm,
            contractual_interest_due_by_firm,
            _,
        ) = self._scheduled_service_by_borrower_and_bank(
            ("st_loans", "lt_loans"),
            loan_states=loan_states,
            principal_arrears_by_key=self._firm_principal_arrears_by_cell,
        )
        opening_interest_arrears_by_key = _copy_firm_service_schedule(self._firm_interest_arrears_by_cell)
        opening_principal_arrears_by_key = _copy_firm_service_schedule(self._firm_principal_arrears_by_cell)
        opening_interest_arrears_by_firm = np.zeros(n_firms)
        opening_principal_arrears_by_firm = np.zeros(n_firms)
        for key in ("st_loans", "lt_loans"):
            opening_interest_arrears_by_firm += opening_interest_arrears_by_key[key].sum(axis=0)
            opening_principal_arrears_by_firm += opening_principal_arrears_by_key[key].sum(axis=0)

        scheduled_interest_due_by_key = {
            key: contractual_interest_due_by_key[key] + opening_interest_arrears_by_key[key]
            for key in ("st_loans", "lt_loans")
        }
        scheduled_principal_due_by_key = {
            key: contractual_principal_due_by_key[key] + opening_principal_arrears_by_key[key]
            for key in ("st_loans", "lt_loans")
        }
        return {
            "opening_interest_arrears_by_key": opening_interest_arrears_by_key,
            "opening_principal_arrears_by_key": opening_principal_arrears_by_key,
            "contractual_interest_due_by_key": contractual_interest_due_by_key,
            "contractual_principal_due_by_key": contractual_principal_due_by_key,
            "scheduled_interest_due_by_key": scheduled_interest_due_by_key,
            "scheduled_principal_due_by_key": scheduled_principal_due_by_key,
            "opening_interest_arrears_by_firm": opening_interest_arrears_by_firm,
            "opening_principal_arrears_by_firm": opening_principal_arrears_by_firm,
            "contractual_interest_due_by_firm": contractual_interest_due_by_firm,
            "contractual_principal_due_by_firm": contractual_principal_due_by_firm,
            "scheduled_interest_due_by_firm": opening_interest_arrears_by_firm + contractual_interest_due_by_firm,
            "scheduled_principal_due_by_firm": opening_principal_arrears_by_firm + contractual_principal_due_by_firm,
        }

    def compute_scheduled_firm_installments_preview(self) -> dict[str, np.ndarray]:
        """Return current-period firm debt-service buckets before credit clearing."""
        buckets = self._compute_firm_installment_buckets()
        return {
            "opening_interest_arrears": np.asarray(buckets["opening_interest_arrears_by_firm"], dtype=float).copy(),
            "opening_principal_arrears": np.asarray(buckets["opening_principal_arrears_by_firm"], dtype=float).copy(),
            "contractual_interest_due": np.asarray(buckets["contractual_interest_due_by_firm"], dtype=float).copy(),
            "contractual_principal_due": np.asarray(buckets["contractual_principal_due_by_firm"], dtype=float).copy(),
            "scheduled_interest_due": np.asarray(buckets["scheduled_interest_due_by_firm"], dtype=float).copy(),
            "scheduled_principal_due": np.asarray(buckets["scheduled_principal_due_by_firm"], dtype=float).copy(),
        }

    def schedule_firm_installments(self) -> dict[str, np.ndarray]:
        """Stage current-period firm loan service for later cash-feasible settlement."""
        buckets = self._compute_firm_installment_buckets(
            loan_states=self._serviceable_loans_this_period,
        )
        self._scheduled_firm_opening_interest_arrears_by_cell = _copy_firm_service_schedule(
            buckets["opening_interest_arrears_by_key"]
        )
        self._scheduled_firm_opening_principal_arrears_by_cell = _copy_firm_service_schedule(
            buckets["opening_principal_arrears_by_key"]
        )
        self._scheduled_firm_contractual_interest_due_by_cell = _copy_firm_service_schedule(
            buckets["contractual_interest_due_by_key"]
        )
        self._scheduled_firm_contractual_principal_due_by_cell = _copy_firm_service_schedule(
            buckets["contractual_principal_due_by_key"]
        )
        self._scheduled_firm_interest_due_by_cell = _copy_firm_service_schedule(
            buckets["scheduled_interest_due_by_key"]
        )
        self._scheduled_firm_principal_due_by_cell = _copy_firm_service_schedule(
            buckets["scheduled_principal_due_by_key"]
        )
        return {
            "opening_interest_arrears": np.asarray(buckets["opening_interest_arrears_by_firm"], dtype=float).copy(),
            "opening_principal_arrears": np.asarray(buckets["opening_principal_arrears_by_firm"], dtype=float).copy(),
            "contractual_interest_due": np.asarray(buckets["contractual_interest_due_by_firm"], dtype=float).copy(),
            "contractual_principal_due": np.asarray(buckets["contractual_principal_due_by_firm"], dtype=float).copy(),
            "scheduled_interest_due": np.asarray(buckets["scheduled_interest_due_by_firm"], dtype=float).copy(),
            "scheduled_principal_due": np.asarray(buckets["scheduled_principal_due_by_firm"], dtype=float).copy(),
        }

    def settle_firm_installments(
        self,
        payable_principal_by_firm: np.ndarray,
        payable_interest_by_firm: np.ndarray,
        *,
        overwrite_bank_interest: bool = False,
        payable_opening_interest_arrears_by_firm: np.ndarray | None = None,
        payable_contractual_interest_by_firm: np.ndarray | None = None,
        payable_opening_principal_arrears_by_firm: np.ndarray | None = None,
        payable_contractual_principal_by_firm: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply actual firm debt-service payments after cash-feasibility is known."""
        n_firms = self.states["st_loans"].shape[2]
        payable_principal_by_firm = np.asarray(payable_principal_by_firm, dtype=float).copy()
        payable_interest_by_firm = np.asarray(payable_interest_by_firm, dtype=float).copy()
        opening_interest_arrears_by_firm = np.zeros(n_firms)
        opening_principal_arrears_by_firm = np.zeros(n_firms)
        contractual_interest_due_by_firm = np.zeros(n_firms)
        contractual_principal_due_by_firm = np.zeros(n_firms)
        for key in ("st_loans", "lt_loans"):
            opening_interest_arrears_by_firm += self._scheduled_firm_opening_interest_arrears_by_cell[key].sum(axis=0)
            opening_principal_arrears_by_firm += self._scheduled_firm_opening_principal_arrears_by_cell[key].sum(axis=0)
            contractual_interest_due_by_firm += self._scheduled_firm_contractual_interest_due_by_cell[key].sum(axis=0)
            contractual_principal_due_by_firm += self._scheduled_firm_contractual_principal_due_by_cell[key].sum(axis=0)
        scheduled_interest_due_by_firm = opening_interest_arrears_by_firm + contractual_interest_due_by_firm
        scheduled_principal_due_by_firm = opening_principal_arrears_by_firm + contractual_principal_due_by_firm

        payable_principal_by_firm = np.minimum(
            np.maximum(payable_principal_by_firm, 0.0),
            scheduled_principal_due_by_firm,
        )
        payable_interest_by_firm = np.minimum(
            np.maximum(payable_interest_by_firm, 0.0),
            scheduled_interest_due_by_firm,
        )
        if payable_opening_interest_arrears_by_firm is None:
            payable_opening_interest_arrears_by_firm = np.minimum(
                opening_interest_arrears_by_firm,
                payable_interest_by_firm,
            )
        else:
            payable_opening_interest_arrears_by_firm = np.minimum(
                np.maximum(np.asarray(payable_opening_interest_arrears_by_firm, dtype=float), 0.0),
                np.minimum(opening_interest_arrears_by_firm, payable_interest_by_firm),
            )
        remaining_interest_capacity = np.maximum(
            0.0, payable_interest_by_firm - payable_opening_interest_arrears_by_firm
        )
        if payable_contractual_interest_by_firm is None:
            payable_contractual_interest_by_firm = np.minimum(
                contractual_interest_due_by_firm,
                remaining_interest_capacity,
            )
        else:
            payable_contractual_interest_by_firm = np.minimum(
                np.maximum(np.asarray(payable_contractual_interest_by_firm, dtype=float), 0.0),
                np.minimum(contractual_interest_due_by_firm, remaining_interest_capacity),
            )
        payable_interest_by_firm = payable_opening_interest_arrears_by_firm + payable_contractual_interest_by_firm

        if payable_opening_principal_arrears_by_firm is None:
            payable_opening_principal_arrears_by_firm = np.minimum(
                opening_principal_arrears_by_firm,
                payable_principal_by_firm,
            )
        else:
            payable_opening_principal_arrears_by_firm = np.minimum(
                np.maximum(np.asarray(payable_opening_principal_arrears_by_firm, dtype=float), 0.0),
                np.minimum(opening_principal_arrears_by_firm, payable_principal_by_firm),
            )
        remaining_principal_capacity = np.maximum(
            0.0,
            payable_principal_by_firm - payable_opening_principal_arrears_by_firm,
        )
        if payable_contractual_principal_by_firm is None:
            payable_contractual_principal_by_firm = np.minimum(
                contractual_principal_due_by_firm,
                remaining_principal_capacity,
            )
        else:
            payable_contractual_principal_by_firm = np.minimum(
                np.maximum(np.asarray(payable_contractual_principal_by_firm, dtype=float), 0.0),
                np.minimum(contractual_principal_due_by_firm, remaining_principal_capacity),
            )
        payable_principal_by_firm = payable_opening_principal_arrears_by_firm + payable_contractual_principal_by_firm

        principal_paid_by_firm = np.zeros(n_firms)
        interest_paid_by_firm = np.zeros(n_firms)
        interest_paid_by_bank = np.zeros(self.states["st_loans"].shape[1])

        for key in ("st_loans", "lt_loans"):
            loans = self.states[key]
            serviceable = self._serviceable_loans_this_period.get(key, loans).copy()
            current_new = self._new_loans_this_period.get(key, np.zeros_like(loans))
            opening_interest_due = self._scheduled_firm_opening_interest_arrears_by_cell[key]
            contractual_interest_due = self._scheduled_firm_contractual_interest_due_by_cell[key]
            opening_principal_due = self._scheduled_firm_opening_principal_arrears_by_cell[key]
            contractual_principal_due = self._scheduled_firm_contractual_principal_due_by_cell[key]

            opening_interest_ratio = np.divide(
                payable_opening_interest_arrears_by_firm,
                opening_interest_arrears_by_firm,
                out=np.zeros_like(payable_opening_interest_arrears_by_firm),
                where=opening_interest_arrears_by_firm > 0.0,
            )
            contractual_interest_ratio = np.divide(
                payable_contractual_interest_by_firm,
                contractual_interest_due_by_firm,
                out=np.zeros_like(payable_contractual_interest_by_firm),
                where=contractual_interest_due_by_firm > 0.0,
            )
            opening_principal_ratio = np.divide(
                payable_opening_principal_arrears_by_firm,
                opening_principal_arrears_by_firm,
                out=np.zeros_like(payable_opening_principal_arrears_by_firm),
                where=opening_principal_arrears_by_firm > 0.0,
            )
            contractual_principal_ratio = np.divide(
                payable_contractual_principal_by_firm,
                contractual_principal_due_by_firm,
                out=np.zeros_like(payable_contractual_principal_by_firm),
                where=contractual_principal_due_by_firm > 0.0,
            )

            opening_interest_paid = opening_interest_due * opening_interest_ratio[None, :]
            contractual_interest_paid = contractual_interest_due * contractual_interest_ratio[None, :]
            opening_principal_paid = opening_principal_due * opening_principal_ratio[None, :]
            contractual_principal_paid = contractual_principal_due * contractual_principal_ratio[None, :]
            interest_paid = opening_interest_paid + contractual_interest_paid
            principal_paid = opening_principal_paid + contractual_principal_paid
            serviceable_principal = np.minimum(serviceable[0], loans[0])
            capitalized_interest = np.maximum(0.0, opening_interest_due - opening_interest_paid)
            capitalized_interest += np.maximum(0.0, contractual_interest_due - contractual_interest_paid)

            loans[0] = np.maximum(loans[0] - principal_paid, 0.0)
            loans[0] += capitalized_interest

            remaining_serviceable_principal = np.maximum(serviceable_principal - principal_paid, 0.0)
            fully_repaid = np.isclose(remaining_serviceable_principal + capitalized_interest, 0.0, atol=1e-2)
            loans[2] = np.maximum(loans[2] - np.where(fully_repaid, serviceable[2], 0.0), 0.0)
            self._firm_interest_arrears_by_cell[key] = np.zeros_like(capitalized_interest)
            self._firm_principal_arrears_by_cell[key] = np.maximum(0.0, opening_principal_due - opening_principal_paid)
            self._firm_principal_arrears_by_cell[key] += np.maximum(
                0.0,
                contractual_principal_due - contractual_principal_paid,
            )
            self._firm_principal_arrears_by_cell[key] = np.minimum(self._firm_principal_arrears_by_cell[key], loans[0])
            self._firm_interest_arrears_by_cell[key][fully_repaid] = 0.0
            self._firm_principal_arrears_by_cell[key][fully_repaid] = 0.0

            rate_weighted_principal = (
                remaining_serviceable_principal * serviceable[1]
                + capitalized_interest * serviceable[1]
                + current_new[0] * current_new[1]
            )
            loans[1] = np.divide(
                rate_weighted_principal,
                loans[0],
                out=np.zeros_like(loans[0]),
                where=loans[0] > 0.0,
            )

            principal_paid_by_firm += principal_paid.sum(axis=0)
            interest_paid_by_firm += interest_paid.sum(axis=0)
            interest_paid_by_bank += interest_paid.sum(axis=1)

            self._new_loans_this_period[key] = np.zeros_like(loans)
            self._serviceable_loans_this_period[key] = loans.copy()

        self._last_interest_by_firm = interest_paid_by_firm
        if overwrite_bank_interest:
            self._last_firm_interest_by_bank = interest_paid_by_bank
        else:
            self._last_firm_interest_by_bank += interest_paid_by_bank
        self._last_interest_by_bank = self._last_firm_interest_by_bank + self._last_household_interest_by_bank
        self._scheduled_firm_opening_interest_arrears_by_cell = _zero_like_firm_service_schedule(self.states)
        self._scheduled_firm_opening_principal_arrears_by_cell = _zero_like_firm_service_schedule(self.states)
        self._scheduled_firm_contractual_interest_due_by_cell = _zero_like_firm_service_schedule(self.states)
        self._scheduled_firm_contractual_principal_due_by_cell = _zero_like_firm_service_schedule(self.states)
        self._scheduled_firm_interest_due_by_cell = _zero_like_firm_service_schedule(self.states)
        self._scheduled_firm_principal_due_by_cell = _zero_like_firm_service_schedule(self.states)
        return principal_paid_by_firm

    def pay_firm_installments(self) -> np.ndarray:
        """Process current-period principal payments on existing firm loans.

        Current-quarter originations are excluded from same-quarter service.
        Interest due is stored for `compute_interest_paid_by_firm`.
        """
        staged_installments = self.schedule_firm_installments()
        return self.settle_firm_installments(
            payable_principal_by_firm=staged_installments["scheduled_principal_due"],
            payable_interest_by_firm=staged_installments["scheduled_interest_due"],
            overwrite_bank_interest=True,
            payable_opening_interest_arrears_by_firm=staged_installments["opening_interest_arrears"],
            payable_contractual_interest_by_firm=staged_installments["contractual_interest_due"],
            payable_opening_principal_arrears_by_firm=staged_installments["opening_principal_arrears"],
            payable_contractual_principal_by_firm=staged_installments["contractual_principal_due"],
        )

    def pay_household_installments(self) -> np.ndarray:
        """Process current-period principal payments on existing household loans.

        Current-quarter originations are excluded from same-quarter service.
        Interest due is stored for `compute_interest_paid_by_household`.
        """
        if self._household_service_snapshot is not None:
            principal_paid, interest_paid, interest_by_bank = self._service_loans(("mort_loans",))
            self._last_mortgage_principal_paid = principal_paid.copy()
            self._last_mortgage_interest_paid = interest_paid.copy()
            self._last_mortgage_interest_by_bank = interest_by_bank.copy()
            self._last_interest_by_household = interest_paid.copy()
            self._last_household_interest_by_bank = interest_by_bank.copy()
            self._last_interest_by_bank = self._last_firm_interest_by_bank + self._last_household_interest_by_bank
            return principal_paid
        if self._pending_consumer_loans_this_period is not None:
            raise RuntimeError("Consumer-credit settlement must be booked before household loan servicing.")
        new_consumer_loans = self._new_loans_this_period["cons_loans"].copy()
        principal_paid, interest_paid, interest_by_bank = self._service_loans(("cons_loans", "mort_loans"))
        if self._consumer_loan_remodulation_maturity is not None:
            self._remodulate_settled_consumer_loan_schedule(
                settled_loans=new_consumer_loans,
                consumer_loan_maturity=self._consumer_loan_remodulation_maturity,
            )
            self._consumer_loan_remodulation_maturity = None
        self._last_interest_by_household = interest_paid
        self._last_household_interest_by_bank = interest_by_bank.copy()
        self._last_interest_by_bank = self._last_firm_interest_by_bank + self._last_household_interest_by_bank
        return principal_paid

    def remove_repaid_loans(self, loan_keys: tuple[str, ...] | None = None) -> None:
        """Clean up fully repaid loans from the market state.

        Identifies loans with near-zero balances (accounting for numerical precision)
        and removes them from the market state by zeroing out all their attributes.
        """
        if loan_keys is None:
            loan_keys = ("st_loans", "lt_loans", "cons_loans", "mort_loans")
        for key in loan_keys:
            loans = self.states[key]
            if key == "cons_loans":
                ind = np.isclose(loans[0], 0.0, atol=1e-2) & np.isclose(
                    self._consumer_interest_arrears_by_cell,
                    0.0,
                    atol=1e-2,
                )
            else:
                ind = np.isclose(loans[0], 0.0, atol=1e-2)
            loans[:, ind] = 0.0
            if key in ("st_loans", "lt_loans"):
                self._firm_interest_arrears_by_cell[key][ind] = 0.0
                self._firm_principal_arrears_by_cell[key][ind] = 0.0
            elif key == "cons_loans":
                self._consumer_principal_arrears_by_cell[ind] = 0.0
                self._consumer_interest_arrears_by_cell[ind] = 0.0

    def compute_aggregates(self) -> None:
        """Update aggregate loan statistics.

        Calculates and records total outstanding loan amounts by type:
        - Short-term firm loans
        - Long-term firm loans
        - Consumer loans
        - Mortgages
        """
        self.ts.total_outstanding_loans_granted_firms_short_term.append([self.states["st_loans"][0].sum()])
        self.ts.total_outstanding_loans_granted_firms_long_term.append([self.states["lt_loans"][0].sum()])
        self.ts.total_outstanding_loans_granted_households_consumption.append(
            [(self.states["cons_loans"][0] + self._consumer_interest_arrears_by_cell).sum()]
        )
        self.ts.total_outstanding_loans_granted_mortgages.append([self.states["mort_loans"][0].sum()])

    def compute_outstanding_short_term_loans_by_firm(self) -> np.ndarray:
        """Calculate total short-term loans for each firm.

        Returns:
            np.ndarray: Array of total short-term loan balances by firm
        """
        return self.states["st_loans"][0].sum(axis=0)

    def compute_outstanding_long_term_loans_by_firm(self) -> np.ndarray:
        """Calculate total long-term loans for each firm.

        Returns:
            np.ndarray: Array of total long-term loan balances by firm
        """
        return self.states["lt_loans"][0].sum(axis=0)

    def compute_outstanding_consumption_loans_by_household(self) -> np.ndarray:
        """Calculate total consumer loans for each household.

        Returns:
            np.ndarray: Array of total consumer loan balances by household
        """
        return self.states["cons_loans"][0].sum(axis=0) + self._consumer_interest_arrears_by_cell.sum(axis=0)

    def compute_outstanding_mortgages_by_household(self) -> np.ndarray:
        """Calculate total mortgage loans for each household.

        Returns:
            np.ndarray: Array of total mortgage balances by household
        """
        return self.states["mort_loans"][0].sum(axis=0)

    def compute_scheduled_mortgage_payments_by_household(self) -> np.ndarray:
        """Calculate scheduled mortgage service for each household."""
        return self.compute_scheduled_mortgage_service_by_household()

    def compute_scheduled_consumption_loan_payments_by_household(self) -> np.ndarray:
        """Calculate scheduled consumer-loan service for each household."""
        return self.compute_scheduled_consumer_service_by_household()

    def compute_outstanding_loans_by_bank(self) -> np.ndarray:
        """Calculate total loans outstanding for each bank.

        Returns:
            np.ndarray: Array of total loan balances by bank across all loan types
        """
        return (
            self.states["st_loans"][0].sum(axis=1)
            + self.states["lt_loans"][0].sum(axis=1)
            + self.states["cons_loans"][0].sum(axis=1)
            + self._consumer_interest_arrears_by_cell.sum(axis=1)
            + self.states["mort_loans"][0].sum(axis=1)
        )

    def compute_outstanding_short_term_firm_loans_by_bank(self) -> np.ndarray:
        """Calculate total short-term firm loans for each bank.

        Returns:
            np.ndarray: Array of short-term firm loan balances by bank
        """
        return self.states["st_loans"][0].sum(axis=1)

    def compute_outstanding_long_term_firm_loans_by_bank(self) -> np.ndarray:
        """Calculate total long-term firm loans for each bank.

        Returns:
            np.ndarray: Array of long-term firm loan balances by bank
        """
        return self.states["lt_loans"][0].sum(axis=1)

    def compute_outstanding_household_consumption_loans_by_bank(self) -> np.ndarray:
        """Calculate total consumer loans for each bank.

        Returns:
            np.ndarray: Array of consumer loan balances by bank
        """
        return self.states["cons_loans"][0].sum(axis=1) + self._consumer_interest_arrears_by_cell.sum(axis=1)

    def compute_scheduled_consumer_service_by_household(self) -> np.ndarray:
        """Return opening consumer service, including carried arrears."""
        if self._household_service_snapshot is not None:
            return self._household_service_snapshot.consumer_total_due.copy()
        if not np.any(self._consumer_interest_arrears_by_cell) and not np.any(self._consumer_principal_arrears_by_cell):
            return self.states["cons_loans"][2].sum(axis=0)
        (
            opening_interest_arrears,
            opening_principal_arrears,
            contractual_interest,
            contractual_principal,
        ) = self._consumer_service_components(self.states["cons_loans"])
        return (
            opening_interest_arrears + opening_principal_arrears + contractual_interest + contractual_principal
        ).sum(axis=0)

    def compute_scheduled_mortgage_service_by_household(self) -> np.ndarray:
        """Return opening mortgage service for the current period."""
        if self._household_service_snapshot is not None:
            return self._household_service_snapshot.mortgage_total_due.copy()
        mortgage = self.states["mort_loans"]
        return mortgage[2].sum(axis=0)

    def compute_recognized_interest_received_by_bank(self) -> np.ndarray:
        """Return cash loan interest adjusted for new consumer-interest accrual."""
        return (
            self._last_interest_by_bank
            - self._last_consumer_opening_interest_collection_by_bank
            + self._last_consumer_accrued_interest_by_bank
        )

    def compute_cash_distributable_loan_interest_by_bank(self) -> np.ndarray:
        """Return cash loan interest without treating accrual as distribution."""
        return self._last_interest_by_bank.copy()

    def compute_outstanding_mortgages_by_bank(self) -> np.ndarray:
        """Calculate total mortgage loans for each bank.

        Returns:
            np.ndarray: Array of mortgage balances by bank
        """
        return self.states["mort_loans"][0].sum(axis=1)

    def compute_interest_paid_by_firm(self) -> np.ndarray:
        """Calculate total interest paid by each firm.

        Returns:
            np.ndarray: Array of interest payments by firm across all loan types
        """
        return self._last_interest_by_firm

    def compute_interest_paid_by_household(self) -> np.ndarray:
        """Calculate total interest paid by each household.

        Returns:
            np.ndarray: Array of interest payments by household across all loan types
        """
        return self._last_interest_by_household

    def compute_interest_received_by_bank(self) -> np.ndarray:
        """Calculate total interest received by each bank.

        Returns:
            np.ndarray: Array of interest income by bank across all loan types
        """
        return self._last_interest_by_bank

    def compute_defaulted_firm_loan_writeoff_by_bank(self, default_flag: np.ndarray) -> np.ndarray:
        """Return firm loan principal to write off by bank before mutating loan books."""
        default_flag = np.asarray(default_flag, dtype=bool)
        if default_flag.size == 0 or not np.any(default_flag):
            return np.zeros(self.states["st_loans"].shape[1])
        return self.states["st_loans"][0][:, default_flag].sum(axis=1) + self.states["lt_loans"][0][
            :, default_flag
        ].sum(axis=1)

    def remove_loans_to_firm(self, firm_id: int | np.ndarray) -> float:
        """Remove all loans associated with specified firm(s).

        Used when firms default or exit the market. Returns the total amount written off.

        Args:
            firm_id (int | np.ndarray): ID(s) of firm(s) to remove loans for

        Returns:
            float: Total amount of loans written off
        """
        total_amount = self.states["st_loans"][0][:, firm_id].sum() + self.states["lt_loans"][0][:, firm_id].sum()
        self.states["st_loans"][:, :, firm_id] = 0.0
        self.states["lt_loans"][:, :, firm_id] = 0.0
        self._firm_interest_arrears_by_cell["st_loans"][:, firm_id] = 0.0
        self._firm_interest_arrears_by_cell["lt_loans"][:, firm_id] = 0.0
        self._firm_principal_arrears_by_cell["st_loans"][:, firm_id] = 0.0
        self._firm_principal_arrears_by_cell["lt_loans"][:, firm_id] = 0.0
        return total_amount

    def remove_loans_to_households(self, household_id: int | np.ndarray) -> Tuple[float, float]:
        """Remove all loans associated with specified household(s).

        Used when households default. Returns the total amounts written off by loan type.

        Args:
            household_id (int | np.ndarray): ID(s) of household(s) to remove loans for

        Returns:
            Tuple[float, float]: Total consumer loans and mortgages written off
        """
        cons_amount = self.states["cons_loans"][0][:, household_id].sum()
        mort_amount = self.states["mort_loans"][0][:, household_id].sum()
        self.states["cons_loans"][:, :, household_id] = 0.0
        self.states["mort_loans"][:, :, household_id] = 0.0
        return cons_amount, mort_amount

    def remove_loans_by_bank(self, bank_id: int | np.ndarray) -> None:
        """Remove all loans associated with specified bank(s).

        Used when banks fail or exit the market.

        Args:
            bank_id (int | np.ndarray): ID(s) of bank(s) to remove loans for
        """
        self.states["st_loans"][:, bank_id] = 0.0
        self.states["lt_loans"][:, bank_id] = 0.0
        self.states["cons_loans"][:, bank_id] = 0.0
        self.states["mort_loans"][:, bank_id] = 0.0
        self._firm_interest_arrears_by_cell["st_loans"][bank_id] = 0.0
        self._firm_interest_arrears_by_cell["lt_loans"][bank_id] = 0.0
        self._firm_principal_arrears_by_cell["st_loans"][bank_id] = 0.0
        self._firm_principal_arrears_by_cell["lt_loans"][bank_id] = 0.0

    def save_to_h5(self, group: h5py.Group):
        """Save credit market state to HDF5 file.

        Args:
            group (h5py.Group): HDF5 group to save data to
        """
        self.ts.write_to_h5("CM", group)
