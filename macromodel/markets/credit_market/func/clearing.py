"""Credit market clearing implementations.

This module provides different strategies for clearing the credit market, matching
lenders (banks) with borrowers (firms and households) while respecting various
constraints and priorities. It implements multiple clearing mechanisms:

1. Default Clearing:
   - Interest rate based matching
   - Risk assessment and constraints
   - Regulatory compliance

2. Poledna Clearing:
   - Simplified firm-focused clearing
   - Basic capital requirements
   - Risk-weighted allocation

3. Water Bucket Clearing:
   - Network flow based allocation
   - Priority-based lending
   - Minimum fill rates

Each clearing mechanism handles:
- Capital adequacy requirements
- Risk assessment criteria
- Interest rate determination
- Loan type preferences
- Default risk management
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
from numba import njit

from macromodel.agents.banks.banks import Banks
from macromodel.agents.firms import Firms
from macromodel.agents.households.households import Households
from macromodel.markets.credit_market.types_of_loans import LoanTypes


def _clip_nonneg(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values), np.maximum(0.0, values), 0.0)


def _nan_array(n: int) -> np.ndarray:
    return np.full(n, np.nan)


def _append_scalar(ts: object, key: str, value: float) -> None:
    getattr(ts, key).append([float(value)])


def _append_vector(ts: object, key: str, value: np.ndarray) -> None:
    getattr(ts, key).append(np.asarray(value, dtype=float))


# Binding reason enum (stored as float codes in HDF5)
_BINDING_NOT_BINDING = 0.0
_BINDING_NO_DEMAND = 1.0
_BINDING_COLLATERAL = 2.0
_BINDING_DSCR = 3.0
_BINDING_ROA = 4.0
_BINDING_ROE = 5.0
_BINDING_BANK_CAR = 6.0
_BINDING_BANK_PREF = 7.0
_BINDING_OTHER = 8.0


def _compute_firm_cfads(firms: Firms, window: int, haircut: float) -> np.ndarray:
    """Compute lagged cash flow available for firm debt service."""
    n_firms = firms.ts.current("n_firms")
    cfads_components = (
        "nominal_amount_sold_in_lcu",
        "total_wage",
        "nominal_amount_spent_in_lcu",
        "taxes_paid_on_production",
        "corporate_taxes_paid",
    )
    histories = {name: firms.ts.historic(name) for name in cfads_components}
    available_periods = min(window, *(len(history) for history in histories.values()))
    if available_periods <= 0:
        return np.zeros(n_firms)

    period_cfads = []
    for period in range(-available_periods, 0):
        cash_sales = np.asarray(histories["nominal_amount_sold_in_lcu"][period], dtype=float)
        goods_purchases = np.asarray(histories["nominal_amount_spent_in_lcu"][period], dtype=float).sum(axis=1)
        costs = (
            np.asarray(histories["total_wage"][period], dtype=float)
            + goods_purchases
            + np.asarray(histories["taxes_paid_on_production"][period], dtype=float)
            + np.asarray(histories["corporate_taxes_paid"][period], dtype=float)
        )

        period_value = cash_sales - costs
        valid_period = np.isfinite(cash_sales) & np.isfinite(costs) & np.isfinite(period_value)
        period_cfads.append(np.where(valid_period, np.maximum(0.0, period_value), 0.0))

    cfads = haircut * np.mean(np.vstack(period_cfads), axis=0)
    return np.where(np.isfinite(cfads), np.maximum(0.0, cfads), 0.0)


def _compute_loan_type_preference_caps(
    banks: Banks,
    current_npl_firm_loans: float,
    current_npl_hh_cons_loans: float,
    current_npl_mortgages: float,
    credit_supply_temperature: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute currency-valued bank supply caps from loan-type preference shares."""
    max_car = np.maximum(
        0.0,
        banks.ts.current("equity") / banks.parameters.capital_adequacy_ratio
        - banks.ts.current("total_outstanding_loans"),
    )
    firm_weights = banks.ts.initial("new_loans_fraction_firms") * np.exp(
        -credit_supply_temperature * current_npl_firm_loans
    )
    hh_cons_weights = banks.ts.initial("new_loans_fraction_hh_cons") * np.exp(
        -credit_supply_temperature * current_npl_hh_cons_loans
    )
    mortgage_weights = banks.ts.initial("new_loans_fraction_mortgages") * np.exp(
        -credit_supply_temperature * current_npl_mortgages
    )
    weights_sum = firm_weights + hh_cons_weights + mortgage_weights

    firm_caps = np.divide(
        max_car * firm_weights,
        weights_sum,
        out=np.zeros(max_car.shape),
        where=weights_sum != 0.0,
    )
    hh_cons_caps = np.divide(
        max_car * hh_cons_weights,
        weights_sum,
        out=np.zeros(max_car.shape),
        where=weights_sum != 0.0,
    )
    mortgage_caps = np.divide(
        max_car * mortgage_weights,
        weights_sum,
        out=np.zeros(max_car.shape),
        where=weights_sum != 0.0,
    )
    return firm_caps, hh_cons_caps, mortgage_caps


def _firm_dscr_underwriting_rate(banks_ir: np.ndarray, mode: str) -> float:
    """Return the borrower-level underwriting rate used for DSCR capacity."""
    if mode != "max_bank_rate":
        raise ValueError("Unknown firm DSCR underwriting rate mode", mode)
    finite_rates = np.asarray(banks_ir, dtype=float)
    finite_rates = finite_rates[np.isfinite(finite_rates)]
    if finite_rates.size == 0:
        return np.nan
    return max(0.0, finite_rates.max())


def _annuity_payment_factor(period_rate: float | np.ndarray, loan_maturity: int) -> float | np.ndarray:
    """Return fixed-payment debt service per unit principal."""
    if loan_maturity <= 0:
        if np.isscalar(period_rate):
            return np.nan
        return np.full_like(np.asarray(period_rate, dtype=float), np.nan, dtype=float)

    rate = np.asarray(period_rate, dtype=float)
    zero_rate_factor = 1.0 / loan_maturity
    factor = np.divide(
        rate,
        1.0 - np.power(1.0 + rate, -loan_maturity),
        out=np.full_like(rate, zero_rate_factor, dtype=float),
        where=np.isfinite(rate) & (rate > 0.0),
    )
    factor = np.where(np.isfinite(factor) & (factor > 0.0), factor, zero_rate_factor)
    if np.isscalar(period_rate):
        return float(factor)
    return factor


def _annuity_payment(principal: np.ndarray, period_rate: np.ndarray, loan_maturity: int) -> np.ndarray:
    """Return fixed scheduled annuity payments for loan principals."""
    return np.asarray(principal, dtype=float) * _annuity_payment_factor(period_rate, loan_maturity)


def _empty_loan_arrays(
    banks: Banks,
    firms: Firms,
    households: Households,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_banks = banks.ts.current("n_banks")
    n_firms = firms.ts.current("n_firms")
    n_households = households.ts.current("n_households")
    return (
        np.zeros((3, n_banks, n_firms)),
        np.zeros((3, n_banks, n_firms)),
        np.zeros((3, n_banks, n_households)),
        np.zeros((3, n_banks, n_households)),
    )


def _add_loan_to_array(
    loan_array: np.ndarray,
    bank_id: int,
    recipient_id: int,
    value: float,
    period_rate: float,
    maturity: int,
) -> None:
    """Aggregate a granted loan into one bank-borrower facility cell."""
    if value <= 0.0:
        return
    old_principal = loan_array[0, bank_id, recipient_id]
    total_principal = old_principal + value
    loan_array[1, bank_id, recipient_id] = (
        old_principal * loan_array[1, bank_id, recipient_id] + value * period_rate
    ) / total_principal
    loan_array[0, bank_id, recipient_id] = total_principal
    loan_array[2, bank_id, recipient_id] += value * _annuity_payment_factor(period_rate, maturity)


def _loan_frame_to_arrays(
    loans: pd.DataFrame,
    banks: Banks,
    firms: Firms,
    households: Households,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert legacy DataFrame loan outputs to the array loan contract."""
    st_loans, lt_loans, cons_loans, mort_loans = _empty_loan_arrays(banks, firms, households)
    if loans.empty:
        return st_loans, lt_loans, cons_loans, mort_loans

    for row in loans.itertuples(index=False):
        loan_type = row.loan_type
        bank_id = int(row.loan_bank_id)
        recipient_id = int(row.loan_recipient_id)
        if loan_type == LoanTypes.FIRM_SHORT_TERM_LOAN:
            loan_array = st_loans
        elif loan_type == LoanTypes.FIRM_LONG_TERM_LOAN:
            loan_array = lt_loans
        elif loan_type == LoanTypes.HOUSEHOLD_CONSUMPTION_LOAN:
            loan_array = cons_loans
        elif loan_type == LoanTypes.MORTGAGE:
            loan_array = mort_loans
        else:
            raise ValueError("Unknown loan type", loan_type)
        _add_loan_to_array(
            loan_array=loan_array,
            bank_id=bank_id,
            recipient_id=recipient_id,
            value=float(row.loan_value),
            period_rate=float(row.loan_interest_rate),
            maturity=int(row.loan_maturity),
        )
    return st_loans, lt_loans, cons_loans, mort_loans


def _compute_firm_dscr_capacity(
    firms: Firms,
    agents_with_demand: np.ndarray,
    min_dscr: float,
    cfads_window: int,
    cfads_haircut: float,
    underwriting_rate: float,
    loan_maturity: int,
    new_debt_service_by_firm: np.ndarray,
) -> np.ndarray:
    """Compute maximum firm loan principal allowed by DSCR capacity."""
    if not np.isfinite(underwriting_rate) or loan_maturity <= 0:
        return np.zeros(agents_with_demand.shape)

    incremental_service_rate = _annuity_payment_factor(underwriting_rate, loan_maturity)
    if incremental_service_rate <= 0 or not np.isfinite(incremental_service_rate):
        return np.zeros(agents_with_demand.shape)

    cfads = _compute_firm_cfads(firms, cfads_window, cfads_haircut)[agents_with_demand]
    existing_service = firms.ts.current("scheduled_debt_service")[agents_with_demand]
    new_service = new_debt_service_by_firm[agents_with_demand]

    valid_capacity = np.isfinite(cfads) & np.isfinite(existing_service) & np.isfinite(new_service)
    service_room = np.zeros(agents_with_demand.shape)
    service_room[valid_capacity] = np.maximum(
        0.0,
        cfads[valid_capacity] / min_dscr - existing_service[valid_capacity] - new_service[valid_capacity],
    )
    dscr_capacity = service_room / incremental_service_rate
    return np.where(np.isfinite(dscr_capacity), np.maximum(0.0, dscr_capacity), 0.0)


def _compute_firm_borrower_credit_room_for_type(
    banks: Banks,
    firms: Firms,
    loan_type: LoanTypes,
    new_credit_by_firm: np.ndarray,
    new_debt_service_by_firm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return borrower-side firm credit room without demand or bank-supply clipping."""
    n_firms = firms.ts.current("n_firms")
    agents = np.arange(n_firms)

    if loan_type == LoanTypes.FIRM_SHORT_TERM_LOAN:
        loan_maturity = banks.parameters.short_term_firm_loan_maturity
        banks_ir = banks.ts.current("interest_rates_on_short_term_firm_loans")
    elif loan_type == LoanTypes.FIRM_LONG_TERM_LOAN:
        loan_maturity = banks.parameters.long_term_firm_loan_maturity
        banks_ir = banks.ts.current("interest_rates_on_long_term_firm_loans")
    else:
        raise ValueError("Expected firm loan type", loan_type)

    collateral_cap = _clip_nonneg(
        banks.parameters.firm_loans_capital_stock_collateral_ratio * firms.ts.current("capital_inputs_stock_value")
        - firms.ts.current("debt")
        - new_credit_by_firm
        + np.minimum(0.0, firms.ts.current("deposits"))
    )

    roa_cap = np.full(n_firms, np.inf)
    if banks.parameters.enable_firm_loans_return_on_assets_restriction:
        firm_expected_profits = firms.ts.current("expected_profits")
        firm_capital_stock = firms.ts.current("capital_inputs_stock_value")
        firm_roa = np.divide(
            firm_expected_profits,
            firm_capital_stock,
            out=np.zeros_like(firm_expected_profits),
            where=firm_capital_stock != 0.0,
        )
        roa_cap[firm_roa < banks.parameters.firm_loans_return_on_assets_ratio] = 0.0

    roe_cap = np.full(n_firms, np.inf)
    if banks.parameters.enable_firm_loans_return_on_equity_restriction:
        roe_cap = _clip_nonneg(
            firms.ts.current("capital_inputs_stock_value")
            + firms.ts.current("deposits")
            - firms.ts.current("debt")
            - new_credit_by_firm
            - firms.ts.current("expected_profits") / banks.parameters.firm_loans_return_on_equity_ratio
        )

    dscr_cap = np.full(n_firms, np.inf)
    if banks.parameters.enable_firm_loans_dscr_restriction:
        underwriting_rate = _firm_dscr_underwriting_rate(
            banks_ir,
            banks.parameters.firm_loans_dscr_underwriting_rate_mode,
        )
        dscr_cap = _clip_nonneg(
            _compute_firm_dscr_capacity(
                firms=firms,
                agents_with_demand=agents,
                min_dscr=banks.parameters.firm_loans_min_dscr,
                cfads_window=banks.parameters.firm_loans_cfads_window,
                cfads_haircut=banks.parameters.firm_loans_cfads_haircut,
                underwriting_rate=underwriting_rate,
                loan_maturity=loan_maturity,
                new_debt_service_by_firm=new_debt_service_by_firm,
            )
        )

    room = _clip_nonneg(np.minimum.reduce((collateral_cap, roa_cap, roe_cap, dscr_cap)))
    return room, collateral_cap, dscr_cap


def compute_firm_borrower_credit_room(
    banks: Banks,
    firms: Firms,
    allow_short_term_firm_loans: bool,
) -> dict[str, np.ndarray]:
    """Compute diagnostic borrower-side firm credit room, independent of bank supply.

    The helper mirrors borrower constraints used by credit clearing, but it does
    not allocate loans, consume priorities, apply bank-supply scarcity, or clip
    room by current target-credit demand.
    """
    n_firms = firms.ts.current("n_firms")
    new_credit_by_firm = np.zeros(n_firms)
    new_debt_service_by_firm = np.zeros(n_firms)

    if allow_short_term_firm_loans:
        st_room, st_collateral_cap, st_dscr_cap = _compute_firm_borrower_credit_room_for_type(
            banks=banks,
            firms=firms,
            loan_type=LoanTypes.FIRM_SHORT_TERM_LOAN,
            new_credit_by_firm=new_credit_by_firm,
            new_debt_service_by_firm=new_debt_service_by_firm,
        )
        new_credit_by_firm += st_room
        st_underwriting_rate = _firm_dscr_underwriting_rate(
            banks.ts.current("interest_rates_on_short_term_firm_loans"),
            banks.parameters.firm_loans_dscr_underwriting_rate_mode,
        )
        st_annuity_factor = _annuity_payment_factor(
            st_underwriting_rate,
            banks.parameters.short_term_firm_loan_maturity,
        )
        new_debt_service_by_firm += st_room * st_annuity_factor
    else:
        st_room = np.zeros(n_firms)
        st_collateral_cap = np.zeros(n_firms)
        st_dscr_cap = _nan_array(n_firms)

    lt_room, lt_collateral_cap, lt_dscr_cap = _compute_firm_borrower_credit_room_for_type(
        banks=banks,
        firms=firms,
        loan_type=LoanTypes.FIRM_LONG_TERM_LOAN,
        new_credit_by_firm=new_credit_by_firm,
        new_debt_service_by_firm=new_debt_service_by_firm,
    )

    return {
        "short_term": st_room,
        "long_term": lt_room,
        "total": st_room + lt_room,
        "short_term_collateral_cap": st_collateral_cap,
        "short_term_dscr_cap": st_dscr_cap,
        "long_term_collateral_cap": lt_collateral_cap,
        "long_term_dscr_cap": lt_dscr_cap,
    }


class CreditMarketClearer(ABC):
    """Abstract base class for credit market clearing mechanisms.

    This class defines the interface and common functionality for all credit market
    clearing implementations. It handles the matching of lenders (banks) with
    borrowers (firms and households) while respecting various financial constraints
    and regulatory requirements.

    The clearing process follows these general steps:
    1. Assessment of bank lending capacity
    2. Evaluation of borrower creditworthiness
    3. Interest rate determination
    4. Loan allocation and origination

    Attributes:
        allow_short_term_firm_loans (bool): Whether to allow short-term lending to firms
        allow_household_loans (bool): Whether to allow lending to households
        firms_max_number_of_banks_visiting (int): Max banks a firm can approach
        households_max_number_of_banks_visiting (int): Max banks a household can approach
        consider_loan_type_fractions (bool): Whether to use loan type preferences
        credit_supply_temperature (float): Sensitivity to NPL rates in supply
        interest_rates_selection_temperature (float): Rate sensitivity in matching
        creditor_selection_is_deterministic (bool): Whether to use deterministic matching
        creditor_minimum_fill (bool): Whether to enforce minimum lending amounts
        debtor_minimum_fill (bool): Whether to enforce minimum borrowing amounts
    """

    def __init__(
        self,
        allow_short_term_firm_loans: bool,
        allow_household_loans: bool,
        firms_max_number_of_banks_visiting: int,
        households_max_number_of_banks_visiting: int,
        consider_loan_type_fractions: bool,
        credit_supply_temperature: float,
        interest_rates_selection_temperature: float,
        creditor_selection_is_deterministic: bool,
        creditor_minimum_fill: bool,
        debtor_minimum_fill: bool,
    ):
        """Initialize market clearer with configuration parameters.

        Args:
            allow_short_term_firm_loans (bool): Allow short-term firm lending
            allow_household_loans (bool): Allow household lending
            firms_max_number_of_banks_visiting (int): Max banks per firm
            households_max_number_of_banks_visiting (int): Max banks per household
            consider_loan_type_fractions (bool): Use loan type preferences
            credit_supply_temperature (float): NPL sensitivity parameter
            interest_rates_selection_temperature (float): Rate sensitivity
            creditor_selection_is_deterministic (bool): Use deterministic matching
            creditor_minimum_fill (bool): Enforce minimum lending
            debtor_minimum_fill (bool): Enforce minimum borrowing
        """
        self.allow_short_term_firm_loans = allow_short_term_firm_loans
        self.allow_household_loans = allow_household_loans
        self.firms_max_number_of_banks_visiting = firms_max_number_of_banks_visiting
        self.households_max_number_of_banks_visiting = households_max_number_of_banks_visiting
        self.consider_loan_type_fractions = consider_loan_type_fractions
        self.credit_supply_temperature = credit_supply_temperature
        self.interest_rates_selection_temperature = interest_rates_selection_temperature
        self.creditor_selection_is_deterministic = creditor_selection_is_deterministic
        self.creditor_minimum_fill = creditor_minimum_fill
        self.debtor_minimum_fill = debtor_minimum_fill

    @staticmethod
    @abstractmethod
    def clear(
        banks: Banks,
        firms: Firms,
        households: Households,
        current_npl_firm_loans: float,
        current_npl_hh_cons_loans: float,
        current_npl_mortgages: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Execute market clearing algorithm.

        Abstract method that must be implemented by concrete clearers to match
        lenders with borrowers and originate loans.

        Args:
            banks (Banks): Banking sector agent
            firms (Firms): Corporate sector agents
            households (Households): Household sector agents
            current_npl_firm_loans (float): Current NPL rate for firm loans
            current_npl_hh_cons_loans (float): Current NPL rate for consumer loans
            current_npl_mortgages (float): Current NPL rate for mortgages

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: New loan arrays
                for short-term firm, long-term firm, household consumption, and
                mortgage loans.
        """
        pass


class NoCreditMarketClearer(CreditMarketClearer):
    """Null implementation of credit market clearing.

    This implementation does nothing during clearing, effectively creating
    a market with no lending. Useful for testing and debugging.
    """

    @staticmethod
    def clear(
        banks: Banks,
        firms: Firms,
        households: Households,
        current_npl_firm_loans: float,
        current_npl_hh_cons_loans: float,
        current_npl_mortgages: float,
    ) -> pd.DataFrame:
        """No-op implementation of market clearing.

        Returns an empty DataFrame with the expected loan columns.

        Args:
            banks (Banks): Unused
            firms (Firms): Unused
            households (Households): Unused
            current_npl_firm_loans (float): Unused
            current_npl_hh_cons_loans (float): Unused
            current_npl_mortgages (float): Unused

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Empty loan arrays
        """
        return _empty_loan_arrays(banks, firms, households)


class DefaultCreditMarketClearer(CreditMarketClearer):
    """Default implementation of credit market clearing.

    This implementation uses an interest rate based matching algorithm with risk
    assessment and regulatory constraints. For each loan type, it:
    1. Evaluates bank lending capacity considering capital requirements
    2. Assesses borrower creditworthiness using various metrics
    3. Matches borrowers with banks based on interest rates
    4. Originates loans respecting all constraints

    Key Features:
    - Multiple loan types (short-term, long-term, consumer, mortgage)
    - Risk-based lending limits
    - Interest rate based matching
    - Regulatory compliance checks
    - NPL-sensitive credit supply
    """

    def clear(
        self,
        banks: Banks,
        firms: Firms,
        households: Households,
        current_npl_firm_loans: float,
        current_npl_hh_cons_loans: float,
        current_npl_mortgages: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Execute the default market clearing algorithm.

        Processes each loan type sequentially, matching borrowers with lenders
        based on interest rates and various constraints.

        Args:
            banks (Banks): Banking sector agent
            firms (Firms): Corporate sector agents
            households (Households): Household sector agents
            current_npl_firm_loans (float): Current NPL rate for firm loans
            current_npl_hh_cons_loans (float): Current NPL rate for consumer loans
            current_npl_mortgages (float): Current NPL rate for mortgages

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: New loan arrays

        Note:
            The clearing process:
            1. Updates bank lending preferences based on NPL rates
            2. Clears firm loans (short-term then long-term)
            3. Clears household loans (consumer then mortgage)
            4. Combines and returns all new loans
        """
        empty = pd.DataFrame(
            data={
                "loan_type": [],
                "loan_value_initial": [],
                "loan_value": [],
                "loan_maturity": [],
                "loan_interest_rate": [],
                "loan_bank_id": [],
                "loan_recipient_id": [],
            }
        )

        # Keeping track of new credit
        new_credit_by_bank = np.zeros(banks.ts.current("n_banks"))
        new_credit_by_firm = np.zeros(firms.ts.current("n_firms"))
        new_credit_by_household = np.zeros(households.ts.current("n_households"))
        new_firm_preference_credit_by_bank = np.zeros(banks.ts.current("n_banks"))
        new_household_consumption_preference_credit_by_bank = np.zeros(banks.ts.current("n_banks"))
        new_mortgage_preference_credit_by_bank = np.zeros(banks.ts.current("n_banks"))

        # Bank-side diagnostic rooms/caps at the start of clearing
        car_room_initial = np.maximum(
            0.0,
            banks.ts.current("equity") / banks.parameters.capital_adequacy_ratio
            - banks.ts.current("total_outstanding_loans"),
        )
        _append_scalar(banks.ts, "credit_market_car_room_initial", float(car_room_initial.sum()))

        # Firm CFADS proxy (only meaningful when DSCR is enabled)
        if banks.parameters.enable_firm_loans_dscr_restriction:
            cfads = _compute_firm_cfads(
                firms,
                window=banks.parameters.firm_loans_cfads_window,
                haircut=banks.parameters.firm_loans_cfads_haircut,
            )
        else:
            cfads = _nan_array(firms.ts.current("n_firms"))
        _append_vector(firms.ts, "credit_market_firm_cfads", cfads)

        # Banks may update their preferences for different types of loans, impacting their supply
        if self.consider_loan_type_fractions:
            (
                max_supply_based_on_preferences_firms,
                max_supply_based_on_preferences_hh_cons,
                max_supply_based_on_preferences_mortgages,
            ) = _compute_loan_type_preference_caps(
                banks,
                current_npl_firm_loans,
                current_npl_hh_cons_loans,
                current_npl_mortgages,
                self.credit_supply_temperature,
            )
            _append_scalar(
                banks.ts,
                "credit_market_pref_cap_firms_initial",
                float(np.sum(max_supply_based_on_preferences_firms)),
            )
            _append_scalar(
                banks.ts,
                "credit_market_pref_cap_hh_cons_initial",
                float(np.sum(max_supply_based_on_preferences_hh_cons)),
            )
            _append_scalar(
                banks.ts,
                "credit_market_pref_cap_mortgages_initial",
                float(np.sum(max_supply_based_on_preferences_mortgages)),
            )
        else:
            max_supply_based_on_preferences_firms = np.full(banks.ts.current("n_banks"), np.inf)
            max_supply_based_on_preferences_hh_cons = np.full(banks.ts.current("n_banks"), np.inf)
            max_supply_based_on_preferences_mortgages = np.full(banks.ts.current("n_banks"), np.inf)
            _append_scalar(banks.ts, "credit_market_pref_cap_firms_initial", np.nan)
            _append_scalar(banks.ts, "credit_market_pref_cap_hh_cons_initial", np.nan)
            _append_scalar(banks.ts, "credit_market_pref_cap_mortgages_initial", np.nan)

        # Firm loans
        if self.allow_short_term_firm_loans:
            new_short_term_firm_loans = self.clear_firm_loans(
                banks=banks,
                firms=firms,
                loan_type=LoanTypes.FIRM_SHORT_TERM_LOAN,
                new_credit_by_bank=new_credit_by_bank,
                new_credit_by_firm=new_credit_by_firm,
                max_supply_based_on_preferences=max_supply_based_on_preferences_firms,
                new_preference_credit_by_bank=new_firm_preference_credit_by_bank,
            )
        else:
            new_short_term_firm_loans = empty.copy()
        new_long_term_firm_loans = self.clear_firm_loans(
            banks=banks,
            firms=firms,
            loan_type=LoanTypes.FIRM_LONG_TERM_LOAN,
            new_credit_by_bank=new_credit_by_bank,
            new_credit_by_firm=new_credit_by_firm,
            max_supply_based_on_preferences=max_supply_based_on_preferences_firms,
            new_preference_credit_by_bank=new_firm_preference_credit_by_bank,
        )

        # Household loans
        if self.allow_household_loans:
            new_household_consumption_loans = self.clear_household_consumption_loans(
                banks=banks,
                households=households,
                new_credit_by_bank=new_credit_by_bank,
                new_credit_by_household=new_credit_by_household,
                max_supply_based_on_preferences=max_supply_based_on_preferences_hh_cons,
                new_preference_credit_by_bank=new_household_consumption_preference_credit_by_bank,
            )
            new_mortgages = self.clear_mortgages(
                banks=banks,
                households=households,
                loan_type=LoanTypes.MORTGAGE,
                new_credit_by_bank=new_credit_by_bank,
                new_credit_by_household=new_credit_by_household,
                max_supply_based_on_preferences=max_supply_based_on_preferences_mortgages,
                new_preference_credit_by_bank=new_mortgage_preference_credit_by_bank,
            )
        else:
            new_household_consumption_loans = empty.copy()
            new_mortgages = empty.copy()

        # Collect them all
        new_loans = pd.concat(
            (
                new_short_term_firm_loans,
                new_long_term_firm_loans,
                new_household_consumption_loans,
                new_mortgages,
            ),
            axis=0,
        ).reset_index(drop=True)
        new_loans["loan_bank_id"] = new_loans["loan_bank_id"].astype(int)
        new_loans["loan_recipient_id"] = new_loans["loan_recipient_id"].astype(int)

        return _loan_frame_to_arrays(new_loans, banks, firms, households)

    def clear_firm_loans(
        self,
        banks: Banks,
        firms: Firms,
        loan_type: LoanTypes,
        new_credit_by_bank: np.ndarray,
        new_credit_by_firm: np.ndarray,
        max_supply_based_on_preferences: np.ndarray,
        new_preference_credit_by_bank: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Clear the market for firm loans (short-term or long-term).

        Matches firms with banks for lending, considering:
        - Capital adequacy requirements
        - Capital-stock collateral limits
        - Return on equity/assets requirements
        - Interest rates

        Args:
            banks (Banks): Banking sector agent
            firms (Firms): Corporate sector agents
            loan_type (LoanTypes): FIRM_SHORT_TERM_LOAN or FIRM_LONG_TERM_LOAN
            new_credit_by_bank (np.ndarray): Running total of new lending by bank
            new_credit_by_firm (np.ndarray): Running total of new borrowing by firm
            max_supply_based_on_preferences (np.ndarray): Maximum lending by bank
            new_preference_credit_by_bank (np.ndarray | None): Running total of
                new lending against this loan-type preference cap

        Returns:
            pd.DataFrame: Details of newly originated firm loans

        Note:
            The process for each firm:
            1. Randomly select subset of banks to approach
            2. Sort banks by interest rate (lowest first)
            3. Try to borrow from each bank up to constraints
            4. Move to next bank if more credit needed
        """
        # Data on new loans
        new_loan_types = []
        new_loan_value = []
        new_loan_maturity = []
        new_loan_interest_rate = []
        new_loan_bank_id = []
        new_loan_recipient_id = []
        if new_preference_credit_by_bank is None:
            new_preference_credit_by_bank = new_credit_by_bank

        # Select loan maturity
        if loan_type == LoanTypes.FIRM_SHORT_TERM_LOAN:
            loan_maturity = banks.parameters.short_term_firm_loan_maturity
        else:
            loan_maturity = banks.parameters.long_term_firm_loan_maturity

        # Get bank interest rates
        if loan_type == LoanTypes.FIRM_SHORT_TERM_LOAN:
            banks_ir = banks.ts.current("interest_rates_on_short_term_firm_loans")
        else:
            banks_ir = banks.ts.current("interest_rates_on_long_term_firm_loans")

        # Iterate over firms with financing needs
        if loan_type == LoanTypes.FIRM_SHORT_TERM_LOAN:
            firm_target_credit = firms.ts.current("target_short_term_credit")
        else:
            firm_target_credit = firms.ts.current("target_long_term_credit")
        firms_with_needs = np.where(firm_target_credit > 0)[0]
        firms_with_needs_shuffled = np.random.choice(firms_with_needs, len(firms_with_needs), replace=False)
        for firm_id in firms_with_needs_shuffled:
            # Take a subset of all banks
            banks_subset = np.random.choice(
                range(banks.ts.current("n_banks")),
                min(
                    self.firms_max_number_of_banks_visiting,
                    banks.ts.current("n_banks"),
                ),
                replace=False,
            )

            # Iterate over all banks based on the offered interest rate
            for bank_id in banks_subset[np.argsort(banks_ir[banks_subset])]:
                if firm_target_credit[firm_id] == 0:
                    break

                # Supply
                total_credit_supply = (
                    banks.ts.current("equity")[bank_id] / banks.parameters.capital_adequacy_ratio
                    - banks.ts.current("total_outstanding_loans")[bank_id]
                    - new_credit_by_bank[bank_id]
                )

                # Capital-stock collateral limit
                capital_stock_collateral_restrictions = (
                    banks.parameters.firm_loans_capital_stock_collateral_ratio
                    * firms.ts.current("capital_inputs_stock_value")[firm_id]
                    - firms.ts.current("debt")[firm_id]
                    - new_credit_by_firm[firm_id]
                    + min(0, firms.ts.current("deposits")[firm_id])
                )

                # Return on equity
                return_on_equity_restrictions = (
                    firms.ts.current("capital_inputs_stock_value")[firm_id]
                    + firms.ts.current("deposits")[firm_id]
                    - firms.ts.current("debt")[firm_id]
                    - new_credit_by_firm[firm_id]
                    - firms.ts.current("expected_profits")[firm_id] / banks.parameters.firm_loans_return_on_equity_ratio
                )

                # Return on assets
                return_on_assets_restrictions = (
                    np.inf
                    if firms.ts.current("expected_profits")[firm_id]
                    / firms.ts.current("capital_inputs_stock_value")[firm_id]
                    >= banks.parameters.firm_loans_return_on_assets_ratio
                    else 0.0
                )

                # Combine
                value_granted = max(
                    0.0,
                    min(
                        firm_target_credit[firm_id],
                        total_credit_supply,
                        max_supply_based_on_preferences[bank_id] - new_preference_credit_by_bank[bank_id],
                        capital_stock_collateral_restrictions,
                        return_on_equity_restrictions,
                        return_on_assets_restrictions,
                    ),
                )

                # Record the new loans
                if value_granted > 0:
                    new_credit_by_bank[bank_id] += value_granted
                    new_preference_credit_by_bank[bank_id] += value_granted
                    new_credit_by_firm[firm_id] += value_granted
                    firm_target_credit[firm_id] -= value_granted
                    new_loan_types.append(loan_type)
                    new_loan_value.append(value_granted)
                    new_loan_maturity.append(loan_maturity)
                    new_loan_interest_rate.append(banks_ir[bank_id])
                    new_loan_bank_id.append(bank_id)
                    new_loan_recipient_id.append(firm_id)

        return pd.DataFrame(
            data={
                "loan_type": new_loan_types,
                "loan_value_initial": new_loan_value,
                "loan_value": new_loan_value,
                "loan_maturity": new_loan_maturity,
                "loan_interest_rate": new_loan_interest_rate,
                "loan_bank_id": new_loan_bank_id,
                "loan_recipient_id": new_loan_recipient_id,
            }
        )

    def clear_household_consumption_loans(
        self,
        banks: Banks,
        households: Households,
        new_credit_by_bank: np.ndarray,
        new_credit_by_household: np.ndarray,
        max_supply_based_on_preferences: np.ndarray,
        new_preference_credit_by_bank: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Clear the market for household consumption loans.

        Matches households with banks for consumer lending, considering:
        - Capital adequacy requirements
        - Loan-to-income ratios
        - Interest rates
        - Bank lending preferences

        Args:
            banks (Banks): Banking sector agent
            households (Households): Household sector agents
            new_credit_by_bank (np.ndarray): Running total of new lending by bank
            new_credit_by_household (np.ndarray): Running total of new borrowing by household
            max_supply_based_on_preferences (np.ndarray): Maximum lending by bank
            new_preference_credit_by_bank (np.ndarray | None): Running total of
                new lending against this loan-type preference cap

        Returns:
            pd.DataFrame: Details of newly originated consumer loans

        Note:
            The process for each household:
            1. Randomly select subset of banks to approach
            2. Sort banks by interest rate (lowest first)
            3. Try to borrow from each bank up to constraints
            4. Move to next bank if more credit needed
        """
        # Data on new loans
        new_loan_types = []
        new_loan_value = []
        new_loan_maturity = []
        new_loan_interest_rate = []
        new_loan_bank_id = []
        new_loan_recipient_id = []
        if new_preference_credit_by_bank is None:
            new_preference_credit_by_bank = new_credit_by_bank

        # Loan maturity
        loan_maturity = banks.parameters.household_consumption_loan_maturity

        # Bank interest rates
        banks_ir = banks.ts.current("interest_rates_on_household_consumption_loans")

        # Iterate over households with financing needs
        household_target_credit = households.ts.current("target_consumption_loans")
        households_with_needs = np.where(household_target_credit > 0)[0]
        households_with_needs_shuffled = np.random.choice(
            households_with_needs,
            len(households_with_needs),
            replace=False,
        )
        for household_id in households_with_needs_shuffled:
            # Take a subset of all banks
            banks_subset = np.random.choice(
                range(banks.ts.current("n_banks")),
                min(
                    self.households_max_number_of_banks_visiting,
                    banks.ts.current("n_banks"),
                ),
                replace=False,
            )

            # Iterate over all banks based on the offered interest rate
            for bank_id in banks_subset[np.argsort(banks_ir[banks_subset])]:
                if household_target_credit[household_id] == 0:
                    break

                # Supply
                total_credit_supply = (
                    banks.ts.current("equity")[bank_id] / banks.parameters.capital_adequacy_ratio
                    - banks.ts.current("total_outstanding_loans")[bank_id]
                    - new_credit_by_bank[bank_id]
                )

                # Loan to income
                loan_to_income_restrictions = (
                    banks.parameters.household_consumption_loans_loan_to_income_ratio
                    * 0.5
                    * (households.ts.prev("income")[household_id] + households.ts.current("income")[household_id])
                    - households.ts.current("debt")[household_id]
                    - new_credit_by_household[household_id]
                )

                # Combine
                value_granted = max(
                    0.0,
                    min(
                        household_target_credit[household_id],
                        total_credit_supply,
                        max_supply_based_on_preferences[bank_id] - new_preference_credit_by_bank[bank_id],
                        loan_to_income_restrictions,
                    ),
                )

                # Record the new loans
                if value_granted > 0:
                    new_credit_by_bank[bank_id] += value_granted
                    new_preference_credit_by_bank[bank_id] += value_granted
                    new_credit_by_household[household_id] += value_granted
                    household_target_credit[household_id] -= value_granted
                    new_loan_types.append(LoanTypes.HOUSEHOLD_CONSUMPTION_LOAN)
                    new_loan_value.append(value_granted)
                    new_loan_maturity.append(loan_maturity)
                    new_loan_interest_rate.append(banks_ir[bank_id])
                    new_loan_bank_id.append(bank_id)
                    new_loan_recipient_id.append(household_id)

        return pd.DataFrame(
            data={
                "loan_type": new_loan_types,
                "loan_value_initial": new_loan_value,
                "loan_value": new_loan_value,
                "loan_maturity": new_loan_maturity,
                "loan_interest_rate": new_loan_interest_rate,
                "loan_bank_id": new_loan_bank_id,
                "loan_recipient_id": new_loan_recipient_id,
            }
        )

    def clear_mortgages(
        self,
        banks: Banks,
        households: Households,
        loan_type: LoanTypes,
        new_credit_by_bank: np.ndarray,
        new_credit_by_household: np.ndarray,
        max_supply_based_on_preferences: np.ndarray,
        new_preference_credit_by_bank: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Clear the market for mortgage loans.

        Matches households with banks for mortgage lending, considering:
        - Capital adequacy requirements
        - Loan-to-income ratios
        - Loan-to-value ratios
        - Debt service coverage
        - Interest rates

        Args:
            banks (Banks): Banking sector agent
            households (Households): Household sector agents
            loan_type (LoanTypes): Should be MORTGAGE
            new_credit_by_bank (np.ndarray): Running total of new lending by bank
            new_credit_by_household (np.ndarray): Running total of new borrowing by household
            max_supply_based_on_preferences (np.ndarray): Maximum lending by bank
            new_preference_credit_by_bank (np.ndarray | None): Running total of
                new lending against this loan-type preference cap

        Returns:
            pd.DataFrame: Details of newly originated mortgages

        Note:
            The process for each household:
            1. Randomly select subset of banks to approach
            2. Sort banks by interest rate (lowest first)
            3. Try to borrow full amount from each bank
            4. Only accept if full amount can be borrowed
        """
        # Data on new loans
        new_loan_types = []
        new_loan_value = []
        new_loan_maturity = []
        new_loan_interest_rate = []
        new_loan_bank_id = []
        new_loan_recipient_id = []
        if new_preference_credit_by_bank is None:
            new_preference_credit_by_bank = new_credit_by_bank

        # Get bank interest rates
        banks_ir = banks.ts.current("interest_rates_on_mortgages")

        # Iterate over households with financing needs
        household_target_credit = households.ts.current("target_mortgage")
        households_with_needs = np.where(household_target_credit > 0)[0]
        households_with_needs_shuffled = np.random.choice(
            households_with_needs,
            len(households_with_needs),
            replace=False,
        )
        for household_id in households_with_needs_shuffled:
            # Take a subset of all banks
            banks_subset = np.random.choice(
                range(banks.ts.current("n_banks")),
                min(
                    self.households_max_number_of_banks_visiting,
                    banks.ts.current("n_banks"),
                ),
                replace=False,
            )

            # Iterate over all banks based on the offered interest rate
            for bank_id in banks_subset[np.argsort(banks_ir[banks_subset])]:
                if household_target_credit[household_id] == 0:
                    break

                # Supply
                total_credit_supply = (
                    banks.ts.current("equity")[bank_id] / banks.parameters.capital_adequacy_ratio
                    - banks.ts.current("total_outstanding_loans")[bank_id]
                    - new_credit_by_bank[bank_id]
                )

                # Loan to income
                loan_to_income_restrictions = (
                    banks.parameters.mortgage_loan_to_income_ratio
                    * 0.5
                    * (households.ts.prev("income")[household_id] + households.ts.current("income")[household_id])
                    - households.ts.current("debt")[household_id]
                    - new_credit_by_household[household_id]
                )

                # Loan to value
                loan_to_value_restrictions = (
                    banks.parameters.mortgage_loan_to_value_ratio
                    / (1 - banks.parameters.mortgage_loan_to_value_ratio)
                    * households.ts.current("wealth_financial_assets")[household_id]
                )

                # Debt service to income
                debt_service_to_income_restrictions = (
                    banks.parameters.mortgage_debt_service_to_income_ratio
                    * households.ts.current("income")[household_id]
                    * (1 - (1 + banks_ir[bank_id]) ** (-banks.parameters.mortgage_maturity))
                    / banks_ir[bank_id]
                )

                # Combine
                value_granted = max(
                    0.0,
                    min(
                        household_target_credit[household_id],
                        total_credit_supply,
                        max_supply_based_on_preferences[bank_id] - new_preference_credit_by_bank[bank_id],
                        loan_to_income_restrictions,
                        loan_to_value_restrictions,
                        debt_service_to_income_restrictions,
                    ),
                )

                # Only take the mortgage if we get the full amount
                if value_granted < household_target_credit[household_id]:
                    continue

                # Record the new mortgage
                if value_granted > 0:
                    new_credit_by_bank[bank_id] += value_granted
                    new_preference_credit_by_bank[bank_id] += value_granted
                    new_credit_by_household[household_id] += value_granted
                    household_target_credit[household_id] -= value_granted
                    new_loan_types.append(loan_type)
                    new_loan_value.append(value_granted)
                    new_loan_maturity.append(banks.parameters.mortgage_maturity)
                    new_loan_interest_rate.append(banks_ir[bank_id])
                    new_loan_bank_id.append(bank_id)
                    new_loan_recipient_id.append(household_id)

        return pd.DataFrame(
            data={
                "loan_type": new_loan_types,
                "loan_value_initial": new_loan_value,
                "loan_value": new_loan_value,
                "loan_maturity": new_loan_maturity,
                "loan_interest_rate": new_loan_interest_rate,
                "loan_bank_id": new_loan_bank_id,
                "loan_recipient_id": new_loan_recipient_id,
            }
        )


class PolednaCreditMarketClearer(CreditMarketClearer):
    """Simplified credit market clearing based on Poledna et al.

    This implementation provides a simpler clearing mechanism focused on firm lending
    with basic capital requirements. It uses:
    - Basic capital adequacy checks
    - Simplified risk assessment
    - Interest rate based matching
    - Short-term firm loans only
    """

    def clear(
        self,
        banks: Banks,
        firms: Firms,
        households: Households,
        current_npl_firm_loans: float,
        current_npl_hh_cons_loans: float,
        current_npl_mortgages: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Execute the Poledna market clearing algorithm.

        Focuses only on short-term firm lending with simplified constraints.

        Args:
            banks (Banks): Banking sector agent
            firms (Firms): Corporate sector agents
            households (Households): Unused
            current_npl_firm_loans (float): Unused
            current_npl_hh_cons_loans (float): Unused
            current_npl_mortgages (float): Unused

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: New loan arrays
        """
        new_credit_by_bank = np.zeros(banks.ts.current("n_banks"))
        new_credit_by_firm = np.zeros(firms.ts.current("n_firms"))
        new_loans = self.clear_firm_loans(
            banks=banks,
            firms=firms,
            loan_type=LoanTypes.FIRM_SHORT_TERM_LOAN,
            new_credit_by_bank=new_credit_by_bank,
            new_credit_by_firm=new_credit_by_firm,
        ).reset_index(drop=True)
        new_loans["loan_bank_id"] = new_loans["loan_bank_id"].astype(int)
        new_loans["loan_recipient_id"] = new_loans["loan_recipient_id"].astype(int)
        return _loan_frame_to_arrays(new_loans, banks, firms, households)

    def clear_firm_loans(
        self,
        banks: Banks,
        firms: Firms,
        loan_type: LoanTypes,
        new_credit_by_bank: np.ndarray,
        new_credit_by_firm: np.ndarray,
    ) -> pd.DataFrame:
        """Clear the market for firm loans using simplified constraints.

        Uses basic capital adequacy and risk assessment for firm lending.

        Args:
            banks (Banks): Banking sector agent
            firms (Firms): Corporate sector agents
            loan_type (LoanTypes): Type of firm loan
            new_credit_by_bank (np.ndarray): Running total of new lending by bank
            new_credit_by_firm (np.ndarray): Running total of new borrowing by firm

        Returns:
            pd.DataFrame: Details of newly originated firm loans
        """
        # Data on new loans
        new_loan_types = []
        new_loan_value = []
        new_loan_maturity = []
        new_loan_interest_rate = []
        new_loan_bank_id = []
        new_loan_recipient_id = []

        # Select loan maturity
        if loan_type == LoanTypes.FIRM_SHORT_TERM_LOAN:
            loan_maturity = banks.parameters.short_term_firm_loan_maturity
        else:
            loan_maturity = banks.parameters.long_term_firm_loan_maturity

        # Get bank interest rates
        if loan_type == LoanTypes.FIRM_SHORT_TERM_LOAN:
            banks_ir = banks.ts.current("interest_rates_on_short_term_firm_loans")
        else:
            banks_ir = banks.ts.current("interest_rates_on_long_term_firm_loans")

        # Iterate over firms with financing needs
        if loan_type == LoanTypes.FIRM_SHORT_TERM_LOAN:
            firm_target_credit = firms.ts.current("target_short_term_credit")
        else:
            firm_target_credit = firms.ts.current("target_long_term_credit")
        firms_with_needs = np.where(firm_target_credit > 0)[0]
        firms_with_needs_shuffled = np.random.choice(
            firms_with_needs,
            len(firms_with_needs),
            replace=False,
        )
        for firm_id in firms_with_needs_shuffled:
            # Take a subset of all banks
            banks_subset = np.random.choice(
                range(banks.ts.current("n_banks")),
                min(
                    self.firms_max_number_of_banks_visiting,
                    banks.ts.current("n_banks"),
                ),
                replace=False,
            )

            # Iterate over all banks based on the offered interest rate
            for bank_id in banks_subset[np.argsort(banks_ir[banks_subset])]:
                if firm_target_credit[firm_id] == 0:
                    break
                bank_cap_req = (
                    banks.ts.current("equity")[bank_id] / banks.parameters.capital_adequacy_ratio
                    - (1 - 1.0 / loan_maturity) * banks.ts.current("total_outstanding_loans")[bank_id]
                    - new_credit_by_bank[bank_id]
                )
                capital_stock_collateral_capacity = (
                    banks.parameters.firm_loans_capital_stock_collateral_ratio
                    * firms.ts.current("expected_capital_inputs_stock_value")[firm_id]
                    - (1 - 1.0 / loan_maturity) * firms.ts.current("debt")[firm_id]
                )
                value_granted = max(
                    0.0,
                    min(
                        firm_target_credit[firm_id],
                        bank_cap_req,
                        capital_stock_collateral_capacity,
                    ),
                )

                # Record the new loans
                if value_granted > 0:
                    new_credit_by_bank[bank_id] += value_granted
                    new_credit_by_firm[firm_id] += value_granted
                    firm_target_credit[firm_id] -= value_granted
                    new_loan_types.append(loan_type)
                    new_loan_value.append(value_granted)
                    new_loan_maturity.append(loan_maturity)
                    new_loan_interest_rate.append(banks_ir[bank_id])
                    new_loan_bank_id.append(bank_id)
                    new_loan_recipient_id.append(firm_id)

        return pd.DataFrame(
            data={
                "loan_type": new_loan_types,
                "loan_value_initial": new_loan_value,
                "loan_value": new_loan_value,
                "loan_maturity": new_loan_maturity,
                "loan_interest_rate": new_loan_interest_rate,
                "loan_bank_id": new_loan_bank_id,
                "loan_recipient_id": new_loan_recipient_id,
            }
        )


class WaterBucketCreditMarketClearer(CreditMarketClearer):
    """Network flow-based credit market clearing using the water bucket algorithm.

    This implementation models credit allocation as a network flow problem, where
    credit supply and demand are distributed through the network like water flowing
    through buckets. The algorithm:

    1. Credit Supply Management:
       - Uses bank preferences for different loan types
       - Adjusts supply based on NPL rates
       - Respects capital adequacy requirements

    2. Priority-Based Allocation:
       - Supports both deterministic and stochastic creditor selection
       - Uses interest rates to determine priorities
       - Enforces minimum fill rates

    3. Multi-Stage Clearing:
       - Clears each loan type separately
       - Maintains running totals of lending/borrowing
       - Updates bank portfolio composition

    Key Features:
    - Network flow optimization for efficient allocation
    - Support for all loan types
    - Risk-sensitive credit supply
    - Interest rate based prioritization
    - Regulatory compliance
    """

    def clear(
        self,
        banks: Banks,
        firms: Firms,
        households: Households,
        current_npl_firm_loans: float,
        current_npl_hh_cons_loans: float,
        current_npl_mortgages: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Execute the water bucket credit market clearing algorithm.

        This method implements a sophisticated credit allocation mechanism that
        models credit flows like water flowing through a network of buckets.
        It operates in multiple stages to ensure efficient and compliant allocation.

        Firm short-term demand is split into ordinary working-capital demand,
        scheduled-principal rollover demand, and emergency overdraft-refinance
        demand. Ordinary short-term firm loans, long-term firm loans, household
        consumption loans, and mortgages clear first against their normal caps.
        Debt rollover and overdraft refinance clear afterward from residual bank
        CAR only, so legacy-liability repair cannot reallocate ordinary firm
        lending capacity away from long-term investment loans.

        Args:
            banks (Banks): Banking sector agent
            firms (Firms): Corporate sector agents
            households (Households): Household sector agents
            current_npl_firm_loans (float): Current NPL rate for firm loans
            current_npl_hh_cons_loans (float): Current NPL rate for consumer loans
            current_npl_mortgages (float): Current NPL rate for mortgages

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Arrays of new loans:
                - Short-term firm loans [3, n_banks, n_firms]
                - Long-term firm loans [3, n_banks, n_firms]
                - Consumer loans [3, n_banks, n_households]
                - Mortgages [3, n_banks, n_households]

        Note:
            Each loan array has shape [3, n_banks, n_borrowers] where:
            - Index 0: Principal amount
            - Index 1: Period interest rate
            - Index 2: Scheduled annuity payment
        """
        # Keeping track of new credit
        new_credit_by_bank = np.zeros(banks.ts.current("n_banks"))
        new_credit_by_firm = np.zeros(firms.ts.current("n_firms"))
        new_debt_service_by_firm = np.zeros(firms.ts.current("n_firms"))
        new_credit_by_household = np.zeros(households.ts.current("n_households"))
        new_firm_preference_credit_by_bank = np.zeros(banks.ts.current("n_banks"))
        new_household_consumption_preference_credit_by_bank = np.zeros(banks.ts.current("n_banks"))
        new_mortgage_preference_credit_by_bank = np.zeros(banks.ts.current("n_banks"))

        # Firm CFADS proxy (only meaningful when DSCR is enabled)
        if banks.parameters.enable_firm_loans_dscr_restriction:
            cfads = _compute_firm_cfads(
                firms,
                window=banks.parameters.firm_loans_cfads_window,
                haircut=banks.parameters.firm_loans_cfads_haircut,
            )
        else:
            cfads = _nan_array(firms.ts.current("n_firms"))
        _append_vector(firms.ts, "credit_market_firm_cfads", cfads)

        # Banks may update their preferences for different types of loans, impacting their supply
        if self.consider_loan_type_fractions:
            (
                max_supply_based_on_preferences_firms,
                max_supply_based_on_preferences_hh_cons,
                max_supply_based_on_preferences_mortgages,
            ) = _compute_loan_type_preference_caps(
                banks,
                current_npl_firm_loans,
                current_npl_hh_cons_loans,
                current_npl_mortgages,
                self.credit_supply_temperature,
            )
        else:
            max_supply_based_on_preferences_firms = np.full(banks.ts.current("n_banks"), np.inf)
            max_supply_based_on_preferences_hh_cons = np.full(banks.ts.current("n_banks"), np.inf)
            max_supply_based_on_preferences_mortgages = np.full(banks.ts.current("n_banks"), np.inf)

        target_short_term_credit = np.asarray(firms.ts.current("target_short_term_credit"), dtype=float)
        target_debt_rollover_credit = np.asarray(
            firms.ts.current("target_debt_rollover_credit"),
            dtype=float,
        )
        target_debt_rollover_credit = np.minimum(
            np.maximum(0.0, target_debt_rollover_credit),
            np.maximum(0.0, target_short_term_credit),
        )
        target_overdraft_refinance_credit = np.asarray(
            firms.ts.current("target_overdraft_refinance_credit"),
            dtype=float,
        )
        target_overdraft_refinance_credit = np.minimum(
            np.maximum(0.0, target_overdraft_refinance_credit),
            np.maximum(0.0, target_short_term_credit - target_debt_rollover_credit),
        )
        ordinary_target_short_term_credit = np.asarray(
            firms.ts.current("ordinary_target_short_term_credit"),
            dtype=float,
        )
        if (
            np.nansum(ordinary_target_short_term_credit) == 0.0
            and np.nansum(target_debt_rollover_credit) == 0.0
            and np.nansum(target_overdraft_refinance_credit) == 0.0
        ):
            ordinary_target_short_term_credit = target_short_term_credit.copy()
        ordinary_target_short_term_credit = np.maximum(
            0.0,
            np.minimum(
                ordinary_target_short_term_credit,
                target_short_term_credit - target_debt_rollover_credit - target_overdraft_refinance_credit,
            ),
        )

        self._last_received_debt_rollover_credit_by_firm = np.zeros(firms.ts.current("n_firms"))
        self._last_received_overdraft_refinance_credit_by_firm = np.zeros(firms.ts.current("n_firms"))
        initial_new_credit_by_firm = new_credit_by_firm.copy()

        # Ordinary firm loans clear before rollover/refinance so legacy-liability repair cannot
        # reallocate ordinary long-term capacity.
        new_ordinary_st_loans = self.clear_loans(
            banks=banks,
            firms=firms,
            households=households,
            loan_type=LoanTypes.FIRM_SHORT_TERM_LOAN,
            new_credit_by_bank=new_credit_by_bank,
            new_credit_by_firm=new_credit_by_firm,
            new_credit_by_household=new_credit_by_household,
            max_supply_based_on_preferences=max_supply_based_on_preferences_firms,
            new_debt_service_by_firm=new_debt_service_by_firm,
            new_preference_credit_by_bank=new_firm_preference_credit_by_bank,
            target_credit_override=ordinary_target_short_term_credit,
            append_firm_diagnostics=False,
        )
        new_credit_by_firm += new_ordinary_st_loans[0].sum(axis=0)
        new_credit_by_bank += new_ordinary_st_loans[0].sum(axis=1)
        new_firm_preference_credit_by_bank += new_ordinary_st_loans[0].sum(axis=1)
        new_lt_loans = self.clear_loans(
            banks=banks,
            firms=firms,
            households=households,
            loan_type=LoanTypes.FIRM_LONG_TERM_LOAN,
            new_credit_by_bank=new_credit_by_bank,
            new_credit_by_firm=new_credit_by_firm,
            new_credit_by_household=new_credit_by_household,
            max_supply_based_on_preferences=max_supply_based_on_preferences_firms,
            new_debt_service_by_firm=new_debt_service_by_firm,
            new_preference_credit_by_bank=new_firm_preference_credit_by_bank,
        )
        new_credit_by_firm += new_lt_loans[0].sum(axis=0)
        new_credit_by_bank += new_lt_loans[0].sum(axis=1)
        new_firm_preference_credit_by_bank += new_lt_loans[0].sum(axis=1)

        # Household loans
        new_cons_loans = self.clear_loans(
            banks=banks,
            firms=firms,
            households=households,
            loan_type=LoanTypes.HOUSEHOLD_CONSUMPTION_LOAN,
            new_credit_by_bank=new_credit_by_bank,
            new_credit_by_firm=new_credit_by_firm,
            new_credit_by_household=new_credit_by_household,
            max_supply_based_on_preferences=max_supply_based_on_preferences_hh_cons,
            new_preference_credit_by_bank=new_household_consumption_preference_credit_by_bank,
        )
        new_credit_by_household += new_cons_loans[0].sum(axis=0)
        new_credit_by_bank += new_cons_loans[0].sum(axis=1)
        new_household_consumption_preference_credit_by_bank += new_cons_loans[0].sum(axis=1)
        new_mort_loans = self.clear_loans(
            banks=banks,
            firms=firms,
            households=households,
            loan_type=LoanTypes.MORTGAGE,
            new_credit_by_bank=new_credit_by_bank,
            new_credit_by_firm=new_credit_by_firm,
            new_credit_by_household=new_credit_by_household,
            max_supply_based_on_preferences=max_supply_based_on_preferences_mortgages,
            new_preference_credit_by_bank=new_mortgage_preference_credit_by_bank,
        )

        new_credit_by_household += new_mort_loans[0].sum(axis=0)
        new_credit_by_bank += new_mort_loans[0].sum(axis=1)
        new_mortgage_preference_credit_by_bank += new_mort_loans[0].sum(axis=1)

        residual_car_room = np.maximum(
            0.0,
            banks.ts.current("equity") / banks.parameters.capital_adequacy_ratio
            - banks.ts.current("total_outstanding_loans")
            - new_credit_by_bank,
        )
        new_debt_rollover_st_loans = self.clear_loans(
            banks=banks,
            firms=firms,
            households=households,
            loan_type=LoanTypes.FIRM_SHORT_TERM_LOAN,
            new_credit_by_bank=new_credit_by_bank,
            new_credit_by_firm=new_credit_by_firm,
            new_credit_by_household=new_credit_by_household,
            max_supply_based_on_preferences=residual_car_room,
            new_debt_service_by_firm=new_debt_service_by_firm,
            new_preference_credit_by_bank=np.zeros(banks.ts.current("n_banks")),
            target_credit_override=target_debt_rollover_credit,
            append_firm_diagnostics=False,
        )
        self._last_received_debt_rollover_credit_by_firm = new_debt_rollover_st_loans[0].sum(axis=0)
        new_credit_by_firm += new_debt_rollover_st_loans[0].sum(axis=0)
        new_credit_by_bank += new_debt_rollover_st_loans[0].sum(axis=1)

        residual_car_room = np.maximum(
            0.0,
            banks.ts.current("equity") / banks.parameters.capital_adequacy_ratio
            - banks.ts.current("total_outstanding_loans")
            - new_credit_by_bank,
        )
        firm_pref_room_before_refinance = np.maximum(
            0.0,
            max_supply_based_on_preferences_firms - new_firm_preference_credit_by_bank,
        )
        firm_pref_likely_binding_before_refinance = bool(
            self.consider_loan_type_fractions
            and np.isfinite(firm_pref_room_before_refinance.sum())
            and firm_pref_room_before_refinance.sum() < residual_car_room.sum()
        )
        new_refinance_st_loans = self.clear_loans(
            banks=banks,
            firms=firms,
            households=households,
            loan_type=LoanTypes.FIRM_SHORT_TERM_LOAN,
            new_credit_by_bank=new_credit_by_bank,
            new_credit_by_firm=new_credit_by_firm,
            new_credit_by_household=new_credit_by_household,
            max_supply_based_on_preferences=residual_car_room,
            new_debt_service_by_firm=new_debt_service_by_firm,
            new_preference_credit_by_bank=np.zeros(banks.ts.current("n_banks")),
            target_credit_override=target_overdraft_refinance_credit,
            append_firm_diagnostics=False,
        )
        self._last_received_overdraft_refinance_credit_by_firm = new_refinance_st_loans[0].sum(axis=0)
        new_credit_by_firm += new_refinance_st_loans[0].sum(axis=0)
        new_credit_by_bank += new_refinance_st_loans[0].sum(axis=1)
        new_st_loans = np.zeros_like(new_ordinary_st_loans)
        new_st_loans[0] = new_ordinary_st_loans[0] + new_debt_rollover_st_loans[0] + new_refinance_st_loans[0]
        new_st_loans[1] = np.divide(
            new_ordinary_st_loans[1] * new_ordinary_st_loans[0]
            + new_debt_rollover_st_loans[1] * new_debt_rollover_st_loans[0]
            + new_refinance_st_loans[1] * new_refinance_st_loans[0],
            new_st_loans[0],
            out=np.zeros_like(new_st_loans[0]),
            where=new_st_loans[0] > 0.0,
        )
        new_st_loans[2] = new_ordinary_st_loans[2] + new_debt_rollover_st_loans[2] + new_refinance_st_loans[2]

        n_firms = firms.ts.current("n_firms")
        st_granted_by_firm = new_st_loans[0].sum(axis=0)
        st_reason = np.full(n_firms, _BINDING_NO_DEMAND, dtype=float)
        st_amount = np.zeros(n_firms, dtype=float)
        st_collateral_cap = np.zeros(n_firms, dtype=float)
        st_dscr_cap = _nan_array(n_firms)
        st_has_demand = target_short_term_credit > 0.0
        st_fully_granted = st_has_demand & (st_granted_by_firm >= target_short_term_credit - 1e-9)
        st_unmet = st_has_demand & ~st_fully_granted
        st_reason[st_fully_granted] = _BINDING_NOT_BINDING
        st_amount[st_fully_granted] = target_short_term_credit[st_fully_granted]
        if np.any(st_has_demand):
            st_agents_with_demand = np.where(st_has_demand)[0]
            st_collateral_cap[st_agents_with_demand] = _clip_nonneg(
                banks.parameters.firm_loans_capital_stock_collateral_ratio
                * firms.ts.current("capital_inputs_stock_value")[st_agents_with_demand]
                - firms.ts.current("debt")[st_agents_with_demand]
                - initial_new_credit_by_firm[st_agents_with_demand]
                + np.minimum(0.0, firms.ts.current("deposits")[st_agents_with_demand])
            )
            st_roa_cap = np.full(n_firms, np.inf)
            if banks.parameters.enable_firm_loans_return_on_assets_restriction:
                firm_expected_profits = firms.ts.current("expected_profits")[st_agents_with_demand]
                firm_capital_stock = firms.ts.current("capital_inputs_stock_value")[st_agents_with_demand]
                firm_roa = np.divide(
                    firm_expected_profits,
                    firm_capital_stock,
                    out=np.zeros_like(firm_expected_profits),
                    where=firm_capital_stock != 0,
                )
                st_roa_cap[st_agents_with_demand[firm_roa < banks.parameters.firm_loans_return_on_assets_ratio]] = 0.0

            st_roe_cap = np.full(n_firms, np.inf)
            if banks.parameters.enable_firm_loans_return_on_equity_restriction:
                st_roe_cap[st_agents_with_demand] = _clip_nonneg(
                    firms.ts.current("capital_inputs_stock_value")[st_agents_with_demand]
                    + firms.ts.current("deposits")[st_agents_with_demand]
                    - firms.ts.current("debt")[st_agents_with_demand]
                    - initial_new_credit_by_firm[st_agents_with_demand]
                    - firms.ts.current("expected_profits")[st_agents_with_demand]
                    / banks.parameters.firm_loans_return_on_equity_ratio
                )

            st_dscr_cap_eff = np.full(n_firms, np.inf)
            if banks.parameters.enable_firm_loans_dscr_restriction:
                st_underwriting_rate = _firm_dscr_underwriting_rate(
                    banks.ts.current("interest_rates_on_short_term_firm_loans"),
                    banks.parameters.firm_loans_dscr_underwriting_rate_mode,
                )
                st_dscr_cap[st_agents_with_demand] = _clip_nonneg(
                    _compute_firm_dscr_capacity(
                        firms=firms,
                        agents_with_demand=st_agents_with_demand,
                        min_dscr=banks.parameters.firm_loans_min_dscr,
                        cfads_window=banks.parameters.firm_loans_cfads_window,
                        cfads_haircut=banks.parameters.firm_loans_cfads_haircut,
                        underwriting_rate=st_underwriting_rate,
                        loan_maturity=banks.parameters.short_term_firm_loan_maturity,
                        new_debt_service_by_firm=np.zeros(n_firms),
                    )
                )
                st_dscr_cap_eff = np.where(np.isfinite(st_dscr_cap), st_dscr_cap, np.inf)

            st_borrower_cap = np.minimum.reduce((st_collateral_cap, st_roa_cap, st_roe_cap, st_dscr_cap_eff))
            st_borrower_binds = st_unmet & (st_borrower_cap < target_short_term_credit - 1e-9)

            st_collateral_binds = (
                st_borrower_binds
                & (st_collateral_cap <= st_roa_cap)
                & (st_collateral_cap <= st_roe_cap)
                & (st_collateral_cap <= st_dscr_cap_eff)
            )
            st_reason[st_collateral_binds] = _BINDING_COLLATERAL
            st_amount[st_collateral_binds] = st_collateral_cap[st_collateral_binds]

            st_dscr_binds = (
                st_borrower_binds
                & ~st_collateral_binds
                & (st_dscr_cap_eff <= st_roa_cap)
                & (st_dscr_cap_eff <= st_roe_cap)
            )
            st_reason[st_dscr_binds] = _BINDING_DSCR
            st_amount[st_dscr_binds] = st_dscr_cap[st_dscr_binds]

            st_roa_binds = st_borrower_binds & ~st_collateral_binds & ~st_dscr_binds & (st_roa_cap <= st_roe_cap)
            st_reason[st_roa_binds] = _BINDING_ROA
            st_amount[st_roa_binds] = 0.0

            st_roe_binds = st_borrower_binds & ~st_collateral_binds & ~st_dscr_binds & ~st_roa_binds
            st_reason[st_roe_binds] = _BINDING_ROE
            st_amount[st_roe_binds] = st_roe_cap[st_roe_binds]

            st_bank_binds = st_unmet & ~st_borrower_binds
            st_reason[st_bank_binds] = (
                _BINDING_BANK_PREF if firm_pref_likely_binding_before_refinance else _BINDING_BANK_CAR
            )
            st_amount[st_bank_binds] = st_granted_by_firm[st_bank_binds]

        _append_vector(firms.ts, "credit_market_firm_st_capacity", st_granted_by_firm)
        _append_vector(firms.ts, "credit_market_firm_st_collateral_cap", st_collateral_cap)
        _append_vector(firms.ts, "credit_market_firm_st_dscr_cap", st_dscr_cap)
        _append_vector(firms.ts, "credit_market_firm_st_binding_reason", st_reason)
        _append_vector(firms.ts, "credit_market_firm_st_binding_amount", st_amount)

        # End-of-clearing diagnostic usage
        _append_scalar(banks.ts, "credit_market_car_room_used", float(np.sum(new_credit_by_bank)))
        if self.consider_loan_type_fractions:
            _append_scalar(
                banks.ts,
                "credit_market_pref_cap_firms_used",
                float(np.sum(new_firm_preference_credit_by_bank)),
            )
            _append_scalar(
                banks.ts,
                "credit_market_pref_cap_hh_cons_used",
                float(np.sum(new_household_consumption_preference_credit_by_bank)),
            )
            _append_scalar(
                banks.ts,
                "credit_market_pref_cap_mortgages_used",
                float(np.sum(new_mortgage_preference_credit_by_bank)),
            )
        else:
            _append_scalar(banks.ts, "credit_market_pref_cap_firms_used", np.nan)
            _append_scalar(banks.ts, "credit_market_pref_cap_hh_cons_used", np.nan)
            _append_scalar(banks.ts, "credit_market_pref_cap_mortgages_used", np.nan)

        return (
            new_st_loans,
            new_lt_loans,
            new_cons_loans,
            new_mort_loans,
        )

    def clear_loans(
        self,
        banks: Banks,
        firms: Firms,
        households: Households,
        loan_type: LoanTypes,
        new_credit_by_bank: np.ndarray,
        new_credit_by_firm: np.ndarray,
        new_credit_by_household: np.ndarray,
        max_supply_based_on_preferences: np.ndarray,
        new_debt_service_by_firm: np.ndarray | None = None,
        new_preference_credit_by_bank: np.ndarray | None = None,
        target_credit_override: np.ndarray | None = None,
        append_firm_diagnostics: bool = True,
    ) -> np.ndarray:
        """Clear the market for a specific loan type using the water bucket algorithm.

        This method implements the core water bucket algorithm for credit allocation:
        1. Calculate supply and demand matrices
        2. Apply regulatory and risk constraints
        3. Distribute credit through the network
        4. Update running totals

        Args:
            banks (Banks): Banking sector agent
            firms (Firms): Corporate sector agents
            households (Households): Household sector agents
            loan_type (LoanTypes): Type of loan to clear
            new_credit_by_bank (np.ndarray): Running total of new lending by bank
            new_credit_by_firm (np.ndarray): Running total of new borrowing by firm
            new_credit_by_household (np.ndarray): Running total of new borrowing by household
            max_supply_based_on_preferences (np.ndarray): Maximum supply based on bank preferences
            new_debt_service_by_firm (np.ndarray | None): Running total of newly
                committed firm debt service during this clearing pass
            new_preference_credit_by_bank (np.ndarray | None): Running total of
                new lending against this loan-type preference cap
            target_credit_override (np.ndarray | None): Optional demand vector used
                for tranche-level clearing, such as ordinary short-term or overdraft
                refinance demand. The recorded target series is left unchanged.
            append_firm_diagnostics (bool): Whether this call appends firm-level
                binding diagnostics. Split short-term tranches disable this and
                append one aggregate short-term diagnostic after the tranches are
                recombined.

        Returns:
            np.ndarray: Array of shape [3, n_banks, n_borrowers] containing:
                - Index 0: Principal amount
                - Index 1: Period interest rate
                - Index 2: Scheduled annuity payment
        """
        # Get bank interest rates
        if loan_type == LoanTypes.FIRM_SHORT_TERM_LOAN:
            loan_maturity = banks.parameters.short_term_firm_loan_maturity
            banks_ir = banks.ts.current("interest_rates_on_short_term_firm_loans")
            target_credit = firms.ts.current("target_short_term_credit")
        elif loan_type == LoanTypes.FIRM_LONG_TERM_LOAN:
            loan_maturity = banks.parameters.long_term_firm_loan_maturity
            banks_ir = banks.ts.current("interest_rates_on_long_term_firm_loans")
            target_credit = firms.ts.current("target_long_term_credit")
        elif loan_type == LoanTypes.HOUSEHOLD_CONSUMPTION_LOAN:
            loan_maturity = banks.parameters.household_consumption_loan_maturity
            banks_ir = banks.ts.current("interest_rates_on_household_consumption_loans")
            target_credit = households.ts.current("target_consumption_loans")
        elif loan_type == LoanTypes.MORTGAGE:
            loan_maturity = banks.parameters.mortgage_maturity
            banks_ir = banks.ts.current("interest_rates_on_mortgages")
            target_credit = households.ts.current("target_mortgage")
        else:
            raise ValueError("Unknown loan type", loan_type)
        if target_credit_override is not None:
            target_credit = np.asarray(target_credit_override, dtype=float)

        # For recording data
        new_loans = np.zeros((3, banks.ts.current("n_banks"), target_credit.shape[0]))
        if new_debt_service_by_firm is None and (
            loan_type == LoanTypes.FIRM_SHORT_TERM_LOAN or loan_type == LoanTypes.FIRM_LONG_TERM_LOAN
        ):
            new_debt_service_by_firm = np.zeros(firms.ts.current("n_firms"))
        if new_preference_credit_by_bank is None:
            new_preference_credit_by_bank = new_credit_by_bank

        # Select agents wanting credit and priorities
        agents_with_demand = np.where(target_credit > 0)[0]
        debtor_priorities = self.get_debtor_priorities(n_agents=agents_with_demand.shape[0])

        # Determine capacities
        if loan_type == LoanTypes.FIRM_SHORT_TERM_LOAN or loan_type == LoanTypes.FIRM_LONG_TERM_LOAN:
            capital_stock_collateral_restrictions = (
                banks.parameters.firm_loans_capital_stock_collateral_ratio
                * firms.ts.current("capital_inputs_stock_value")[agents_with_demand]
                - firms.ts.current("debt")[agents_with_demand]
                - new_credit_by_firm[agents_with_demand]
                + np.minimum(0, firms.ts.current("deposits")[agents_with_demand])
            )
            return_on_equity_restrictions = np.full(agents_with_demand.shape, np.inf)
            if banks.parameters.enable_firm_loans_return_on_equity_restriction:
                return_on_equity_restrictions = (
                    firms.ts.current("capital_inputs_stock_value")[agents_with_demand]
                    + firms.ts.current("deposits")[agents_with_demand]
                    - firms.ts.current("debt")[agents_with_demand]
                    - new_credit_by_firm[agents_with_demand]
                    - firms.ts.current("expected_profits")[agents_with_demand]
                    / banks.parameters.firm_loans_return_on_equity_ratio
                )

            return_on_assets_restrictions = np.full(agents_with_demand.shape, np.inf)
            if banks.parameters.enable_firm_loans_return_on_assets_restriction:
                # Calculate ROA safely to avoid division by zero
                firm_expected_profits = firms.ts.current("expected_profits")[agents_with_demand]
                firm_capital_stock = firms.ts.current("capital_inputs_stock_value")[agents_with_demand]

                firm_roa = np.divide(
                    firm_expected_profits,
                    firm_capital_stock,
                    out=np.zeros_like(firm_expected_profits),
                    where=firm_capital_stock != 0,
                )
                return_on_assets_restrictions[firm_roa < banks.parameters.firm_loans_return_on_assets_ratio] = 0.0

            dscr_restrictions = np.full(agents_with_demand.shape, np.inf)
            underwriting_rate = np.nan
            if banks.parameters.enable_firm_loans_dscr_restriction:
                underwriting_rate = _firm_dscr_underwriting_rate(
                    banks_ir,
                    banks.parameters.firm_loans_dscr_underwriting_rate_mode,
                )
                dscr_restrictions = _compute_firm_dscr_capacity(
                    firms=firms,
                    agents_with_demand=agents_with_demand,
                    min_dscr=banks.parameters.firm_loans_min_dscr,
                    cfads_window=banks.parameters.firm_loans_cfads_window,
                    cfads_haircut=banks.parameters.firm_loans_cfads_haircut,
                    underwriting_rate=underwriting_rate,
                    loan_maturity=loan_maturity,
                    new_debt_service_by_firm=new_debt_service_by_firm,
                )

            credit_restrictions = np.minimum.reduce(
                (
                    capital_stock_collateral_restrictions,
                    return_on_equity_restrictions,
                    return_on_assets_restrictions,
                    dscr_restrictions,
                )
            )
        elif loan_type == LoanTypes.HOUSEHOLD_CONSUMPTION_LOAN:
            loan_to_income_restrictions = (
                banks.parameters.household_consumption_loans_loan_to_income_ratio
                * 0.5
                * (
                    households.ts.prev("income")[agents_with_demand]
                    + households.ts.current("income")[agents_with_demand]
                )
                - households.ts.current("debt")[agents_with_demand]
                - new_credit_by_household[agents_with_demand]
            )
            credit_restrictions = loan_to_income_restrictions
        elif loan_type == LoanTypes.MORTGAGE:
            loan_to_income_restrictions = (
                banks.parameters.mortgage_loan_to_income_ratio
                * 0.5
                * (
                    households.ts.prev("income")[agents_with_demand]
                    + households.ts.current("income")[agents_with_demand]
                )
                - households.ts.current("debt")[agents_with_demand]
                - new_credit_by_household[agents_with_demand]
            )
            loan_to_value_restrictions = (
                banks.parameters.mortgage_loan_to_value_ratio
                / (1 - banks.parameters.mortgage_loan_to_value_ratio)
                * households.ts.current("wealth_financial_assets")[agents_with_demand]
            )

            credit_restrictions = np.minimum(loan_to_income_restrictions, loan_to_value_restrictions)
        else:
            raise ValueError("Unknown loan type", loan_type)
        capacities = np.maximum(
            0.0,
            np.minimum(target_credit[agents_with_demand], credit_restrictions),
        )
        capacities_sum = capacities.sum()

        diagnostics_enabled = append_firm_diagnostics and loan_type in (
            LoanTypes.FIRM_SHORT_TERM_LOAN,
            LoanTypes.FIRM_LONG_TERM_LOAN,
        )
        if diagnostics_enabled:
            n_firms = firms.ts.current("n_firms")
            demand_full = np.asarray(target_credit, dtype=float)
            collateral_full = np.zeros(n_firms)
            capacity_full = np.zeros(n_firms)
            dscr_full = _nan_array(n_firms)
            if agents_with_demand.size:
                collateral_full[agents_with_demand] = _clip_nonneg(capital_stock_collateral_restrictions)
                capacity_full[agents_with_demand] = capacities
                if banks.parameters.enable_firm_loans_dscr_restriction:
                    dscr_full[agents_with_demand] = _clip_nonneg(dscr_restrictions)

            car_room = np.maximum(
                0.0,
                banks.ts.current("equity") / banks.parameters.capital_adequacy_ratio
                - banks.ts.current("total_outstanding_loans")
                - new_credit_by_bank,
            )
            pref_room = np.maximum(0.0, max_supply_based_on_preferences - new_preference_credit_by_bank)
            pref_likely_binding = bool(
                self.consider_loan_type_fractions and np.isfinite(pref_room.sum()) and pref_room.sum() < car_room.sum()
            )

            def _append_firm_diagnostics(granted_by_firm: np.ndarray) -> None:
                eps = 1e-9
                reason = np.full(n_firms, _BINDING_NO_DEMAND, dtype=float)
                amount = np.zeros(n_firms, dtype=float)

                has_demand = demand_full > 0.0
                fully_granted = has_demand & (granted_by_firm >= demand_full - eps)
                reason[fully_granted] = _BINDING_NOT_BINDING
                amount[fully_granted] = demand_full[fully_granted]

                unmet = has_demand & ~fully_granted

                coll_cap = collateral_full
                dscr_cap_eff = np.where(np.isfinite(dscr_full), dscr_full, np.inf)

                roa_cap_eff = np.full(n_firms, np.inf)
                if banks.parameters.enable_firm_loans_return_on_assets_restriction and agents_with_demand.size:
                    roa_cap_eff[agents_with_demand] = np.where(
                        np.isfinite(return_on_assets_restrictions) & (return_on_assets_restrictions <= 0.0),
                        0.0,
                        np.inf,
                    )

                roe_cap_eff = np.full(n_firms, np.inf)
                if banks.parameters.enable_firm_loans_return_on_equity_restriction and agents_with_demand.size:
                    roe_cap_eff[agents_with_demand] = _clip_nonneg(return_on_equity_restrictions)

                borrower_min = np.minimum.reduce((coll_cap, dscr_cap_eff, roa_cap_eff, roe_cap_eff))
                borrower_binds = unmet & (borrower_min < demand_full - eps)

                # Tie-break: collateral -> dscr -> roa -> roe
                coll_binds = (
                    borrower_binds
                    & (coll_cap < demand_full - eps)
                    & (coll_cap <= dscr_cap_eff)
                    & (coll_cap <= roa_cap_eff)
                    & (coll_cap <= roe_cap_eff)
                )
                reason[coll_binds] = _BINDING_COLLATERAL
                amount[coll_binds] = coll_cap[coll_binds]

                dscr_binds = (
                    borrower_binds
                    & ~coll_binds
                    & (dscr_cap_eff < demand_full - eps)
                    & (dscr_cap_eff <= roa_cap_eff)
                    & (dscr_cap_eff <= roe_cap_eff)
                )
                reason[dscr_binds] = _BINDING_DSCR
                amount[dscr_binds] = dscr_full[dscr_binds]

                roa_binds = (
                    borrower_binds
                    & ~coll_binds
                    & ~dscr_binds
                    & (roa_cap_eff < demand_full - eps)
                    & (roa_cap_eff <= roe_cap_eff)
                )
                reason[roa_binds] = _BINDING_ROA
                amount[roa_binds] = 0.0

                roe_binds = borrower_binds & ~coll_binds & ~dscr_binds & ~roa_binds & (roe_cap_eff < demand_full - eps)
                reason[roe_binds] = _BINDING_ROE
                amount[roe_binds] = roe_cap_eff[roe_binds]

                bank_binds = unmet & ~borrower_binds
                if np.any(bank_binds):
                    reason[bank_binds] = _BINDING_BANK_PREF if pref_likely_binding else _BINDING_BANK_CAR
                    amount[bank_binds] = granted_by_firm[bank_binds]

                other = unmet & (reason == _BINDING_NO_DEMAND)
                reason[other] = _BINDING_OTHER
                amount[other] = granted_by_firm[other]

                if loan_type == LoanTypes.FIRM_SHORT_TERM_LOAN:
                    _append_vector(firms.ts, "credit_market_firm_st_capacity", capacity_full)
                    _append_vector(firms.ts, "credit_market_firm_st_collateral_cap", collateral_full)
                    _append_vector(firms.ts, "credit_market_firm_st_dscr_cap", dscr_full)
                    _append_vector(firms.ts, "credit_market_firm_st_binding_reason", reason)
                    _append_vector(firms.ts, "credit_market_firm_st_binding_amount", amount)
                else:
                    _append_vector(firms.ts, "credit_market_firm_lt_capacity", capacity_full)
                    _append_vector(firms.ts, "credit_market_firm_lt_collateral_cap", collateral_full)
                    _append_vector(firms.ts, "credit_market_firm_lt_dscr_cap", dscr_full)
                    _append_vector(firms.ts, "credit_market_firm_lt_binding_reason", reason)
                    _append_vector(firms.ts, "credit_market_firm_lt_binding_amount", amount)

        if capacities_sum == 0.0:
            if diagnostics_enabled:
                _append_firm_diagnostics(np.zeros(n_firms))
            return new_loans
        capacities_weights = capacities / capacities_sum

        # Determine total supply and priorities
        supply = np.maximum(
            0.0,
            np.minimum(
                banks.ts.current("equity") / banks.parameters.capital_adequacy_ratio
                - banks.ts.current("total_outstanding_loans")
                - new_credit_by_bank,
                max_supply_based_on_preferences - new_preference_credit_by_bank,
            ),
        )
        supply_sum = supply.sum()
        if supply_sum == 0.0:
            if diagnostics_enabled:
                _append_firm_diagnostics(np.zeros(n_firms))
            return new_loans
        supply_weights = supply / supply_sum
        if self.creditor_selection_is_deterministic:
            creditor_priorities = self.get_creditor_priorities_deterministic(
                self.interest_rates_selection_temperature,
                interest_rates=banks_ir,
            )
        else:
            creditor_priorities = self.get_creditor_priorities_stochastic(
                self.interest_rates_selection_temperature,
                interest_rates=banks_ir,
            )

        # Hand out loans
        if supply_sum >= capacities_sum:
            granted_loans_by_banks = self.fill_buckets(
                capacities=supply,
                fill_amount=capacities_sum,
                priorities=creditor_priorities,
                minimum_fill=self.creditor_minimum_fill,
            )
            # Fix NumPy advanced indexing dimension mismatch by assigning row-by-row
            loan_matrix = np.outer(granted_loans_by_banks, capacities_weights)
            for bank_idx in range(len(granted_loans_by_banks)):
                new_loans[0, bank_idx, agents_with_demand] = loan_matrix[bank_idx, :]
        else:
            received_loans_by_debtors = self.fill_buckets(
                capacities=capacities,
                fill_amount=supply_sum,
                priorities=debtor_priorities,
                minimum_fill=self.debtor_minimum_fill,
            )
            # Fix NumPy advanced indexing dimension mismatch by assigning row-by-row
            loan_matrix = np.outer(supply_weights, received_loans_by_debtors)
            for bank_idx in range(len(supply_weights)):
                new_loans[0, bank_idx, agents_with_demand] = loan_matrix[bank_idx, :]
        new_loans[1, :, agents_with_demand] = np.where(
            new_loans[0, :, agents_with_demand] > 0.0,
            banks_ir[:, np.newaxis],
            0.0,
        )
        new_loans[2, :, agents_with_demand] = _annuity_payment(
            new_loans[0, :, agents_with_demand],
            banks_ir[:, np.newaxis],
            loan_maturity,
        )
        if (
            new_debt_service_by_firm is not None
            and (loan_type == LoanTypes.FIRM_SHORT_TERM_LOAN or loan_type == LoanTypes.FIRM_LONG_TERM_LOAN)
            and banks.parameters.enable_firm_loans_dscr_restriction
        ):
            principal_by_firm = new_loans[0].sum(axis=0)
            new_debt_service_by_firm += principal_by_firm * _annuity_payment_factor(
                underwriting_rate,
                loan_maturity,
            )

        if diagnostics_enabled:
            _append_firm_diagnostics(new_loans[0].sum(axis=0))

        return new_loans

    @staticmethod
    @njit(cache=True)
    def get_debtor_priorities(n_agents: int) -> np.ndarray:
        """Generate random priorities for debtors.

        Creates a random permutation of indices to determine the order in which
        debtors are processed during credit allocation.

        Args:
            n_agents (int): Number of debtors to generate priorities for

        Returns:
            np.ndarray: Random permutation of indices [0, n_agents-1]
        """
        return np.random.choice(n_agents, n_agents, replace=False)

    @staticmethod
    @njit(cache=True)
    def get_creditor_priorities_deterministic(
        interest_rates_selection_temperature: float,
        interest_rates: np.ndarray,
    ) -> np.ndarray:
        """Generate deterministic priorities for creditors based on interest rates.

        Creates a deterministic ordering of creditors based on their offered interest
        rates, with lower rates getting higher priority. The temperature parameter
        controls how strongly the ordering depends on rate differences.

        Args:
            interest_rates_selection_temperature (float): Sensitivity to rate differences
            interest_rates (np.ndarray): Array of interest rates offered by creditors

        Returns:
            np.ndarray: Indices sorted by priority (highest to lowest)
        """
        distribution = np.exp(-interest_rates_selection_temperature * interest_rates)
        return np.argsort(distribution)[::-1]

    @staticmethod
    def get_creditor_priorities_stochastic(
        interest_rates_selection_temperature: float,
        interest_rates: np.ndarray,
    ) -> np.ndarray:
        """Generate stochastic priorities for creditors based on interest rates.

        Creates a random ordering of creditors where the probability of selection
        is influenced by their offered interest rates. Lower rates lead to higher
        probability of being selected earlier.

        Args:
            interest_rates_selection_temperature (float): Sensitivity to rate differences
            interest_rates (np.ndarray): Array of interest rates offered by creditors

        Returns:
            np.ndarray: Random permutation of indices weighted by interest rates
        """
        distribution = np.exp(-interest_rates_selection_temperature * interest_rates)
        return np.random.choice(
            len(distribution),
            len(distribution),
            replace=False,
            p=distribution / np.sum(distribution),
        )

    @staticmethod
    def invert_permutation(p: np.ndarray) -> np.ndarray:
        """Invert a permutation array.

        Given a permutation array p where p[i] gives the new position of element i,
        returns the inverse permutation s where s[p[i]] = i.

        Args:
            p (np.ndarray): Permutation array to invert

        Returns:
            np.ndarray: Inverse permutation array
        """
        s = np.empty_like(p)
        s[p] = np.arange(p.size)
        return s

    def fill_buckets(
        self,
        capacities: np.ndarray,
        fill_amount: float,
        priorities: np.ndarray,
        minimum_fill: float,
    ) -> np.ndarray:
        """Distribute a total amount across buckets using the water bucket algorithm.

        This method implements the core water bucket distribution algorithm, which
        allocates a total amount across multiple buckets (recipients) while respecting:
        - Individual bucket capacities
        - Priority ordering
        - Minimum fill requirements

        The algorithm works like pouring water into a series of buckets arranged
        by priority, where each bucket has a maximum capacity. The water (total amount)
        flows through the buckets in order until fully distributed.

        Args:
            capacities (np.ndarray): Maximum capacity of each bucket
            fill_amount (float): Total amount to distribute
            priorities (np.ndarray): Priority ordering of buckets
            minimum_fill (float): Minimum fraction each bucket should receive if possible

        Returns:
            np.ndarray: Amount allocated to each bucket, respecting original ordering

        Note:
            The algorithm proceeds in two stages:
            1. If minimum_fill > 0, first ensures each bucket gets its minimum share
            2. Then distributes remaining amount according to priorities and capacities
        """
        if np.sum(capacities) == np.sum(capacities) + 1:
            return np.full_like(capacities, fill_amount / len(capacities))
        if np.sum(capacities) == 0:
            return np.zeros(capacities.shape)
        capacities_sorted = capacities[priorities]
        filled_capacities = np.zeros(capacities_sorted.shape)
        if minimum_fill > 0.0:
            filled_capacities += np.minimum(
                capacities_sorted,
                capacities_sorted / np.sum(capacities_sorted) * minimum_fill * fill_amount,
            )
        filled_ind = np.where(
            (capacities_sorted - filled_capacities).cumsum() < fill_amount - np.sum(filled_capacities)
        )[0]
        filled_capacities[filled_ind] = capacities_sorted[filled_ind]
        if len(filled_ind) < len(filled_capacities):
            filled_capacities[len(filled_ind)] += fill_amount - np.sum(filled_capacities)
            filled_capacities[len(filled_ind)] = min(
                filled_capacities[len(filled_ind)],
                capacities_sorted[len(filled_ind)],
            )
        return filled_capacities[self.invert_permutation(priorities)]
