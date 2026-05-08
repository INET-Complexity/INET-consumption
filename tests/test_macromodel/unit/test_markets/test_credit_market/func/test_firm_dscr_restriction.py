"""Tests for firm-loan DSCR restrictions in WaterBucket credit clearing."""

import numpy as np

from macromodel.markets.credit_market.func.clearing import (
    WaterBucketCreditMarketClearer,
    _firm_dscr_underwriting_rate,
)
from macromodel.markets.credit_market.types_of_loans import LoanTypes


def _credit_market_clearer():
    return WaterBucketCreditMarketClearer(
        allow_short_term_firm_loans=True,
        allow_household_loans=False,
        firms_max_number_of_banks_visiting=3,
        households_max_number_of_banks_visiting=3,
        consider_loan_type_fractions=False,
        credit_supply_temperature=1.0,
        interest_rates_selection_temperature=1.0,
        creditor_selection_is_deterministic=True,
        creditor_minimum_fill=False,
        debtor_minimum_fill=False,
    )


def _set_bank_supply_and_rates(test_banks, rate: float = 0.05) -> None:
    n_banks = test_banks.ts.current("n_banks")
    test_banks.ts.override_current("equity", np.full(n_banks, 1.0e12))
    test_banks.ts.override_current("total_outstanding_loans", np.zeros(n_banks))
    test_banks.ts.override_current("interest_rates_on_short_term_firm_loans", np.full(n_banks, rate))
    test_banks.ts.override_current("interest_rates_on_long_term_firm_loans", np.full(n_banks, rate))


def _set_single_firm_case(
    test_firms,
    target_short_term_credit: float = 0.0,
    target_long_term_credit: float = 0.0,
    cfads: float = 0.0,
    expected_profits: float = 1.0e6,
) -> None:
    n_firms = test_firms.ts.current("n_firms")
    first_only = np.zeros(n_firms)
    first_only[0] = 1.0

    test_firms.ts.override_current("target_short_term_credit", first_only * target_short_term_credit)
    test_firms.ts.override_current("target_long_term_credit", first_only * target_long_term_credit)
    test_firms.ts.override_current("capital_inputs_stock_value", first_only * 1.0e9)
    test_firms.ts.override_current("debt", np.zeros(n_firms))
    test_firms.ts.override_current("deposits", np.zeros(n_firms))
    test_firms.ts.override_current("expected_profits", first_only * expected_profits)
    test_firms.ts.override_current("interest_paid_on_loans", np.zeros(n_firms))
    test_firms.ts.override_current("debt_installments", np.zeros(n_firms))

    nominal_amount_sold = np.zeros(n_firms)
    nominal_amount_sold[0] = cfads
    test_firms.ts.override_current("nominal_amount_sold_in_lcu", nominal_amount_sold)
    test_firms.ts.override_current("total_wage", np.zeros(n_firms))
    test_firms.ts.override_current("used_intermediate_inputs_costs", np.zeros(n_firms))
    test_firms.ts.override_current("used_capital_inputs_costs", np.zeros(n_firms))
    test_firms.ts.override_current("taxes_paid_on_production", np.zeros(n_firms))


def test_roa_roe_switches_allow_lending_when_legacy_profitability_gate_would_block(test_banks, test_firms):
    clearer = _credit_market_clearer()
    _set_bank_supply_and_rates(test_banks)
    _set_single_firm_case(test_firms, target_long_term_credit=10_000.0, expected_profits=-1.0)

    n_banks = test_banks.ts.current("n_banks")
    n_firms = test_firms.ts.current("n_firms")
    max_supply = np.full(n_banks, np.inf)

    blocked = clearer.clear_loans(
        banks=test_banks,
        firms=test_firms,
        households=None,
        loan_type=LoanTypes.FIRM_LONG_TERM_LOAN,
        new_credit_by_bank=np.zeros(n_banks),
        new_credit_by_firm=np.zeros(n_firms),
        new_credit_by_household=np.zeros(1),
        max_supply_based_on_preferences=max_supply,
    )
    assert blocked[0, :, 0].sum() == 0.0

    test_banks.parameters.enable_firm_loans_return_on_assets_restriction = False
    test_banks.parameters.enable_firm_loans_return_on_equity_restriction = False
    allowed = clearer.clear_loans(
        banks=test_banks,
        firms=test_firms,
        households=None,
        loan_type=LoanTypes.FIRM_LONG_TERM_LOAN,
        new_credit_by_bank=np.zeros(n_banks),
        new_credit_by_firm=np.zeros(n_firms),
        new_credit_by_household=np.zeros(1),
        max_supply_based_on_preferences=max_supply,
    )

    assert allowed[0, :, 0].sum() > 0.0


def test_dscr_caps_principal_and_keeps_actual_loan_output_semantics(test_banks, test_firms):
    clearer = _credit_market_clearer()
    _set_bank_supply_and_rates(test_banks, rate=0.05)
    _set_single_firm_case(test_firms, target_long_term_credit=1.0e6, cfads=1_250.0)

    test_banks.parameters.enable_firm_loans_return_on_assets_restriction = False
    test_banks.parameters.enable_firm_loans_return_on_equity_restriction = False
    test_banks.parameters.enable_firm_loans_dscr_restriction = True
    test_banks.parameters.firm_loans_min_dscr = 1.25
    test_banks.parameters.firm_loans_cfads_window = 1
    test_banks.parameters.firm_loans_cfads_haircut = 1.0
    test_banks.parameters.long_term_firm_loan_maturity = 8

    n_banks = test_banks.ts.current("n_banks")
    n_firms = test_firms.ts.current("n_firms")
    result = clearer.clear_loans(
        banks=test_banks,
        firms=test_firms,
        households=None,
        loan_type=LoanTypes.FIRM_LONG_TERM_LOAN,
        new_credit_by_bank=np.zeros(n_banks),
        new_credit_by_firm=np.zeros(n_firms),
        new_credit_by_household=np.zeros(1),
        max_supply_based_on_preferences=np.full(n_banks, np.inf),
    )

    expected_principal = (1_250.0 / 1.25) / (0.05 + 1.0 / 8.0)
    assert np.isclose(result[0, :, 0].sum(), expected_principal)
    assert np.allclose(
        result[1, :, 0],
        test_banks.ts.current("interest_rates_on_long_term_firm_loans") * result[0, :, 0],
    )
    assert np.allclose(result[2, :, 0], result[0, :, 0] / 8.0)


def test_dscr_underwriting_rate_uses_max_finite_bank_rate():
    bank_rates = np.array([0.02, np.nan, 0.08, -0.01, 0.04])

    assert _firm_dscr_underwriting_rate(bank_rates, "max_bank_rate") == 0.08


def test_zero_cfads_blocks_dscr_lending(test_banks, test_firms):
    clearer = _credit_market_clearer()
    _set_bank_supply_and_rates(test_banks)
    _set_single_firm_case(test_firms, target_long_term_credit=10_000.0, cfads=0.0)

    test_banks.parameters.enable_firm_loans_return_on_assets_restriction = False
    test_banks.parameters.enable_firm_loans_return_on_equity_restriction = False
    test_banks.parameters.enable_firm_loans_dscr_restriction = True
    test_banks.parameters.firm_loans_cfads_window = 1

    n_banks = test_banks.ts.current("n_banks")
    n_firms = test_firms.ts.current("n_firms")
    result = clearer.clear_loans(
        banks=test_banks,
        firms=test_firms,
        households=None,
        loan_type=LoanTypes.FIRM_LONG_TERM_LOAN,
        new_credit_by_bank=np.zeros(n_banks),
        new_credit_by_firm=np.zeros(n_firms),
        new_credit_by_household=np.zeros(1),
        max_supply_based_on_preferences=np.full(n_banks, np.inf),
    )

    assert result[0, :, 0].sum() == 0.0


def test_non_finite_cfads_blocks_dscr_lending(test_banks, test_firms):
    clearer = _credit_market_clearer()
    _set_bank_supply_and_rates(test_banks)
    _set_single_firm_case(test_firms, target_long_term_credit=10_000.0, cfads=np.nan)

    test_banks.parameters.enable_firm_loans_return_on_assets_restriction = False
    test_banks.parameters.enable_firm_loans_return_on_equity_restriction = False
    test_banks.parameters.enable_firm_loans_dscr_restriction = True
    test_banks.parameters.firm_loans_cfads_window = 1

    n_banks = test_banks.ts.current("n_banks")
    n_firms = test_firms.ts.current("n_firms")
    result = clearer.clear_loans(
        banks=test_banks,
        firms=test_firms,
        households=None,
        loan_type=LoanTypes.FIRM_LONG_TERM_LOAN,
        new_credit_by_bank=np.zeros(n_banks),
        new_credit_by_firm=np.zeros(n_firms),
        new_credit_by_household=np.zeros(1),
        max_supply_based_on_preferences=np.full(n_banks, np.inf),
    )

    assert result[0, :, 0].sum() == 0.0


def test_short_term_lending_consumes_service_room_before_long_term_lending(test_banks, test_firms):
    clearer = _credit_market_clearer()
    _set_bank_supply_and_rates(test_banks, rate=0.05)
    _set_single_firm_case(
        test_firms,
        target_short_term_credit=1.0e6,
        target_long_term_credit=1.0e6,
        cfads=1_250.0,
    )

    test_banks.parameters.enable_firm_loans_return_on_assets_restriction = False
    test_banks.parameters.enable_firm_loans_return_on_equity_restriction = False
    test_banks.parameters.enable_firm_loans_dscr_restriction = True
    test_banks.parameters.firm_loans_min_dscr = 1.25
    test_banks.parameters.firm_loans_cfads_window = 1
    test_banks.parameters.short_term_firm_loan_maturity = 1
    test_banks.parameters.long_term_firm_loan_maturity = 8

    n_banks = test_banks.ts.current("n_banks")
    n_firms = test_firms.ts.current("n_firms")
    new_debt_service_by_firm = np.zeros(n_firms)
    max_supply = np.full(n_banks, np.inf)

    short_term = clearer.clear_loans(
        banks=test_banks,
        firms=test_firms,
        households=None,
        loan_type=LoanTypes.FIRM_SHORT_TERM_LOAN,
        new_credit_by_bank=np.zeros(n_banks),
        new_credit_by_firm=np.zeros(n_firms),
        new_credit_by_household=np.zeros(1),
        max_supply_based_on_preferences=max_supply,
        new_debt_service_by_firm=new_debt_service_by_firm,
    )
    long_term = clearer.clear_loans(
        banks=test_banks,
        firms=test_firms,
        households=None,
        loan_type=LoanTypes.FIRM_LONG_TERM_LOAN,
        new_credit_by_bank=np.zeros(n_banks),
        new_credit_by_firm=np.zeros(n_firms),
        new_credit_by_household=np.zeros(1),
        max_supply_based_on_preferences=max_supply,
        new_debt_service_by_firm=new_debt_service_by_firm,
    )

    assert short_term[0, :, 0].sum() > 0.0
    assert np.isclose(new_debt_service_by_firm[0], 1_250.0 / 1.25)
    assert long_term[0, :, 0].sum() == 0.0


def test_clear_threads_short_term_service_room_into_long_term_lending(test_banks, test_firms, test_households):
    clearer = _credit_market_clearer()
    _set_bank_supply_and_rates(test_banks, rate=0.05)
    _set_single_firm_case(
        test_firms,
        target_short_term_credit=1.0e6,
        target_long_term_credit=1.0e6,
        cfads=1_250.0,
    )

    test_banks.parameters.enable_firm_loans_return_on_assets_restriction = False
    test_banks.parameters.enable_firm_loans_return_on_equity_restriction = False
    test_banks.parameters.enable_firm_loans_dscr_restriction = True
    test_banks.parameters.firm_loans_min_dscr = 1.25
    test_banks.parameters.firm_loans_cfads_window = 1
    test_banks.parameters.short_term_firm_loan_maturity = 1
    test_banks.parameters.long_term_firm_loan_maturity = 8

    n_households = test_households.ts.current("n_households")
    test_households.ts.override_current("target_consumption_loans", np.zeros(n_households))
    test_households.ts.override_current("target_mortgage", np.zeros(n_households))

    short_term, long_term, _, _ = clearer.clear(
        banks=test_banks,
        firms=test_firms,
        households=test_households,
        current_npl_firm_loans=0.0,
        current_npl_hh_cons_loans=0.0,
        current_npl_mortgages=0.0,
    )

    assert short_term[0, :, 0].sum() > 0.0
    assert long_term[0, :, 0].sum() == 0.0
