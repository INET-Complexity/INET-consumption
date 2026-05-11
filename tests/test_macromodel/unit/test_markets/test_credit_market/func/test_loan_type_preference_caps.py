"""Tests for loan-type preference supply caps in credit clearing."""

import numpy as np

from macromodel.markets.credit_market.func.clearing import (
    DefaultCreditMarketClearer,
    WaterBucketCreditMarketClearer,
    _compute_loan_type_preference_caps,
)
from macromodel.markets.credit_market.types_of_loans import LoanTypes


def _waterbucket_clearer(consider_loan_type_fractions: bool = True) -> WaterBucketCreditMarketClearer:
    return WaterBucketCreditMarketClearer(
        allow_short_term_firm_loans=True,
        allow_household_loans=True,
        firms_max_number_of_banks_visiting=3,
        households_max_number_of_banks_visiting=3,
        consider_loan_type_fractions=consider_loan_type_fractions,
        credit_supply_temperature=2.0,
        interest_rates_selection_temperature=1.0,
        creditor_selection_is_deterministic=True,
        creditor_minimum_fill=False,
        debtor_minimum_fill=False,
    )


def _default_clearer(consider_loan_type_fractions: bool = True) -> DefaultCreditMarketClearer:
    return DefaultCreditMarketClearer(
        allow_short_term_firm_loans=True,
        allow_household_loans=True,
        firms_max_number_of_banks_visiting=3,
        households_max_number_of_banks_visiting=3,
        consider_loan_type_fractions=consider_loan_type_fractions,
        credit_supply_temperature=2.0,
        interest_rates_selection_temperature=1.0,
        creditor_selection_is_deterministic=True,
        creditor_minimum_fill=False,
        debtor_minimum_fill=False,
    )


def _set_initial_loan_type_fractions(
    test_banks,
    firm_fraction: float,
    household_consumption_fraction: float,
    mortgage_fraction: float,
) -> None:
    n_banks = test_banks.ts.current("n_banks")
    test_banks.ts.dicts["new_loans_fraction_firms"][0] = np.full(n_banks, firm_fraction)
    test_banks.ts.dicts["new_loans_fraction_hh_cons"][0] = np.full(n_banks, household_consumption_fraction)
    test_banks.ts.dicts["new_loans_fraction_mortgages"][0] = np.full(n_banks, mortgage_fraction)


def _set_bank_lending_room(test_banks, max_car: float) -> None:
    n_banks = test_banks.ts.current("n_banks")
    test_banks.parameters.capital_adequacy_ratio = 0.1
    test_banks.ts.override_current("equity", np.full(n_banks, max_car * test_banks.parameters.capital_adequacy_ratio))
    test_banks.ts.override_current("total_outstanding_loans", np.zeros(n_banks))


def _set_one_firm_one_household_credit_case(test_banks, test_firms, test_households) -> None:
    n_banks = test_banks.ts.current("n_banks")
    n_firms = test_firms.ts.current("n_firms")
    n_households = test_households.ts.current("n_households")
    first_firm = np.zeros(n_firms)
    first_firm[0] = 1.0
    first_household = np.zeros(n_households)
    first_household[0] = 1.0

    test_banks.ts.override_current("interest_rates_on_short_term_firm_loans", np.full(n_banks, 0.05))
    test_banks.ts.override_current("interest_rates_on_long_term_firm_loans", np.full(n_banks, 0.05))
    test_banks.ts.override_current("interest_rates_on_household_consumption_loans", np.full(n_banks, 0.05))
    test_banks.parameters.enable_firm_loans_return_on_assets_restriction = False
    test_banks.parameters.enable_firm_loans_return_on_equity_restriction = False
    test_banks.parameters.enable_firm_loans_dscr_restriction = False
    test_banks.parameters.firm_loans_return_on_assets_ratio = 0.04
    test_banks.parameters.firm_loans_return_on_equity_ratio = 0.15

    test_firms.ts.override_current("target_short_term_credit", first_firm * 60.0)
    test_firms.ts.override_current("target_long_term_credit", np.zeros(n_firms))
    test_firms.ts.override_current("capital_inputs_stock_value", first_firm * 1.0e6)
    test_firms.ts.override_current("debt", np.zeros(n_firms))
    test_firms.ts.override_current("deposits", np.zeros(n_firms))
    test_firms.ts.override_current("expected_profits", first_firm * 50_000.0)

    test_households.ts.override_current("target_consumption_loans", first_household * 100.0)
    test_households.ts.override_current("target_mortgage", np.zeros(n_households))
    test_households.ts.override_current("income", first_household * 1.0e6)
    test_households.ts.override_current("debt", np.zeros(n_households))


def test_preference_caps_convert_fractions_to_currency_units(test_banks):
    _set_bank_lending_room(test_banks, max_car=1.06e13)
    _set_initial_loan_type_fractions(test_banks, 0.691, 0.054, 0.255)

    firm_cap, household_consumption_cap, mortgage_cap = _compute_loan_type_preference_caps(
        banks=test_banks,
        current_npl_firm_loans=0.0,
        current_npl_hh_cons_loans=0.0,
        current_npl_mortgages=0.0,
        credit_supply_temperature=2.0,
    )

    assert np.allclose(firm_cap, 0.691 * 1.06e13)
    assert np.allclose(household_consumption_cap, 0.054 * 1.06e13)
    assert np.allclose(mortgage_cap, 0.255 * 1.06e13)
    assert np.all(firm_cap > 1.0)


def test_preference_caps_with_npl_adjustment_sum_to_bank_lending_room(test_banks):
    _set_bank_lending_room(test_banks, max_car=1.0e6)
    _set_initial_loan_type_fractions(test_banks, 0.5, 0.3, 0.2)

    caps = _compute_loan_type_preference_caps(
        banks=test_banks,
        current_npl_firm_loans=0.1,
        current_npl_hh_cons_loans=0.2,
        current_npl_mortgages=0.0,
        credit_supply_temperature=2.0,
    )

    assert np.allclose(sum(caps), np.full(test_banks.ts.current("n_banks"), 1.0e6))


def test_preference_caps_return_finite_zero_when_no_lending_room_or_no_weights(test_banks):
    _set_bank_lending_room(test_banks, max_car=0.0)
    _set_initial_loan_type_fractions(test_banks, 0.5, 0.3, 0.2)

    no_room_caps = _compute_loan_type_preference_caps(
        banks=test_banks,
        current_npl_firm_loans=0.0,
        current_npl_hh_cons_loans=0.0,
        current_npl_mortgages=0.0,
        credit_supply_temperature=2.0,
    )
    assert all(np.all(cap == 0.0) for cap in no_room_caps)
    assert all(np.all(np.isfinite(cap)) for cap in no_room_caps)

    _set_bank_lending_room(test_banks, max_car=1.0e6)
    _set_initial_loan_type_fractions(test_banks, 0.0, 0.0, 0.0)

    no_weight_caps = _compute_loan_type_preference_caps(
        banks=test_banks,
        current_npl_firm_loans=0.0,
        current_npl_hh_cons_loans=0.0,
        current_npl_mortgages=0.0,
        credit_supply_temperature=2.0,
    )
    assert all(np.all(cap == 0.0) for cap in no_weight_caps)
    assert all(np.all(np.isfinite(cap)) for cap in no_weight_caps)


def test_waterbucket_preference_caps_are_scaled_in_currency_units(test_banks, test_firms, test_households):
    clearer = _waterbucket_clearer()
    _set_bank_lending_room(test_banks, max_car=1.0e6)
    _set_initial_loan_type_fractions(test_banks, 0.5, 0.3, 0.2)

    n_banks = test_banks.ts.current("n_banks")
    n_firms = test_firms.ts.current("n_firms")
    n_households = test_households.ts.current("n_households")
    first_firm = np.zeros(n_firms)
    first_firm[0] = 1.0

    test_banks.ts.override_current("interest_rates_on_short_term_firm_loans", np.full(n_banks, 0.05))
    test_banks.parameters.enable_firm_loans_return_on_assets_restriction = False
    test_banks.parameters.enable_firm_loans_return_on_equity_restriction = False
    test_banks.parameters.enable_firm_loans_dscr_restriction = False
    test_firms.ts.override_current("target_short_term_credit", first_firm * 100.0)
    test_firms.ts.override_current("target_long_term_credit", np.zeros(n_firms))
    test_firms.ts.override_current("capital_inputs_stock_value", first_firm * 1.0e6)
    test_firms.ts.override_current("debt", np.zeros(n_firms))
    test_firms.ts.override_current("deposits", np.zeros(n_firms))
    test_households.ts.override_current("target_consumption_loans", np.zeros(n_households))
    test_households.ts.override_current("target_mortgage", np.zeros(n_households))

    short_term, _, _, _ = clearer.clear(
        banks=test_banks,
        firms=test_firms,
        households=test_households,
        current_npl_firm_loans=0.0,
        current_npl_hh_cons_loans=0.0,
        current_npl_mortgages=0.0,
    )

    assert np.isclose(short_term[0, :, 0].sum(), 100.0)


def test_waterbucket_type_quota_uses_category_usage_not_total_bank_usage(test_banks, test_firms, test_households):
    clearer = _waterbucket_clearer()
    _set_bank_lending_room(test_banks, max_car=100.0)
    n_banks = test_banks.ts.current("n_banks")
    n_firms = test_firms.ts.current("n_firms")
    n_households = test_households.ts.current("n_households")
    first_household = np.zeros(n_households)
    first_household[0] = 1.0

    test_banks.ts.override_current("interest_rates_on_household_consumption_loans", np.full(n_banks, 0.05))
    test_households.ts.override_current("target_consumption_loans", first_household * 100.0)
    test_households.ts.override_current("income", first_household * 1.0e6)
    test_households.ts.override_current("debt", np.zeros(n_households))

    result = clearer.clear_loans(
        banks=test_banks,
        firms=test_firms,
        households=test_households,
        loan_type=LoanTypes.HOUSEHOLD_CONSUMPTION_LOAN,
        new_credit_by_bank=np.full(n_banks, 60.0),
        new_credit_by_firm=np.zeros(n_firms),
        new_credit_by_household=np.zeros(n_households),
        max_supply_based_on_preferences=np.full(n_banks, 50.0),
        new_preference_credit_by_bank=np.zeros(n_banks),
    )

    assert np.isclose(result[0, :, 0].sum(), 40.0)


def test_waterbucket_clear_keeps_household_quota_after_firm_lending(test_banks, test_firms, test_households):
    clearer = _waterbucket_clearer()
    _set_bank_lending_room(test_banks, max_car=100.0)
    _set_initial_loan_type_fractions(test_banks, 0.6, 0.4, 0.0)
    _set_one_firm_one_household_credit_case(test_banks, test_firms, test_households)

    short_term, _, consumption, _ = clearer.clear(
        banks=test_banks,
        firms=test_firms,
        households=test_households,
        current_npl_firm_loans=0.0,
        current_npl_hh_cons_loans=0.0,
        current_npl_mortgages=0.0,
    )

    assert np.isclose(short_term[0, :, 0].sum(), 60.0)
    assert np.isclose(consumption[0, :, 0].sum(), 40.0)


def test_waterbucket_clears_overdraft_refinance_from_residual_capacity_after_lt(test_banks, test_firms, test_households):
    clearer = _waterbucket_clearer(consider_loan_type_fractions=False)
    _set_bank_lending_room(test_banks, max_car=100.0)

    n_banks = test_banks.ts.current("n_banks")
    n_firms = test_firms.ts.current("n_firms")
    n_households = test_households.ts.current("n_households")
    first_firm = np.zeros(n_firms)
    first_firm[0] = 1.0

    test_banks.ts.override_current("interest_rates_on_short_term_firm_loans", np.full(n_banks, 0.05))
    test_banks.ts.override_current("interest_rates_on_long_term_firm_loans", np.full(n_banks, 0.05))
    test_banks.parameters.enable_firm_loans_return_on_assets_restriction = False
    test_banks.parameters.enable_firm_loans_return_on_equity_restriction = False
    test_banks.parameters.enable_firm_loans_dscr_restriction = False

    test_firms.ts.override_current("target_short_term_credit", first_firm * 1_000.0)
    test_firms.ts.override_current("target_overdraft_refinance_credit", first_firm * 1_000.0)
    test_firms.ts.override_current("ordinary_target_short_term_credit", np.zeros(n_firms))
    test_firms.ts.override_current("target_long_term_credit", first_firm * 80.0)
    test_firms.ts.override_current("capital_inputs_stock_value", first_firm * 1.0e6)
    test_firms.ts.override_current("debt", np.zeros(n_firms))
    test_firms.ts.override_current("deposits", first_firm * -1_000.0)
    test_firms.ts.override_current("expected_profits", first_firm * 50_000.0)

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

    assert np.isclose(long_term[0, :, 0].sum(), 80.0)
    assert np.isclose(short_term[0, :, 0].sum(), 20.0)
    assert np.isclose(clearer._last_received_overdraft_refinance_credit_by_firm[0], 20.0)


def test_default_clearer_keeps_household_quota_after_firm_lending(test_banks, test_firms, test_households):
    clearer = _default_clearer()
    _set_bank_lending_room(test_banks, max_car=100.0)
    _set_initial_loan_type_fractions(test_banks, 0.6, 0.4, 0.0)
    _set_one_firm_one_household_credit_case(test_banks, test_firms, test_households)

    short_term_loans, _, consumption_loans, _ = clearer.clear(
        banks=test_banks,
        firms=test_firms,
        households=test_households,
        current_npl_firm_loans=0.0,
        current_npl_hh_cons_loans=0.0,
        current_npl_mortgages=0.0,
    )

    assert np.isclose(short_term_loans[0, :, 0].sum(), 60.0)
    assert np.isclose(consumption_loans[0, :, 0].sum(), 40.0)
