import numpy as np

from macromodel.markets.credit_market.credit_market import (
    _append_credit_supply_caps_to_banks_ts,
    _compute_credit_supply_caps_by_type,
    _ordinary_short_term_target_for_cap_split,
)
from macromodel.timeseries import TimeSeries


class _BankParams:
    def __init__(self, capital_adequacy_ratio: float):
        self.capital_adequacy_ratio = capital_adequacy_ratio


class _BanksStub:
    def __init__(self, ts: TimeSeries, parameters: _BankParams):
        self.ts = ts
        self.parameters = parameters


def _make_banks_stub(
    *,
    equity: np.ndarray,
    total_outstanding_loans: np.ndarray,
    firms_fraction: np.ndarray,
    hh_cons_fraction: np.ndarray,
    mortgage_fraction: np.ndarray,
    capital_adequacy_ratio: float = 0.1,
    overdrafts_to_households: np.ndarray | None = None,
    overdrafts_to_firms: np.ndarray | None = None,
) -> _BanksStub:
    n_banks = int(equity.size)
    ts = TimeSeries(
        n_banks=n_banks,
        equity=equity,
        total_outstanding_loans=total_outstanding_loans,
        overdrafts_to_households=np.zeros(n_banks) if overdrafts_to_households is None else overdrafts_to_households,
        overdrafts_to_firms=np.zeros(n_banks) if overdrafts_to_firms is None else overdrafts_to_firms,
        new_loans_fraction_firms=firms_fraction,
        new_loans_fraction_hh_cons=hh_cons_fraction,
        new_loans_fraction_mortgages=mortgage_fraction,
        credit_supply_cap_total=np.full(n_banks, np.nan),
        credit_supply_cap_firms=np.full(n_banks, np.nan),
        credit_supply_cap_firms_short_term=np.full(n_banks, np.nan),
        credit_supply_cap_firms_long_term=np.full(n_banks, np.nan),
        credit_supply_cap_households_consumption=np.full(n_banks, np.nan),
        credit_supply_cap_mortgages=np.full(n_banks, np.nan),
        total_credit_supply_cap_total=[np.nan],
        total_credit_supply_cap_firms=[np.nan],
        total_credit_supply_cap_firms_short_term=[np.nan],
        total_credit_supply_cap_firms_long_term=[np.nan],
        total_credit_supply_cap_households_consumption=[np.nan],
        total_credit_supply_cap_mortgages=[np.nan],
    )
    return _BanksStub(ts=ts, parameters=_BankParams(capital_adequacy_ratio=capital_adequacy_ratio))


def test_compute_credit_supply_caps_by_type_matches_hand_calculation_when_temperature_is_zero():
    banks = _make_banks_stub(
        equity=np.array([10.0, 20.0]),
        total_outstanding_loans=np.array([0.0, 0.0]),
        firms_fraction=np.array([0.5, 0.5]),
        hh_cons_fraction=np.array([0.3, 0.3]),
        mortgage_fraction=np.array([0.2, 0.2]),
        capital_adequacy_ratio=0.1,
    )

    caps = _compute_credit_supply_caps_by_type(
        banks=banks,
        current_npl_firm_loans=0.0,
        current_npl_hh_cons_loans=0.0,
        current_npl_mortgages=0.0,
        credit_supply_temperature=0.0,
        total_target_short_term_credit=100.0,
        total_target_long_term_credit=0.0,
    )

    assert np.allclose(caps["total"], np.array([100.0, 200.0]))
    assert np.allclose(caps["firms"], np.array([50.0, 100.0]))
    assert np.allclose(caps["firms_short_term"], np.array([50.0, 100.0]))
    assert np.allclose(caps["firms_long_term"], 0.0)
    assert np.allclose(caps["households_consumption"], np.array([30.0, 60.0]))
    assert np.allclose(caps["mortgages"], np.array([20.0, 40.0]))


def test_compute_credit_supply_caps_by_type_excludes_overdraft_refinance_from_st_lt_split():
    banks = _make_banks_stub(
        equity=np.array([10.0, 20.0]),
        total_outstanding_loans=np.array([0.0, 0.0]),
        firms_fraction=np.array([0.5, 0.5]),
        hh_cons_fraction=np.array([0.3, 0.3]),
        mortgage_fraction=np.array([0.2, 0.2]),
        capital_adequacy_ratio=0.1,
    )

    caps = _compute_credit_supply_caps_by_type(
        banks=banks,
        current_npl_firm_loans=0.0,
        current_npl_hh_cons_loans=0.0,
        current_npl_mortgages=0.0,
        credit_supply_temperature=0.0,
        total_target_short_term_credit=1_000.0,
        total_target_long_term_credit=90.0,
        total_ordinary_target_short_term_credit=10.0,
    )

    assert np.allclose(caps["firms"], np.array([50.0, 100.0]))
    assert np.allclose(caps["firms_short_term"], np.array([5.0, 10.0]))
    assert np.allclose(caps["firms_long_term"], np.array([45.0, 90.0]))


def test_ordinary_short_term_cap_split_excludes_rollover_and_refinance():
    ordinary = _ordinary_short_term_target_for_cap_split(
        target_short_term_credit=np.array([100.0, 80.0, 40.0]),
        ordinary_target_short_term_credit=np.array([0.0, 20.0, 100.0]),
        target_debt_rollover_credit=np.array([100.0, 30.0, 10.0]),
        target_overdraft_refinance_credit=np.array([0.0, 20.0, 10.0]),
    )

    assert np.allclose(ordinary, np.array([0.0, 20.0, 20.0]))


def test_ordinary_short_term_cap_split_keeps_legacy_total_st_when_no_emergency_buckets():
    ordinary = _ordinary_short_term_target_for_cap_split(
        target_short_term_credit=np.array([100.0]),
        ordinary_target_short_term_credit=np.array([0.0]),
        target_debt_rollover_credit=np.array([0.0]),
        target_overdraft_refinance_credit=np.array([0.0]),
    )

    assert np.allclose(ordinary, np.array([100.0]))


def test_compute_credit_supply_caps_by_type_returns_zero_type_caps_when_no_weights():
    banks = _make_banks_stub(
        equity=np.array([10.0, 20.0]),
        total_outstanding_loans=np.array([0.0, 0.0]),
        firms_fraction=np.array([0.0, 0.0]),
        hh_cons_fraction=np.array([0.0, 0.0]),
        mortgage_fraction=np.array([0.0, 0.0]),
        capital_adequacy_ratio=0.1,
    )

    caps = _compute_credit_supply_caps_by_type(
        banks=banks,
        current_npl_firm_loans=0.0,
        current_npl_hh_cons_loans=0.0,
        current_npl_mortgages=0.0,
        credit_supply_temperature=2.0,
        total_target_short_term_credit=0.0,
        total_target_long_term_credit=0.0,
    )

    assert np.allclose(caps["total"], np.array([100.0, 200.0]))
    assert np.allclose(caps["firms"], 0.0)
    assert np.allclose(caps["households_consumption"], 0.0)
    assert np.allclose(caps["mortgages"], 0.0)


def test_append_credit_supply_caps_updates_banks_ts_vectors_and_totals():
    banks = _make_banks_stub(
        equity=np.array([10.0, 20.0]),
        total_outstanding_loans=np.array([0.0, 0.0]),
        firms_fraction=np.array([0.5, 0.5]),
        hh_cons_fraction=np.array([0.3, 0.3]),
        mortgage_fraction=np.array([0.2, 0.2]),
        capital_adequacy_ratio=0.1,
    )

    _append_credit_supply_caps_to_banks_ts(
        banks=banks,
        current_npl_firm_loans=0.0,
        current_npl_hh_cons_loans=0.0,
        current_npl_mortgages=0.0,
        credit_supply_temperature=0.0,
        total_target_short_term_credit=20.0,
        total_target_long_term_credit=80.0,
    )

    assert np.allclose(banks.ts.current("credit_supply_cap_total"), np.array([100.0, 200.0]))
    assert np.allclose(banks.ts.current("credit_supply_cap_firms"), np.array([50.0, 100.0]))
    assert np.allclose(banks.ts.current("credit_supply_cap_firms_short_term"), np.array([10.0, 20.0]))
    assert np.allclose(banks.ts.current("credit_supply_cap_firms_long_term"), np.array([40.0, 80.0]))
    assert np.allclose(banks.ts.current("credit_supply_cap_households_consumption"), np.array([30.0, 60.0]))
    assert np.allclose(banks.ts.current("credit_supply_cap_mortgages"), np.array([20.0, 40.0]))

    assert np.allclose(banks.ts.current("total_credit_supply_cap_total")[0], 300.0)
    assert np.allclose(banks.ts.current("total_credit_supply_cap_firms")[0], 150.0)
    assert np.allclose(banks.ts.current("total_credit_supply_cap_firms_short_term")[0], 30.0)
    assert np.allclose(banks.ts.current("total_credit_supply_cap_firms_long_term")[0], 120.0)
    assert np.allclose(banks.ts.current("total_credit_supply_cap_households_consumption")[0], 90.0)
    assert np.allclose(banks.ts.current("total_credit_supply_cap_mortgages")[0], 60.0)


def test_compute_credit_supply_caps_by_type_returns_zero_when_car_is_fully_binding():
    banks = _make_banks_stub(
        equity=np.array([10.0, 20.0]),
        total_outstanding_loans=np.array([100.0, 100.0]),
        firms_fraction=np.array([0.5, 0.5]),
        hh_cons_fraction=np.array([0.3, 0.3]),
        mortgage_fraction=np.array([0.2, 0.2]),
        capital_adequacy_ratio=0.1,
    )

    caps = _compute_credit_supply_caps_by_type(
        banks=banks,
        current_npl_firm_loans=0.0,
        current_npl_hh_cons_loans=0.0,
        current_npl_mortgages=0.0,
        credit_supply_temperature=0.0,
        total_target_short_term_credit=100.0,
        total_target_long_term_credit=0.0,
    )

    # Bank 0 has exactly exhausted CAR headroom; bank 1 retains 100 units.
    assert np.allclose(caps["total"], np.array([0.0, 100.0]))


def test_compute_credit_supply_caps_by_type_counts_overdrafts_in_car_headroom():
    banks = _make_banks_stub(
        equity=np.array([10.0, 20.0]),
        total_outstanding_loans=np.array([60.0, 100.0]),
        firms_fraction=np.array([0.5, 0.5]),
        hh_cons_fraction=np.array([0.3, 0.3]),
        mortgage_fraction=np.array([0.2, 0.2]),
        capital_adequacy_ratio=0.1,
        overdrafts_to_households=np.array([25.0, 0.0]),
        overdrafts_to_firms=np.array([5.0, 50.0]),
    )

    caps = _compute_credit_supply_caps_by_type(
        banks=banks,
        current_npl_firm_loans=0.0,
        current_npl_hh_cons_loans=0.0,
        current_npl_mortgages=0.0,
        credit_supply_temperature=0.0,
        total_target_short_term_credit=100.0,
        total_target_long_term_credit=0.0,
    )

    # Headroom = equity / CAR - (loans + household overdrafts + firm overdrafts).
    assert np.allclose(caps["total"], np.array([10.0, 50.0]))


def test_available_tranche_supply_counts_firm_overdrafts_in_the_exposure():
    from macromodel.markets.credit_market.func.clearing import compute_available_bank_supply_for_credit_tranche

    banks = _make_banks_stub(
        equity=np.array([10.0, 10.0]),
        total_outstanding_loans=np.array([80.0, 80.0]),
        firms_fraction=np.array([0.5, 0.5]),
        hh_cons_fraction=np.array([0.3, 0.3]),
        mortgage_fraction=np.array([0.2, 0.2]),
        capital_adequacy_ratio=0.1,
        overdrafts_to_firms=np.array([20.0, 15.0]),
    )
    supply = compute_available_bank_supply_for_credit_tranche(
        banks=banks,
        new_credit_by_bank=np.zeros(2),
        max_supply_based_on_preferences=np.full(2, np.inf),
        new_preference_credit_by_bank=np.zeros(2),
    )
    np.testing.assert_allclose(supply, [0.0, 5.0])


def _term_out_case(equity, rates, booked=(100.0, 100.0)):
    """Two banks; firm 0 (bank 0) and firm 1 (bank 1) each overdrawn by 100; firm 1 cannot borrow."""
    from types import SimpleNamespace

    from macromodel.markets.credit_market.func.clearing import WaterBucketCreditMarketClearer

    class _Ts:
        def __init__(self, **values):
            self.values = values

        def current(self, name):
            return self.values[name]

    banks = SimpleNamespace(
        parameters=SimpleNamespace(
            capital_adequacy_ratio=0.1,
            short_term_firm_loan_maturity=4,
            firm_loans_capital_stock_collateral_ratio=1.0,
            enable_firm_loans_return_on_equity_restriction=False,
            enable_firm_loans_return_on_assets_restriction=False,
            enable_firm_loans_dscr_restriction=False,
        ),
        ts=_Ts(
            n_banks=2,
            equity=np.asarray(equity, dtype=float),
            total_outstanding_loans=np.zeros(2),
            overdrafts_to_households=np.zeros(2),
            overdrafts_to_firms=np.asarray(booked, dtype=float),
            interest_rates_on_short_term_firm_loans=np.asarray(rates, dtype=float),
        ),
    )
    firms = SimpleNamespace(
        states={"Corresponding Bank ID": np.array([0, 1])},
        ts=_Ts(
            n_firms=2,
            target_short_term_credit=np.array([100.0, 100.0]),
            capital_inputs_stock_value=np.array([1e6, 100.0]),
            debt=np.zeros(2),
            deposits=np.array([-100.0, -100.0]),
        ),
    )
    clearer = WaterBucketCreditMarketClearer(
        allow_short_term_firm_loans=True,
        allow_household_loans=False,
        firms_max_number_of_banks_visiting=2,
        households_max_number_of_banks_visiting=2,
        consider_loan_type_fractions=False,
        credit_supply_temperature=1.0,
        interest_rates_selection_temperature=1.0,
        creditor_selection_is_deterministic=True,
        creditor_minimum_fill=False,
        debtor_minimum_fill=False,
    )
    from macromodel.markets.credit_market.types_of_loans import LoanTypes

    return clearer, clearer.clear_loans(
        banks=banks,
        firms=firms,
        households=None,
        loan_type=LoanTypes.FIRM_SHORT_TERM_LOAN,
        new_credit_by_bank=np.zeros(2),
        new_credit_by_firm=np.zeros(2),
        new_credit_by_household=np.zeros(0),
        max_supply_based_on_preferences=np.full(2, np.inf),
        new_preference_credit_by_bank=np.zeros(2),
        target_credit_override=np.array([100.0, 100.0]),
        append_firm_diagnostics=False,
        own_bank_term_out=firms.states["Corresponding Bank ID"],
    )


def test_overdraft_refinance_relief_applies_only_at_the_bank_holding_the_overdraft():
    # No free capital room at either bank (equity / CAR = 100 = its overdrafts). Firm 1 (bank 1) cannot
    # borrow (collateral 100 - overdraft 100 = 0), so its overdraft at bank 1 is not retired: bank 1 must not
    # lend against it. Firm 0's refinance is termed out by bank 0, which holds that overdraft, even though
    # bank 1 is the cheaper lender.
    clearer, loans = _term_out_case(equity=[10.0, 10.0], rates=[0.08, 0.02])
    np.testing.assert_allclose(clearer._last_term_out_by_bank, [100.0, 0.0])
    np.testing.assert_allclose(loans[0], [[100.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(loans[1], [[0.08, 0.0], [0.0, 0.0]])
    assert loans[2, 0, 0] > 0.0


def test_overdraft_refinance_uses_free_room_first_and_terms_out_the_rest_at_the_own_bank():
    # Bank 1 has 30 of free room (equity / CAR = 130): it lends 30 to firm 0; bank 0 terms out the other 70.
    clearer, loans = _term_out_case(equity=[10.0, 13.0], rates=[0.08, 0.02])
    np.testing.assert_allclose(loans[0], [[70.0, 0.0], [30.0, 0.0]])
    np.testing.assert_allclose(clearer._last_term_out_by_bank, [70.0, 0.0])


def test_overdraft_term_out_is_capped_by_the_overdrafts_the_bank_books():
    # Bank 0 books only 50 of overdrafts (inconsistent with firm 0's 100): it terms out at most 50, net of the
    # 30 bank 1 already lent to firm 0 from free room, so 20.
    clearer, loans = _term_out_case(equity=[5.0, 13.0], rates=[0.08, 0.02], booked=(50.0, 100.0))
    np.testing.assert_allclose(loans[0], [[20.0, 0.0], [30.0, 0.0]])
    np.testing.assert_allclose(clearer._last_term_out_by_bank, [20.0, 0.0])
