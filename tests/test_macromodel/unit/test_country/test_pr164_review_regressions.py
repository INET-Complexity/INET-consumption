"""Cross-function counterexamples from the PR #164 accounting review."""

from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.agents.households.households import Households
from macromodel.markets.credit_market.credit_market import CreditMarket
from macromodel.timeseries import TimeSeries


def test_country_planning_excludes_mortgage_interest_from_consumer_balance(test_country, monkeypatch):
    country = test_country
    n = country.households.ts.current("n_households")
    loans = np.zeros((3, country.banks.ts.current("n_banks"), n))
    mortgage = loans.copy()
    mortgage[:, 0, :] = np.array([90.0, 0.1, 10.0])[:, None]
    country.credit_market = CreditMarket.from_data("TST", loans.copy(), loans.copy(), loans.copy(), mortgage)
    monkeypatch.setattr(country, "household_gross_cash_income", lambda: np.full(n, 100.0))
    country.banks.parameters.household_consumption_loan_maturity = 12
    country.banks.ts.override_current("interest_rates_on_household_consumption_loans", np.zeros(1))
    captured = {}
    original = country.households.compute_and_record_residual_capacity_fallback

    def capture(**kwargs):
        captured.update(kwargs)
        # Exercise the real capacity helper with an explicit borrowing need.
        kwargs["preferred_margin_after_lfa"] = np.ones(n)
        kwargs["preferred_margin_amount"] = np.full(n, 295.0)
        kwargs["other_debt_service"] = np.zeros(n)
        return original(**kwargs)

    monkeypatch.setattr(country.households, "compute_and_record_residual_capacity_fallback", capture)
    country._set_household_target_demand(replace_current=False)
    np.testing.assert_allclose(country.credit_market.preview_opening_household_interest(), 9.0)
    np.testing.assert_array_equal(captured["existing_consumer_balance"], np.zeros(n))
    np.testing.assert_allclose(country.households.ts.current("dsti_maximum_loan_size"), 300.0)
    np.testing.assert_allclose(country.households.ts.current("borrow_planned"), 295.0)
    offered = loans.copy()
    offered[:, 0, :] = np.array([295.0, 0.0, 295.0 / 12])[:, None]
    accepted, _ = country.credit_market.enforce_household_origination_dsti(country.households, 12, offered, loans)
    np.testing.assert_allclose(accepted[0].sum(axis=0), 295.0)


@pytest.mark.parametrize("other_rate,expected_other_margin", [(0.01, 1), (0.2, 2)])
def test_unrelated_bank_rate_does_not_change_low_rate_household_preference(other_rate, expected_other_margin):
    hh = Households.__new__(Households)
    hh.states = {"Corresponding Bank ID": np.array([0, 1, 0])}
    fields = (
        "preferred_margin_after_lfa",
        "preferred_margin_amount",
        "borrow_vs_sell_threshold",
        "borrow_vs_sell_spread",
        "borrow_vs_sell_l_tilde",
        "borrow_vs_sell_comparison_valid_flag",
    )
    hh.ts = TimeSeries(n_households=3, **{name: np.zeros(3) for name in fields})
    hh.functions = {"wealth": SimpleNamespace(phi_1=1.0, lambda_kappa=1.0)}
    banks = SimpleNamespace(
        ts=TimeSeries(n_banks=2, interest_rates_on_household_consumption_loans=np.array([0.01, other_rate]))
    )
    margins, _ = hh.compute_and_record_borrow_vs_sell_choice(
        residual_shortfall_after_lfa=np.ones(3),
        banks=banks,
        borrow_vs_sell_inputs={
            "delta_tilde": np.zeros(3),
            "opening_tfa_scale": np.full(3, 100.0),
            "post_return_ifa": np.full(3, 100.0),
            "r_kappa": np.full(3, 0.05),
        },
    )
    np.testing.assert_array_equal(margins, [1, expected_other_margin, 1])
    np.testing.assert_allclose(hh.ts.current("borrow_vs_sell_spread")[[0, 2]], -0.045)


def test_consumer_interest_reserve_preserves_opening_schedule_and_arrears_convention():
    loans = np.zeros((3, 1, 1))
    consumer = loans.copy()
    consumer[:, 0, 0] = [60.0, 0.02, 6.0]
    mortgage = loans.copy()
    mortgage[:, 0, 0] = [90.0, 0.1, 10.0]
    market = CreditMarket.from_data("TST", loans.copy(), loans.copy(), consumer, mortgage)
    market._consumer_principal_arrears_by_cell[:] = 10.0
    new = loans.copy()
    new[:, 0, 0] = [20.0, 0.03, 2.0]
    market._add_new_loans("cons_loans", new)
    before = market.states["cons_loans"].copy()
    # Only opening performing consumer principal accrues the reserve. New
    # grants and principal arrears are excluded, as in actual underwriting.
    np.testing.assert_allclose(market.preview_opening_consumer_interest(), 1.0)
    np.testing.assert_allclose(market.preview_opening_household_interest(), 10.0)
    np.testing.assert_array_equal(market.states["cons_loans"], before)
