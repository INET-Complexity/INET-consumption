"""Generic household insolvency routed to FICP (consumer_insolvency_opens_ficp)."""

from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.agents.households.func.consumer_distress import (
    CURRENT,
    FICP,
    compute_stage6_consumer_distress_state,
)
from macromodel.agents.households.func.insolvency import DefaultHouseholdInsolvencyHandler
from macromodel.markets.credit_market.credit_market import CreditMarket


def _loan_state(n_banks: int, n_households: int) -> np.ndarray:
    return np.zeros((3, n_banks, n_households))


def _market(cons_principal: np.ndarray, mort_principal: np.ndarray) -> CreditMarket:
    n_banks, n_households = cons_principal.shape
    cons = _loan_state(n_banks, n_households)
    mort = _loan_state(n_banks, n_households)
    cons[0] = cons_principal
    mort[0] = mort_principal
    return CreditMarket.from_data(
        country_name="TST",
        st_loans=_loan_state(n_banks, 1),
        lt_loans=_loan_state(n_banks, 1),
        cons_loans=cons,
        mort_loans=mort,
    )


def _households(net_wealth, liquid, ficp_active):
    series = {
        "net_wealth": np.asarray(net_wealth, dtype=float),
        "liquid_financial_assets": np.asarray(liquid, dtype=float),
        "n_households": len(net_wealth),
    }
    return SimpleNamespace(
        ts=SimpleNamespace(current=lambda name: series[name]),
        current_ficp_active=lambda: np.asarray(ficp_active, dtype=bool),
    )


def _stage6(prior_exclusion, forced):
    n = len(prior_exclusion)
    return compute_stage6_consumer_distress_state(
        scheduled_consumer_payments=np.full(n, 10.0),
        actual_consumer_payments=np.full(n, 10.0),
        unpaid_consumer_payments=np.zeros(n),
        prior_missed_payment_count_consumer=np.zeros(n),
        prior_ficp_exclusion_remaining_periods=np.asarray(prior_exclusion),
        ficp_exclusion_periods=20,
        forced_ficp_trigger=forced,
    )


def test_forced_trigger_opens_full_ficp_episode_without_a_missed_payment():
    state = _stage6(prior_exclusion=[0, 5, 0], forced=np.asarray([True, True, False]))

    np.testing.assert_array_equal(state.ficp_episode_triggered, [True, False, False])
    np.testing.assert_array_equal(state.ficp_exclusion_remaining_periods, [20, 4, 0])
    np.testing.assert_array_equal(state.consumer_distress_state, [FICP, FICP, CURRENT])
    np.testing.assert_array_equal(state.ficp_episode_status, [1, 1, 0])


def test_no_forced_trigger_is_bit_identical_to_legacy():
    legacy = _stage6(prior_exclusion=[0, 3], forced=None)
    zeros = _stage6(prior_exclusion=[0, 3], forced=np.zeros(2, dtype=bool))

    for field in ("ficp_exclusion_remaining_periods", "ficp_episode_triggered", "consumer_distress_state"):
        np.testing.assert_array_equal(getattr(legacy, field), getattr(zeros, field))


def test_forced_trigger_shape_is_validated():
    with pytest.raises(ValueError, match="forced_ficp_trigger"):
        _stage6(prior_exclusion=[0, 0], forced=np.zeros(3, dtype=bool))


def test_ficp_routing_keeps_consumer_loans_books_mortgages_and_flags_new_entrants():
    # Households 0 and 1 insolvent; 1 is already under FICP; 2 solvent.
    market = _market(
        cons_principal=np.asarray([[100.0, 40.0, 30.0]]),
        mort_principal=np.asarray([[0.0, 200.0, 50.0]]),
    )
    households = _households(
        net_wealth=[-1.0, -1.0, 5.0],
        liquid=[-1.0, -1.0, -1.0],
        ficp_active=[False, True, False],
    )
    handler = DefaultHouseholdInsolvencyHandler(consumer_insolvency_opens_ficp=True)

    rate, _, _ = handler.handle_insolvency(households=households, banks=None, credit_market=market)

    assert rate == pytest.approx(2 / 3)
    np.testing.assert_allclose(market.states["cons_loans"][0], [[100.0, 40.0, 30.0]])
    np.testing.assert_allclose(market.states["mort_loans"][0], [[0.0, 0.0, 50.0]])
    np.testing.assert_allclose(market._mortgage_credit_loss_by_bank, [200.0])
    np.testing.assert_array_equal(handler.last_ficp_trigger, [True, False, False])


def test_legacy_handler_still_removes_consumer_loans_without_trigger():
    market = _market(
        cons_principal=np.asarray([[100.0, 30.0]]),
        mort_principal=np.asarray([[0.0, 0.0]]),
    )
    households = _households(net_wealth=[-1.0, 5.0], liquid=[-1.0, -1.0], ficp_active=[False, False])
    handler = DefaultHouseholdInsolvencyHandler()

    handler.handle_insolvency(households=households, banks=None, credit_market=market)

    np.testing.assert_allclose(market.states["cons_loans"][0], [[0.0, 30.0]])
    assert handler.last_ficp_trigger is None


@pytest.mark.parametrize("opens_ficp, expected", [(True, [True, False]), (False, [False, False])])
def test_mortgage_refused_under_ficp_ban_only_with_routing(opens_ficp, expected):
    households = SimpleNamespace(
        consumer_underwriting_income=np.asarray([100.0, 100.0]),
        consumer_underwriting_dsti_limit=0.35,
        consumer_underwriting_other_service=np.zeros(2),
        consumer_insolvency_opens_ficp=opens_ficp,
        current_ficp_active=lambda: np.asarray([True, False]),
    )
    mortgage = _loan_state(1, 2)
    mortgage[0] = [[50.0, 50.0]]
    mortgage[2] = [[1.0, 1.0]]

    rejected = CreditMarket._mortgage_dsti_rejected(
        households, mortgage, future_cons_service=np.zeros(2), old_mortgage_service=np.zeros(2)
    )

    np.testing.assert_array_equal(rejected, expected)


class TestHouseholdsWiring:
    def test_pending_trigger_gates_credit_and_opens_episode_next_period(self, test_households, monkeypatch):
        n = int(test_households.ts.current("n_households"))
        monkeypatch.setitem(
            test_households.functions,
            "insolvency",
            DefaultHouseholdInsolvencyHandler(consumer_insolvency_opens_ficp=True),
        )
        trigger = np.zeros(n, dtype=bool)
        trigger[0] = True
        test_households.ts.ficp_generic_insolvency_trigger.append(trigger)

        # Gate is active before Stage 6 opens the episode.
        np.testing.assert_array_equal(test_households.current_ficp_active(), trigger)

        test_households.record_stage6_consumer_distress_state(
            scheduled_consumer_payments=np.full(n, 10.0),
            actual_consumer_payments=np.full(n, 10.0),
            unpaid_consumer_payments=np.zeros(n),
            time_unit=3,
            consumer_contractual_principal=np.full(n, 25.0),
            consumer_principal_arrears=np.zeros(n),
            consumer_interest_arrears=np.zeros(n),
        )

        remaining = test_households.ts.current("ficp_exclusion_remaining_periods")
        assert remaining[0] == 20
        assert np.all(remaining[1:] == 0)
        assert test_households.ts.current("ficp_episode_id")[0] == 1
        assert test_households.ts.current("ficp_open_balance")[0] == pytest.approx(25.0)

    def test_trigger_ignored_without_routing(self, test_households):
        n = int(test_households.ts.current("n_households"))
        trigger = np.ones(n, dtype=bool)
        test_households.ts.ficp_generic_insolvency_trigger.append(trigger)

        assert not test_households.consumer_insolvency_opens_ficp
        assert not np.any(test_households.current_ficp_active())
