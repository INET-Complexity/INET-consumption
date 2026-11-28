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


def test_ficp_trigger_requires_consumer_loans():
    """All four insolvent; only household 0 has consumer loans and enters FICP.

    Household 1 is insolvent through its mortgage, 2 through its overdraft only,
    and 3 owes only consumer interest arrears (the path of unpaid overdraft interest).
    """
    market = _market(
        cons_principal=np.asarray([[50.0, 0.0, 0.0, 0.0]]),
        mort_principal=np.asarray([[0.0, 200.0, 0.0, 0.0]]),
    )
    market._consumer_interest_arrears_by_cell[0, 3] = 5.0
    households = _households(net_wealth=[-1.0] * 4, liquid=[-1.0] * 4, ficp_active=[False] * 4)
    handler = DefaultHouseholdInsolvencyHandler(consumer_insolvency_opens_ficp=True)

    handler.handle_insolvency(households=households, banks=None, credit_market=market)

    np.testing.assert_array_equal(handler.last_ficp_trigger, [True, False, False, False])
    # The mortgage-only household still has its mortgage removed and booked.
    np.testing.assert_allclose(market.states["mort_loans"][0], [[0.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(market._mortgage_credit_loss_by_bank, [200.0])
    np.testing.assert_allclose(market._consumer_interest_arrears_by_cell, [[0.0, 0.0, 0.0, 5.0]])


def test_planned_uses_never_open_a_new_overdraft_for_a_banned_household():
    """A banned household gets no consumer credit (K = 0); the Stage 5 allocation spends only
    income + max(LFA, 0), so planned closing deposits stay >= min(opening LFA, 0).

    Households: overdrawn (-50), small positive balance (20), zero balance. Desired uses far
    exceed income, so the allocation is binding for all three.
    """
    from macromodel.agents.households.func.financial_feasibility import (
        HouseholdFinancialFeasibility,
        PostGrantFeasiblePlan,
    )

    lfa = np.asarray([-50.0, 20.0, 0.0])
    income = np.full(3, 100.0)
    rent = np.full(3, 30.0)
    floor = np.full(3, 40.0)
    service = np.full(3, 20.0)
    overdraft_interest = np.asarray([2.0, 0.0, 0.0])
    zeros = np.zeros(3)
    plan = PostGrantFeasiblePlan(
        credit_granted=zeros,
        credit_rationing_gap=zeros,
        planned_liquidation_total=zeros,
        residual_shortfall_after_granted_credit=zeros,
    )
    for support_enabled in (True, False):
        settled = HouseholdFinancialFeasibility().settle_consumption_floor(
            plan,
            consumption_before_floor=np.full(3, 300.0),
            subsistence_floor=floor,
            cash_rent=rent,
            resources=income + np.maximum(lfa, 0.0),
            scheduled_consumer_service=service,
            scheduled_mortgage_service=zeros,
            investment=np.full(3, 50.0),
            support_enabled=support_enabled,
            scheduled_overdraft_interest=overdraft_interest,
            opening_overdraft=np.maximum(-lfa, 0.0),
        )
        uses = (
            settled.consumption_after_floor
            + (rent - settled.unpaid_rent)
            + (service - settled.unpaid_consumer_service)
            + (overdraft_interest - settled.unpaid_overdraft_interest)
            + settled.investment_after_floor
        )
        closing = lfa + income + settled.floor_support - uses

        assert np.all(closing >= np.minimum(lfa, 0.0) - 1e-9), (support_enabled, closing)


def test_mortgage_refused_under_ficp_ban_through_purchase_completion():
    """Household 1 buys vacant property 10 for 150 with a mortgage of 100 that passes DSTI.

    Banned, its mortgage is zeroed and, with 50 of its own cash, the purchase fails.
    """
    import pandas as pd

    from macromodel.markets.housing_market.purchase_completion import PurchaseCompletionInputs

    n = 3

    def run(banned: bool):
        market = CreditMarket.from_data(
            "TST",
            np.zeros((3, 1, 1)),
            np.zeros((3, 1, 1)),
            np.zeros((3, 1, n)),
            np.zeros((3, 1, n)),
        )
        households = SimpleNamespace(
            consumer_underwriting_income=np.full(n, 100.0),
            consumer_underwriting_other_service=np.zeros(n),
            consumer_underwriting_dsti_limit=0.35,
            uses_feasibility_resolver=True,
            consumer_insolvency_opens_ficp=True,
            current_ficp_active=lambda: np.asarray([False, banned, False]),
        )
        mortgage = np.zeros((3, 1, n))
        mortgage[:, 0, 1] = [100.0, 0.01, 10.0]
        inputs = PurchaseCompletionInputs(
            current_sales=pd.DataFrame(
                [("Sell", 1, 2, 10, 150.0)],
                columns=["sales_types", "buyer_id", "seller_id", "property_id", "price_or_rent"],
            ),
            property_owner=pd.Series([2], index=[10]),
            property_value=pd.Series([150.0], index=[10]),
            property_inhabitant=pd.Series([-1], index=[10]),
            cash_need=np.zeros(n),
            liquid_assets=np.asarray([0.0, 50.0, 0.0]),
            illiquid_assets=np.zeros(n),
            planned_liquidation=np.zeros(n),
            sale_cost=np.zeros(n),
        )
        _, mort = market._complete_property_purchases(households, 12, np.zeros((3, 1, n)), mortgage, inputs)
        return mort, market.current_purchase_completion()

    mort, completion = run(banned=False)
    assert mort[0, 0, 1] == 100.0
    np.testing.assert_array_equal(completion.kept_sales, [True])

    mort, completion = run(banned=True)
    assert mort[0].sum() == 0.0
    np.testing.assert_array_equal(completion.mortgage_zeroed, [False, True, False])
    np.testing.assert_array_equal(completion.kept_sales, [False])


class TestTriggerLifecycle:
    """Households wiring across F2 -> Stage 6 -> F2, with the real handler and credit market."""

    @staticmethod
    def _routing(households, monkeypatch):
        monkeypatch.setitem(
            households.functions,
            "insolvency",
            DefaultHouseholdInsolvencyHandler(consumer_insolvency_opens_ficp=True),
        )

    @staticmethod
    def _consumer_market(principal: np.ndarray) -> CreditMarket:
        n = principal.shape[0]
        market = _market(cons_principal=principal[None, :], mort_principal=np.zeros((1, n)))
        active = principal > 0.0
        market.states["cons_loans"][1][0, active] = 0.05
        market.states["cons_loans"][2][0, active] = 10.0
        return market

    @staticmethod
    def _set_insolvent(households, insolvent: np.ndarray) -> None:
        households.ts.net_wealth.append(np.where(insolvent, -1.0, 1.0))
        households.ts.liquid_financial_assets.append(np.where(insolvent, -1.0, 1.0))

    @staticmethod
    def _stage6(households, market, period: int):
        n = int(households.ts.current("n_households"))
        principal, principal_arrears, interest_arrears = market.current_consumer_debt_components_by_household()
        return households.record_stage6_consumer_distress_state(
            scheduled_consumer_payments=np.zeros(n),
            actual_consumer_payments=np.zeros(n),
            unpaid_consumer_payments=np.zeros(n),
            time_unit=3,
            period=period,
            consumer_contractual_principal=principal,
            consumer_principal_arrears=principal_arrears,
            consumer_interest_arrears=interest_arrears,
        )

    def test_masked_trigger_is_consumed_and_household_retested_next_f2(self, test_households, monkeypatch):
        hh = test_households
        self._routing(hh, monkeypatch)
        n = int(hh.ts.current("n_households"))
        principal = np.zeros(n)
        principal[:2] = [100.0, 80.0]
        market = self._consumer_market(principal)
        insolvent = np.zeros(n, dtype=bool)
        insolvent[:3] = True  # household 2 is overdraft-only

        # F2, period t: Households.handle_insolvency appends the trigger.
        self._set_insolvent(hh, insolvent)
        hh.handle_insolvency(banks=None, credit_market=market)
        expected = np.zeros(n, dtype=bool)
        expected[:2] = True
        np.testing.assert_array_equal(hh.ts.current("ficp_generic_insolvency_trigger"), expected)

        # Stage 6, period t+1: household 0 still has an unfinished forgiveness event, so its trigger is masked.
        emitted = np.zeros(n, dtype=bool)
        emitted[0] = True
        hh.ts.override_current("ficp_forgiveness_event_emitted", emitted)
        hh.ts.override_current("ficp_forgiveness_event_stage", np.where(emitted, 2.0, 0.0))
        self._stage6(hh, market, period=1)

        remaining = np.asarray(hh.ts.current("ficp_exclusion_remaining_periods"), dtype=float)
        assert remaining[0] == 0 and remaining[1] == 20 and np.all(remaining[2:] == 0)
        np.testing.assert_array_equal(hh.current_ficp_active(), remaining > 0)
        assert not np.any(hh.ts.current("ficp_generic_insolvency_trigger"))
        np.testing.assert_array_equal(hh.ts.current("ficp_generic_insolvency_trigger_consumed"), expected)

        # The crash path: the stale trigger made household 0 active with zero remaining periods.
        with pytest.raises(ValueError, match="positive integers"):
            market.remodulate_ficp_consumer_loan_schedule(
                active_ficp=(remaining > 0) | expected, remaining_periods=remaining
            )
        market.remodulate_ficp_consumer_loan_schedule(active_ficp=hh.current_ficp_active(), remaining_periods=remaining)

        # F2, period t+1: household 0 is re-triggered (not dropped); household 1 is already in FICP.
        self._set_insolvent(hh, insolvent)
        hh.handle_insolvency(banks=None, credit_market=market)
        retriggered = np.zeros(n, dtype=bool)
        retriggered[0] = True
        np.testing.assert_array_equal(hh.ts.current("ficp_generic_insolvency_trigger"), retriggered)

        # Stage 6, period t+2: the event is finished, so household 0 opens its episode.
        hh.ts.override_current("ficp_forgiveness_event_stage", np.where(emitted, 3.0, 0.0))
        self._stage6(hh, market, period=2)
        remaining = np.asarray(hh.ts.current("ficp_exclusion_remaining_periods"), dtype=float)
        assert remaining[0] == 20 and remaining[1] == 19
        np.testing.assert_array_equal(hh.current_ficp_active(), remaining > 0)

    def test_no_immediate_retrigger_when_the_episode_ends(self, test_households, monkeypatch):
        """At the horizon end the residual is forgiven in F2 before the handler runs: no consumer
        debt is left, so the still-insolvent household does not re-enter FICP."""
        hh = test_households
        self._routing(hh, monkeypatch)
        n = int(hh.ts.current("n_households"))
        principal = np.zeros(n)
        principal[:2] = [60.0, 40.0]
        market = self._consumer_market(principal)
        prior = np.zeros(n)
        prior[0] = 1.0
        hh.ts.override_current("ficp_exclusion_remaining_periods", prior)
        hh.ts.override_current("ficp_episode_status", np.where(prior > 0, 1.0, 0.0))

        events = self._stage6(hh, market, period=5)

        assert [event.household_id for event in events] == [0]
        assert hh.ts.current("ficp_exclusion_remaining_periods")[0] == 0
        # F2 forgiveness (Country._process_ficp_forgiveness_events) removes household 0's consumer cells.
        selection = np.zeros(n, dtype=bool)
        selection[0] = True
        market.remove_consumer_loans_by_cell(market.snapshot_consumer_default_writeoff(selection).removal_mask)
        insolvent = np.zeros(n, dtype=bool)
        insolvent[:2] = True
        self._set_insolvent(hh, insolvent)
        hh.handle_insolvency(banks=None, credit_market=market)

        expected = np.zeros(n, dtype=bool)
        expected[1] = True
        np.testing.assert_array_equal(hh.ts.current("ficp_generic_insolvency_trigger"), expected)

    def test_routing_requires_the_feasibility_resolver(self, test_households, monkeypatch):
        self._routing(test_households, monkeypatch)
        test_households.configure_feasibility_resolver(True)

        with pytest.raises(ValueError, match="uses_feasibility_resolver"):
            test_households.configure_feasibility_resolver(False)
