"""Numerical income guard G replacing the subsistence series S in every behavioural-income floor (PLAN.md 3.4, 5.5).

The five floor sites of PLAN.md section 4.5: (1) ``_record_behavioural_income`` -> ``reconcile_behavioural_income``
and the initial ``income_bridge_current``; (2) the monetary PI / uncertainty moments (``income_floor``); (3) the
consumption-rule fallback (``subsistence_income``); (4) the ``IncomeHistories`` floor history used by
``prepare_inputs``; (5) the borrowing-rate prior income, which reads the same history.
"""

from copy import deepcopy

import numpy as np
import pytest

import macromodel.country.country as country_module
from macromodel.agents.households.borrowing_rates import lagged_annual_income_nominal
from macromodel.agents.households.consumption_inputs import (
    IncomeHistories,
    MonetaryUnits,
    OpeningSnapshot,
    OpeningStockValues,
    PermanentIncomePayload,
    prepare_inputs,
)
from macromodel.util.behavioural_income import reconcile_behavioural_income
from tests.test_macromodel.unit.test_agents.test_households.func.test_consumption_calibration_v2 import _rule
from tests.test_macromodel.unit.test_country.test_minimum_income_entitlement import (
    RSA,
    _production_simulation,
    _use_minimum_income,
)

PERIODS = 2
SHARE = 0.01


def _spy_floor_sites(country, monkeypatch):
    """Record the floor passed at sites 1-4 during planning (site 5 reads the site-4 history)."""
    calls = {"reconcile": [], "moments": [], "fallback": [], "history": []}
    hh = country.households
    reconcile = country_module.reconcile_behavioural_income

    def spy_reconcile(**kwargs):
        calls["reconcile"].append(np.array(kwargs["subsistence"]))
        return reconcile(**kwargs)

    learning = hh.current_income_belief_learning_inputs

    def spy_learning(**kwargs):
        components = kwargs.get("monetary_income_components")
        if components is not None:
            calls["moments"].append(np.array(components["income_floor"]))
        return learning(**kwargs)

    consumption = hh.functions["consumption"].compute_target_consumption

    def spy_consumption(**kwargs):
        calls["fallback"].append(np.array(kwargs["subsistence_income"]))
        return consumption(**kwargs)

    adapter = country._consumption_input_adapter
    prepare = adapter.prepare

    def spy_prepare(histories, *args, **kwargs):
        calls["history"].append(np.array(histories.guard_nominal_history))
        return prepare(histories, *args, **kwargs)

    monkeypatch.setattr(country_module, "reconcile_behavioural_income", spy_reconcile)
    monkeypatch.setattr(hh, "current_income_belief_learning_inputs", spy_learning)
    monkeypatch.setattr(hh.functions["consumption"], "compute_target_consumption", spy_consumption)
    monkeypatch.setattr(adapter, "prepare", spy_prepare)
    return calls


def _assert_every_site_used(calls, floor, floor_history):
    # Two planning passes: reconcile runs per pass plus once on the realised record.
    assert len(calls["reconcile"]) == 3 and len(calls["moments"]) == 2 and len(calls["fallback"]) == 2
    assert len(calls["history"]) == 2
    for name in ("reconcile", "moments", "fallback"):
        for value in calls[name]:
            np.testing.assert_array_equal(value, floor, err_msg=name)
    for value in calls["history"]:
        np.testing.assert_array_equal(value, floor_history)


def test_legacy_mode_floors_every_site_at_the_subsistence_series(datawrapper, monkeypatch):
    simulation = _production_simulation(datawrapper, {})
    country = simulation.countries["FRA"]
    hh, economy = country.households, country.economy
    s0 = economy.ts.current("subsistence_consumption")
    raw0 = hh.ts.current("expected_behavioural_income_raw")
    np.testing.assert_array_equal(hh.ts.current("expected_behavioural_income"), np.maximum(raw0, s0))
    n0 = hh.ts.current("expected_behavioural_income_non_property")
    np.testing.assert_array_equal(hh.ts.current("income_bridge_current"), np.maximum(n0, s0))
    np.testing.assert_array_equal(hh.ts.current("income_bridge_floor_used"), (n0 < s0).astype(float))

    for period in range(PERIODS):
        calls = _spy_floor_sites(country, monkeypatch)
        simulation.iterate(t=period)
        monkeypatch.undo()
        s = economy.ts.current("subsistence_consumption")
        _assert_every_site_used(calls, s, np.asarray(economy.ts.historic("subsistence_consumption")))
        raw = hh.ts.current("behavioural_income_raw")
        np.testing.assert_array_equal(hh.ts.current("behavioural_income"), np.maximum(raw, s))
        # The guard series carries S, and its flag is the S-binding flag of planning income.
        np.testing.assert_array_equal(
            economy.ts.historic("income_guard"), economy.ts.historic("subsistence_consumption")
        )
        np.testing.assert_array_equal(
            hh.ts.current("income_guard_bound"), hh.ts.current("expected_behavioural_income_floor_used")
        )


def _opening_guard(country):
    """G at t0: share x RSA x scale x months x CU, with I = 1 and CPI ratio 1."""
    cu = country.households._current_consumption_units()
    return SHARE * RSA * country.scale * country.economy.time_unit * cu


@pytest.mark.parametrize("goods_floor", [True, False])
def test_minimum_income_mode_floors_every_site_at_the_guard(datawrapper, monkeypatch, goods_floor):
    simulation = _production_simulation(datawrapper, {"mode": "minimum_income", "goods_floor": goods_floor})
    country = simulation.countries["FRA"]
    hh, economy = country.households, country.economy

    # t0: the initial bridge and income records are guarded by G from the opening M.
    g0 = _opening_guard(country)
    np.testing.assert_allclose(economy.ts.current("income_guard"), g0, rtol=1e-12)
    g0 = economy.ts.current("income_guard")
    assert np.all(g0 < economy.ts.current("subsistence_consumption"))
    np.testing.assert_array_equal(hh.ts.current("minimum_income_level"), 0.0)  # no assessment at t0
    raw0 = hh.ts.current("expected_behavioural_income_raw")
    np.testing.assert_array_equal(hh.ts.current("expected_behavioural_income"), np.maximum(raw0, g0))
    np.testing.assert_array_equal(hh.ts.current("income_guard_bound"), (raw0 < g0).astype(float))
    n0 = hh.ts.current("expected_behavioural_income_non_property")
    np.testing.assert_array_equal(hh.ts.current("income_bridge_current"), np.maximum(n0, g0))
    np.testing.assert_array_equal(hh.ts.current("income_bridge_floor_used"), (n0 < g0).astype(float))

    for period in range(PERIODS):
        calls = _spy_floor_sites(country, monkeypatch)
        simulation.iterate(t=period)
        monkeypatch.undo()
        g = economy.ts.current("income_guard")
        np.testing.assert_allclose(g, SHARE * hh.ts.current("minimum_income_level"), rtol=1e-15)
        history = np.asarray(economy.ts.historic("income_guard"))
        assert history.shape[0] == period + 2
        np.testing.assert_array_equal(history[0], g0)  # no pre-activation rows: the mode is fixed from t0
        _assert_every_site_used(calls, g, history)
        raw = hh.ts.current("expected_behavioural_income_raw")
        np.testing.assert_array_equal(hh.ts.current("income_guard_bound"), (raw < g).astype(float))
        assert len(hh.ts.historic("income_guard_bound")) == period + 2
        raw = hh.ts.current("behavioural_income_raw")
        np.testing.assert_array_equal(hh.ts.current("behavioural_income"), np.maximum(raw, g))
        # The goods floor still reads S (switched by goods_floor), not G.
        s = economy.ts.current("subsistence_consumption")
        np.testing.assert_array_equal(country._effective_goods_floor(), s if goods_floor else 0.0)


def test_reset_into_minimum_income_keeps_pre_activation_rows(datawrapper):
    simulation = _production_simulation(datawrapper, {})
    country = simulation.countries["FRA"]
    hh, economy = country.households, country.economy
    s0 = economy.ts.current("subsistence_consumption").copy()
    income0 = hh.ts.current("expected_behavioural_income").copy()

    configuration = deepcopy(simulation.configuration)
    configuration.country_configurations["FRA"].household_support = {"mode": "minimum_income"}
    simulation.reset(configuration)
    simulation.iterate(t=0)

    history = economy.ts.historic("income_guard")
    np.testing.assert_array_equal(history[0], s0)  # the t0 row predates activation and keeps S
    np.testing.assert_array_equal(hh.ts.historic("expected_behavioural_income")[0], income0)
    np.testing.assert_allclose(history[1], SHARE * hh.ts.current("minimum_income_level"), rtol=1e-15)


def test_guard_requires_the_period_assessment_in_minimum_income_mode(test_country):
    _use_minimum_income(test_country)
    with pytest.raises(RuntimeError, match="assess it first"):
        test_country._update_income_guard(replace_current=False)


def test_legacy_guard_is_a_copy_of_the_subsistence_row(test_country):
    test_country._update_income_guard(replace_current=False)
    guard = test_country.economy.ts.current("income_guard")
    s = test_country.economy.ts.current("subsistence_consumption")
    np.testing.assert_array_equal(guard, s)
    assert not np.shares_memory(guard, s)


def test_guard_binds_strictly_below_and_floors_zero_and_negative_income():
    guard = np.full(4, 4.0)
    parts = reconcile_behavioural_income(
        non_property=np.array([0.0, -5.0, 4.0, 9.0]),
        rental=np.zeros(4),
        financial=np.zeros(4),
        deposit_receipts=np.zeros(4),
        borrower_interest=np.zeros(4),
        imputed_rent=np.zeros(4),
        subsistence=guard,
    )
    np.testing.assert_array_equal(parts["raw"], [0.0, 0.0, 4.0, 9.0])  # raw value kept
    np.testing.assert_array_equal(parts["floor_used"], [1.0, 1.0, 0.0, 0.0])  # raw == G does not bind
    np.testing.assert_array_equal(parts["income"], [4.0, 4.0, 4.0, 9.0])
    # Negative raw income (borrower interest above the blend) is floored at G and flagged.
    negative = reconcile_behavioural_income(
        non_property=np.array([1.0]),
        rental=np.zeros(1),
        financial=np.zeros(1),
        deposit_receipts=np.zeros(1),
        borrower_interest=np.array([3.0]),
        imputed_rent=np.zeros(1),
        subsistence=np.array([0.5]),
    )
    assert negative["raw"][0] == -2.0 and negative["income"][0] == 0.5 and negative["floor_used"][0] == 1.0


def _prepared(behavioural, guard, prices):
    units = MonetaryUnits("EUR", 1)
    tags = dict(period_id=2, pass_kind="first", household_layout_key="guard-test")
    h = behavioural.shape[1]
    history = IncomeHistories(
        behavioural_income_nominal_history=behavioural,
        guard_nominal_history=guard,
        consumer_price_history=prices,
        current_consumer_price=prices[-1],
        initial_consumer_price=1.0,
        price_history_layout="aligned",
        income_units=units,
        guard_units=units,
    )
    snapshot = OpeningSnapshot(2, "guard-test", units, OpeningStockValues(*[np.zeros(h)] * 7))
    pi = PermanentIncomePayload(**tags, units=units, monetary_income_nominal=np.ones(h), pi_log_ratio_original=[0] * h)
    return prepare_inputs(history, snapshot, pi, **tags)


def test_history_floors_zero_negative_and_no_positive_window_income_at_the_guard():
    prices = np.array([1.0, 1.1, 1.2, 1.3])
    # Household 0: no positive income in the window; 1: negative then positive; 2: always above the guard.
    behavioural = np.array([[0.0, -10.0, 50.0], [0.0, 20.0, 60.0], [-3.0, -10.0, 70.0], [0.0, 30.0, 80.0]])
    guard = np.array([[2.0, 3.0, 4.0], [2.2, 3.3, 4.4], [2.4, 3.6, 4.8], [2.6, 3.9, 5.2]])
    prepared = _prepared(behavioural, guard, prices)
    floored = np.maximum(behavioural, guard)
    expected = np.exp(np.mean(np.log(floored) - np.log(prices)[:, None], axis=0))
    np.testing.assert_allclose(prepared.income_base_real, expected, rtol=1e-12)
    np.testing.assert_allclose(prepared.income_base_real[0], np.exp(np.mean(np.log(guard[:, 0] / prices))), rtol=1e-12)
    np.testing.assert_array_equal(prepared.floor_hit_mask, behavioural < guard)
    # Site 5: the borrowing-rate prior income excludes the current row and uses the same guarded observations.
    prior = np.exp(np.mean(np.log(floored[:3]) - np.log(prices[:3])[:, None], axis=0))
    np.testing.assert_allclose(lagged_annual_income_nominal(prepared), 4.0 * prior * prices[2], rtol=1e-12)


def test_consumption_fallback_without_positive_window_returns_the_guard():
    rule = _rule(income_denominator="geometric_average", income_denominator_window=4)
    history = np.array([[0.0, 5.0], [-1.0, 5.0], [0.0, 5.0]])
    guard = np.array([0.7, 0.7])
    got = rule._geometric_average_income(history, 1.0, np.ones(3), subsistence_income=guard)
    np.testing.assert_allclose(got, [0.7, 5.0], rtol=1e-12)
