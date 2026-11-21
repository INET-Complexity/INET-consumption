"""Allocator without Stage 5 support and the goods-floor switch (PLAN.md section 5.3 steps 3-4, section 5.4).

Invariants (PLAN.md section 6): 2 (Stage 5 = 0 every period in minimum_income mode), 3 (T independent of
period-t desired goods, spending and credit), 7 (floor off: paid goods <= desired goods; floor on: goods above
desired only through the funded floor, unmet floor visible).
"""

import numpy as np
import pytest

from tests.test_macromodel.unit.test_country.test_minimum_income_entitlement import _production_simulation

PERIODS = 3


def _spy_floor_settlement(households):
    """Record the keyword arguments of every floor settlement, then settle as usual."""
    calls = []
    settle = households.apply_consumption_floor_to_post_grant_plan

    def spy(**kwargs):
        calls.append(
            {name: np.array(value) if isinstance(value, np.ndarray) else value for name, value in kwargs.items()}
        )
        return settle(**kwargs)

    households.apply_consumption_floor_to_post_grant_plan = spy
    return calls


def test_legacy_mode_settles_with_support_and_the_full_goods_floor(datawrapper):
    simulation = _production_simulation(datawrapper, {})
    country = simulation.countries["FRA"]
    calls = _spy_floor_settlement(country.households)

    simulation.iterate(t=0)

    assert len(calls) == 1
    assert calls[0]["support_enabled"] is True
    np.testing.assert_array_equal(calls[0]["subsistence_floor"], country.economy.ts.current("subsistence_consumption"))


@pytest.mark.parametrize("goods_floor", [True, False])
def test_minimum_income_mode_pays_no_stage5_support_and_applies_the_goods_floor_switch(datawrapper, goods_floor):
    simulation = _production_simulation(datawrapper, {"mode": "minimum_income", "goods_floor": goods_floor})
    country = simulation.countries["FRA"]
    hh, government = country.households, country.central_government
    assert country.configuration.households.parameters.uses_feasibility_resolver
    calls = _spy_floor_settlement(hh)
    for period in range(PERIODS):
        simulation.iterate(t=period)
        floor = country.economy.ts.current("subsistence_consumption")
        plan = hh.post_grant_feasible_plan

        # Steps 3-4: the allocator runs once, without support, on F x goods_floor.
        assert len(calls) == period + 1
        assert calls[-1]["support_enabled"] is False
        np.testing.assert_array_equal(calls[-1]["subsistence_floor"], floor if goods_floor else 0.0)

        # Invariant 2: Stage 5 is zero every period, at every writer and in the deficit.
        for name in ("stage5_subsistence_support", "post_relief_funding_gap"):
            np.testing.assert_array_equal(hh.ts.current(name), 0.0, err_msg=name)
        np.testing.assert_array_equal(plan.floor_support, 0.0)
        np.testing.assert_array_equal(country.current_stage5_subsistence_support_by_household(), 0.0)
        assert government.ts.current("total_necessity_support")[0] == 0.0
        # The shortfall handed on is unpaid debt service only.
        np.testing.assert_array_equal(
            hh.ts.current("remaining_subsistence_shortfall"),
            plan.unpaid_consumer_service + plan.unpaid_mortgage_service,
        )
        # The no-op reconciliation keeps the financing diagnostic, one row per period.
        assert len(hh.ts.historic("financing_income_error")) == len(hh.ts.historic("income"))

        # New plan fields are persisted.
        np.testing.assert_array_equal(hh.ts.current("unpaid_rent"), plan.unpaid_rent)
        np.testing.assert_array_equal(hh.ts.current("unmet_floor"), plan.unmet_floor)
        np.testing.assert_array_equal(hh.ts.current("resource_deficit"), plan.resource_deficit)
        np.testing.assert_array_equal(
            plan.resource_deficit, np.maximum(-np.asarray(calls[-1]["resources"], dtype=float), 0.0)
        )

        # Invariant 7.
        desired = plan.consumption_before_floor
        paid = plan.consumption_after_floor
        if goods_floor:
            floor_paid = floor - plan.unmet_floor
            assert np.all(plan.unmet_floor >= 0.0) and np.all(floor_paid >= 0.0)
            # Goods above desired only through the funded floor: paid <= max(desired, Fp), and a household
            # paid above its desired goods buys exactly the funded floor, not the floor on top of desired.
            assert np.all(paid <= np.maximum(desired, floor_paid) * (1.0 + 1e-12))
            # F + (G - F) can exceed G by one ulp: a top-up is paid above desired beyond round-off.
            topped_up = paid > desired * (1.0 + 1e-12)
            assert topped_up.any()
            np.testing.assert_allclose(paid[topped_up], floor_paid[topped_up], rtol=1e-12)
            # An unmet floor leaves nothing for goods above it.
            unmet = plan.unmet_floor > 0.0
            np.testing.assert_allclose(paid[unmet], floor_paid[unmet], rtol=1e-12)
        else:
            assert np.all(paid <= desired)
            np.testing.assert_array_equal(plan.unmet_floor, 0.0)
            ordered = hh.ts.current("target_consumption").sum(axis=1) * hh.consumption_cash_factor()
            assert np.all(ordered <= desired * (1.0 + 1e-12))

        # The closing cash ledger still reconciles (unpaid rent is not yet settled differently: increment 5).
        assert np.max(np.abs(hh.ts.current("stage5_cash_ledger_residual"))) <= 1e-7
    assert np.all(np.isfinite(hh.ts.current("liquid_financial_assets")))


def _scale_desired_goods(country, factor):
    """Scale planned goods after both demand passes: a period-t outcome the entitlement must not see."""
    set_demand = country._set_household_target_demand

    def scaled(*args, **kwargs):
        result = set_demand(*args, **kwargs)
        country.households.ts.override_current(
            "target_consumption", factor * country.households.ts.current("target_consumption")
        )
        return result

    country._set_household_target_demand = scaled


def _scale_period_wages(country, factor):
    """Scale every period-t wage receipt (financing, realised income, taxes): T must not see it."""
    wages = country.nominal_employee_income

    def scaled():
        return factor * wages()

    country.nominal_employee_income = scaled


@pytest.mark.parametrize(
    "perturb,moved_series",
    [
        (_scale_desired_goods, ("consumption_before_floor", "nominal_amount_spent_in_lcu")),
        (_scale_period_wages, ("income_employee", "financing_income", "income")),
    ],
    ids=["desired_goods", "wages"],
)
def test_top_up_is_independent_of_period_t_outcomes(datawrapper, perturb, moved_series):
    """Invariant 3: T is fixed at period start from t-1 data; period-t demand, spending, wages and realised
    income cannot move it."""
    support = {"mode": "minimum_income", "goods_floor": True}
    # Construction seeds the global RNG: build and run each simulation in turn so both see the same draws.
    base = _production_simulation(datawrapper, support)
    base.iterate(t=0)
    shifted = _production_simulation(datawrapper, support)
    perturb(shifted.countries["FRA"], 1.5)
    shifted.iterate(t=0)

    hh_base, hh_shifted = base.countries["FRA"].households, shifted.countries["FRA"].households
    for name in moved_series:
        assert not np.array_equal(hh_base.ts.current(name), hh_shifted.ts.current(name)), name
    assert hh_base.ts.current("minimum_income_entitlement").sum() > 0.0
    for name in ("minimum_income_entitlement", "income_minimum_income"):
        np.testing.assert_array_equal(hh_shifted.ts.current(name), hh_base.ts.current(name), err_msg=name)
