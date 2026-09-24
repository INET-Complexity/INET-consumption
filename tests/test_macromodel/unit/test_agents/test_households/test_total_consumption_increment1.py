"""Cash-unit and settled-authority regressions for total consumption Increment 1."""

from dataclasses import fields

import numpy as np
import pytest

from macromodel.agents.households.func.consumption import CreditAugmentedConsumption
from macromodel.agents.households.func.financial_feasibility import HouseholdFinancialFeasibility, PostGrantFeasiblePlan
from tests.test_macromodel.unit.test_agents.test_households.func.test_consumption import (
    TestCreditAugmentedHouseholdConsumption as ConsumptionFixture,
)


@pytest.mark.parametrize(
    "rent,imputed", [(0, 0), (20, 0), (30, 0), (0, 20), (0, 30), (120, 0), (0, 120), (60, 60), (-10, -20)]
)
def test_total_goods_cash_and_mpc(rent, imputed):
    rule = CreditAugmentedConsumption()
    args = ConsumptionFixture()._housing_carve_out_args(n_households=1)
    args["tau_vat"] = 0.2
    rule.compute_target_consumption(**args, rent=np.zeros(1), rent_imputed=np.zeros(1))
    total = rule.last_target_consumption_components["target_consumption_calibrated_total"].copy()
    legacy_mpc = rule.last_formula_implied_mpc.copy()
    target_mpc = rule.last_target_consumption_components["target_consumption_total_mpc"].copy()
    goods = rule.compute_target_consumption(**args, rent=np.array([rent]), rent_imputed=np.array([imputed]))
    r, h = max(rent, 0), max(imputed, 0)
    np.testing.assert_allclose(1.2 * goods.sum(axis=1), np.maximum(0, total - r - h))
    np.testing.assert_allclose(1.2 * goods.sum(axis=1) + r, total - h + np.maximum(0, r + h - total))
    np.testing.assert_allclose(rule.last_formula_implied_mpc, legacy_mpc)
    np.testing.assert_allclose(rule.last_target_consumption_components["target_consumption_total_mpc"], target_mpc)
    np.testing.assert_allclose(target_mpc, 1.2 * legacy_mpc)
    np.testing.assert_allclose(rule.last_target_consumption_components["target_consumption_calibrated_total"], total)


@pytest.mark.parametrize("uses,expected_funded,expected_gap", [(100, 20, 40), (45, 20, 0), (10, 0, 0)])
def test_refresh_keeps_every_finance_authority(uses, expected_funded, expected_gap):
    plan = PostGrantFeasiblePlan(
        credit_granted=np.array([15.0]),
        credit_rationing_gap=np.array([8.0]),
        planned_liquidation_total=np.array([5.0]),
        reserved_liquidation_total=np.array([5.0]),
        liquidation_reservation_ifa=np.array([12.0]),
        residual_shortfall_after_granted_credit=np.array([999.0]),
        granted_consumer_credit_by_bank_and_household=np.array([[10.0], [5.0]]),
        consumer_debt_liability_booking=np.array([15.0]),
        bank_consumer_loan_asset_booking=np.array([10.0, 5.0]),
    )
    updated = HouseholdFinancialFeasibility().refresh_cash_needs(
        plan,
        cash_uses=np.array([uses]),
        cash_income=np.array([20.0]),
        available_lfa=np.array([20.0]),
    )
    np.testing.assert_allclose(updated.funded_from_liquid_assets, [expected_funded])
    np.testing.assert_allclose(updated.residual_shortfall_after_granted_credit, [expected_gap])
    for field in fields(plan):
        if field.name not in {
            "funded_from_liquid_assets",
            "residual_shortfall_after_lfa",
            "residual_shortfall_after_granted_credit",
        }:
            np.testing.assert_equal(getattr(updated, field.name), getattr(plan, field.name))
    np.testing.assert_allclose(plan.residual_shortfall_after_granted_credit, [999.0])


@pytest.mark.parametrize("bad", [np.array([np.nan]), np.array([np.inf]), np.zeros(2)])
def test_refresh_rejects_malformed_funding(bad):
    plan = PostGrantFeasiblePlan(
        credit_granted=np.zeros(1),
        credit_rationing_gap=np.zeros(1),
        planned_liquidation_total=np.zeros(1),
        reserved_liquidation_total=np.zeros(1),
        residual_shortfall_after_granted_credit=np.zeros(1),
    )
    with pytest.raises(ValueError, match="finite household vector"):
        HouseholdFinancialFeasibility().refresh_cash_needs(
            plan, cash_uses=bad, cash_income=np.zeros(1), available_lfa=np.zeros(1)
        )


def test_zero_goods_floor_is_executable_in_net_units(test_households):
    h = test_households
    n = h.ts.current("n_households")
    h.uses_feasibility_resolver = True
    h.consumption_vat_rate = 0.2
    h.ts.override_current("target_consumption", np.zeros_like(h.ts.current("target_consumption")))
    h.ts.override_current("target_investment", np.zeros_like(h.ts.current("target_investment")))
    h.post_grant_feasible_plan = PostGrantFeasiblePlan(
        credit_granted=np.zeros(n),
        credit_rationing_gap=np.zeros(n),
        planned_liquidation_total=np.zeros(n),
        residual_shortfall_after_granted_credit=np.full(n, 30.0),
    )
    shortfall = h.prepare_goods_market_clearing(exchange_rate_usd_to_lcu=1.0, subsistence_consumption=np.full(n, 12.0))
    np.testing.assert_allclose(h.ts.current("target_consumption").sum(axis=1), 10.0)
    np.testing.assert_allclose(shortfall, 42.0)  # 30 mandatory rent plus gross goods floor 12


def test_refresh_counts_investment_property_and_mortgage_once(test_households):
    h = test_households
    n = h.ts.current("n_households")
    h.consumption_vat_rate = 0.2
    h.investment_tax_rate = 0.1
    for key, value in {
        "expected_income": 50,
        "rent": 30,
        "liquid_financial_assets": 10,
        "received_mortgages": 40,
        "price_paid_for_property": 60,
    }.items():
        h.ts.override_current(key, np.full(n, value, dtype=float))
    for key, total in [("target_consumption", 100), ("target_investment", 20)]:
        matrix = np.zeros_like(h.ts.current(key))
        matrix[:, 0] = total
        h.ts.override_current(key, matrix)
    h.post_grant_feasible_plan = PostGrantFeasiblePlan(
        credit_granted=np.full(n, 15.0),
        credit_rationing_gap=np.zeros(n),
        planned_liquidation_total=np.full(n, 5.0),
        reserved_liquidation_total=np.full(n, 5.0),
        residual_shortfall_after_granted_credit=np.zeros(n),
    )
    h.refresh_post_labour_feasibility(np.full(n, 8.0))
    # Uses = 120+30+22+60+8=240; income+new mortgage=90; liquid+credit+sale=30.
    np.testing.assert_allclose(h.post_grant_feasible_plan.residual_shortfall_after_granted_credit, 120.0)
    np.testing.assert_allclose(h.post_grant_feasible_plan.credit_granted, 15.0)


@pytest.mark.parametrize("resolver", [False, True])
def test_wealth_debits_consumption_vat_once_and_excludes_investment(test_households, monkeypatch, resolver):
    from tests.test_macromodel.unit.test_agents.test_households.test_households import (
        TestHouseholdsUpdateWealthPortfolioSettlement,
    )

    h = test_households
    fixture = TestHouseholdsUpdateWealthPortfolioSettlement()
    fixture._configure_update_wealth(h, monkeypatch, resolver=resolver, settles=False, use_real_liquidation=True)
    n = h.ts.current("n_households")
    h.ts.override_current("consumption", np.full(n, 40.0))
    h.ts.override_current("rent", np.full(n, 5.0))
    matrix = np.zeros_like(h.ts.current("nominal_amount_spent_in_lcu"))
    matrix[:, 0] = 50.0
    h.ts.override_current("nominal_amount_spent_in_lcu", matrix)
    inv = np.zeros_like(h.ts.current("investment"))
    inv[:, 0] = 10.0
    h.ts.override_current("investment", inv)
    captured = {}

    def distribute(**kwargs):
        captured["surplus"] = kwargs["new_wealth"].copy()
        return kwargs["new_wealth"], np.zeros(n)

    monkeypatch.setattr(h.functions["wealth"], "distribute_new_wealth", distribute)
    h.update_wealth(housing_data=None, tau_cf=0.0, tau_vat=0.2)
    np.testing.assert_allclose(captured["surplus"], 37.0)  # 100 - 50 purchases - 5 rent - 8 consumption VAT


def test_early_repayment_reserves_rent_and_gross_investment(test_households):
    h = test_households
    n = h.ts.current("n_households")
    h.investment_tax_rate = 0.2
    for key, value in [("expected_income", 100.0), ("rent", 30.0)]:
        h.ts.override_current(key, np.full(n, value))
    inv = np.zeros_like(h.ts.current("target_investment"))
    inv[:, 0] = 10.0
    h.ts.override_current("target_investment", inv)
    h.post_grant_feasible_plan = PostGrantFeasiblePlan(
        credit_granted=np.zeros(n),
        credit_rationing_gap=np.zeros(n),
        planned_liquidation_total=np.zeros(n),
        residual_shortfall_after_granted_credit=np.zeros(n),
        consumption_after_floor=np.full(n, 24.0),
    )
    h.populate_post_grant_early_repayment_capacity(
        mortgage_service=np.full(n, 10.0),
        scheduled_consumer_service=np.full(n, 5.0),
        eligible_ficp=np.ones(n, dtype=bool),
    )
    np.testing.assert_allclose(h.current_early_consumer_repayment_capacity(), 19.0)


@pytest.mark.parametrize("funded", [False, True])
def test_legacy_rent_shortfall_is_explicit(test_households, funded):
    h = test_households
    n = h.ts.current("n_households")
    for key in (
        "expected_income",
        "wealth_financial_assets",
        "received_consumption_loans",
        "received_mortgages",
        "interest_paid",
        "debt_installments",
        "price_paid_for_property",
    ):
        h.ts.override_current(key, np.zeros(n))
    h.ts.override_current("rent", np.full(n, 30.0))
    if funded:
        h.ts.override_current("received_consumption_loans", np.full(n, 30.0))
        h.validate_legacy_mandatory_cash_uses()
    else:
        with pytest.raises(RuntimeError, match="cannot fund mandatory cash rent"):
            h.validate_legacy_mandatory_cash_uses()
