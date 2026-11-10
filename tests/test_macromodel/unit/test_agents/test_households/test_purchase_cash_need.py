"""Increment (b): the cash a buyer must keep for this period (N) and the housing cash need (X, D)."""

import numpy as np

from macromodel.agents.households.func.financial_feasibility import PostGrantFeasiblePlan


def test_planning_cash_need_reserves_goods_rent_and_service_beyond_expected_income(test_households):
    h = test_households
    n = h.ts.current("n_households")
    h.consumption_vat_rate = 0.2
    h._known_receipts_ready = False
    goods = np.zeros_like(h.ts.current("target_consumption"))
    goods[:, 0] = 100.0
    h.ts.override_current("target_consumption", goods)
    h.ts.override_current("expected_income", np.where(np.arange(n) % 2 == 0, 100.0, 500.0))
    # Investment is not reserved, and the planning rent snapshot is used, not the last rent.
    investment = np.zeros_like(h.ts.current("target_investment"))
    investment[:, 0] = 1000.0
    h.ts.override_current("target_investment", investment)
    h.ts.override_current("rent", np.full(n, 999.0))
    h.planning_cash_rent = np.full(n, 30.0)

    need = h.planning_cash_need(np.full(n, 8.0))

    # 120 goods incl. VAT + 30 rent + 8 service = 158.
    np.testing.assert_allclose(need, np.where(np.arange(n) % 2 == 0, 58.0, 0.0))


def test_housing_cash_need_and_lfa_draw_follow_the_booked_flows(test_households):
    h = test_households
    n = h.ts.current("n_households")
    zeros = np.zeros(n)
    for key, value in {
        "price_paid_for_property": 100.0,
        "received_mortgages": 50.0,
        "property_sale_proceeds": 40.0,
        "mortgage_sale_repayment": 15.0,
    }.items():
        h.ts.override_current(key, np.full(n, value))
    h.post_grant_feasible_plan = PostGrantFeasiblePlan(
        credit_granted=zeros.copy(),
        credit_rationing_gap=zeros.copy(),
        planned_liquidation_total=zeros.copy(),
        residual_shortfall_after_granted_credit=zeros.copy(),
        purchase_liquidation_total=np.full(n, 10.0),
    )

    # X = 100 - 50 - (40 - 15) = 25; 10 comes from the illiquid sale, 15 from liquid assets.
    np.testing.assert_allclose(h.current_net_housing_cash_need(), 25.0)
    np.testing.assert_allclose(h.current_housing_lfa_draw(), 15.0)
