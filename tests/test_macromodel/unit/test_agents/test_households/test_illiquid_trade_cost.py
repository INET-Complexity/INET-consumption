"""Increment (a): one illiquid-trade cost per period, paid by households to banks."""

from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.agents.households.func.financial_feasibility import (
    HouseholdFinancialFeasibility,
    PostGrantFeasiblePlan,
    gross_liquidation_sale,
)
from macromodel.agents.households.func.portfolio_rebalancing import PortfolioRebalancingResult
from macromodel.agents.households.func.portfolio_settlement import settle_portfolio_reallocation
from macromodel.agents.households.func.post_liquidation_settlement import settle_post_liquidation
from macromodel.country.country import Country


def _plan(*, planned, residual_after_lfa, granted=0.0):
    n = len(planned)
    zeros = np.zeros(n)
    return PostGrantFeasiblePlan(
        credit_granted=np.full(n, granted),
        credit_rationing_gap=zeros.copy(),
        planned_liquidation_total=np.asarray(planned, dtype=float),
        residual_shortfall_after_granted_credit=zeros.copy(),
        residual_shortfall_after_lfa=np.asarray(residual_after_lfa, dtype=float),
    )


@pytest.mark.parametrize(
    "planned,ifa,cost,gross,cost_paid,residual",
    [
        (80.0, 100.0, 5.0, 85.0, 5.0, 0.0),  # sale grossed up by the cost; need met
        (98.0, 100.0, 5.0, 100.0, 5.0, 3.0),  # IFA cap binds: 95 cash against a need of 98
        (2.0, 2.0, 5.0, 2.0, 2.0, 2.0),  # tiny sale: cost capped by the sale, never lowers LFA
        (0.0, 100.0, 5.0, 0.0, 0.0, 0.0),  # no sale, no cost
    ],
)
def test_reservation_sells_gross_and_counts_only_net_cash(planned, ifa, cost, gross, cost_paid, residual):
    plan = HouseholdFinancialFeasibility().reserve_executable_liquidation(
        _plan(planned=[planned], residual_after_lfa=[planned]),
        available_pre_stage4_ifa=np.array([ifa]),
        sale_cost=np.array([cost]),
    )

    assert plan.planned_liquidation_total[0] == pytest.approx(min(planned, ifa))  # net need unchanged
    assert plan.reserved_liquidation_total[0] == pytest.approx(gross)
    assert plan.liquidation_sale_cost[0] == pytest.approx(cost_paid)
    assert plan.liquidation_sale_cost[0] <= plan.reserved_liquidation_total[0]
    assert plan.residual_shortfall_after_granted_credit[0] == pytest.approx(residual)


def test_reservation_without_cost_is_unchanged():
    plan = HouseholdFinancialFeasibility().reserve_executable_liquidation(
        _plan(planned=[80.0], residual_after_lfa=[100.0]),
        available_pre_stage4_ifa=np.array([100.0]),
    )
    assert plan.reserved_liquidation_total[0] == 80.0
    assert plan.liquidation_sale_cost[0] == 0.0
    assert plan.residual_shortfall_after_granted_credit[0] == pytest.approx(20.0)


def test_refresh_counts_the_sale_net_of_its_cost():
    feasibility = HouseholdFinancialFeasibility()
    plan = feasibility.reserve_executable_liquidation(
        _plan(planned=[80.0], residual_after_lfa=[80.0]),
        available_pre_stage4_ifa=np.array([100.0]),
        sale_cost=np.array([5.0]),
    )
    refreshed = feasibility.refresh_cash_needs(
        plan, cash_uses=np.array([100.0]), cash_income=np.array([10.0]), available_lfa=np.array([0.0])
    )
    # Need 90, cash from the sale 85 - 5 = 80.
    assert refreshed.residual_shortfall_after_granted_credit[0] == pytest.approx(10.0)


def test_gross_sale_rule_is_shared_by_reservation_and_check():
    np.testing.assert_allclose(
        gross_liquidation_sale(np.array([0.0, 80.0, 98.0]), np.array([5.0, 5.0, 5.0]), np.array([100.0] * 3)),
        [0.0, 85.0, 100.0],
    )


def test_settlement_sells_gross_and_credits_net_of_cost():
    settled = settle_post_liquidation(
        base_lfa=np.array([0.0]),
        base_ifa=np.array([100.0]),
        planned_liquidation_total=np.array([85.0]),
        residual_shortfall_after_granted_credit=np.zeros(1),
        sale_cost=np.array([5.0]),
    )
    assert settled.post_liquidation_ifa[0] == pytest.approx(15.0)
    assert settled.post_liquidation_lfa[0] == pytest.approx(80.0)
    with pytest.raises(RuntimeError, match="between zero and the gross sale"):
        settle_post_liquidation(
            base_lfa=np.array([0.0]),
            base_ifa=np.array([100.0]),
            planned_liquidation_total=np.array([3.0]),
            residual_shortfall_after_granted_credit=np.zeros(1),
            sale_cost=np.array([5.0]),
        )


def _rebalancing_buying_ifa(delta_ifa: float, cost: float) -> PortfolioRebalancingResult:
    one = np.ones(1)
    flag = np.zeros(1, dtype=bool)
    return PortfolioRebalancingResult(
        portfolio_participates=np.ones(1, dtype=bool),
        actual_illiquid_share=0.0 * one,
        target_illiquid_assets=0.0 * one,
        delta_tilde=0.0 * one,
        kappa_star_tilde=0.0 * one,
        kappa_tilde=0.0 * one,
        desired_illiquid_adjustment=delta_ifa * one,
        adjustment_cost=cost * one,
        counterfactual_lfa_flow=(-delta_ifa - cost) * one,
        counterfactual_ifa_flow=delta_ifa * one,
        inaction_flag=flag.copy(),
        upper_bound_flag=flag.copy(),
        lower_bound_flag=flag.copy(),
        infeasible_interval_flag=flag.copy(),
        no_financial_assets_flag=flag.copy(),
        portfolio_valid_flag=np.ones(1, dtype=bool),
    )


@pytest.mark.parametrize("waived,closing_lfa,cost", [(False, 79.0, 1.0), (True, 80.0, 0.0)])
def test_portfolio_waiver_zeroes_only_the_cost(waived, closing_lfa, cost):
    settlement = settle_portfolio_reallocation(
        base_lfa=np.array([100.0]),
        base_ifa=np.array([0.0]),
        investable_surplus=np.array([50.0]),
        rebalancing=_rebalancing_buying_ifa(20.0, 1.0),
        settlement_enabled=True,
        forced_liquidation_active=np.zeros(1, dtype=bool),
        waive_fixed_cost=np.array([waived]),
    )
    # Waived: a pure LFA-to-IFA transfer; no money created or destroyed.
    assert settlement.closing_lfa[0] == pytest.approx(closing_lfa)
    assert settlement.closing_ifa[0] == pytest.approx(20.0)
    assert settlement.committed_adjustment_cost[0] == pytest.approx(cost)
    assert settlement.closing_tfa[0] == pytest.approx(100.0 - cost)


def test_fee_income_by_bank_matches_household_cost_paid():
    country = Country.__new__(Country)
    cost_paid = np.array([1.0, 2.0, 0.0, 4.0])
    country.households = SimpleNamespace(
        states={"Corresponding Bank ID": np.array([0, 1, 1, 0])},
        ts=SimpleNamespace(current=lambda key: {"illiquid_trade_cost_paid": cost_paid}[key]),
    )
    country.banks = SimpleNamespace(ts=SimpleNamespace(current=lambda key: {"n_banks": 3}[key]))

    fees = country.household_trade_fee_income_by_bank()

    np.testing.assert_allclose(fees, [5.0, 2.0, 0.0])
    assert fees.sum() == pytest.approx(cost_paid.sum())  # flow-matching residual (ii)


def test_bank_profits_include_sale_fee_income(test_banks):
    n = test_banks.ts.current("n_banks")
    profits = test_banks.compute_profits().copy()
    cash_profits = test_banks.compute_cash_distributable_profits().copy()
    test_banks.ts.override_current("household_sale_fee_income", np.full(n, 7.0))
    np.testing.assert_allclose(test_banks.compute_profits(), profits + 7.0)
    np.testing.assert_allclose(test_banks.compute_cash_distributable_profits(), cash_profits + 7.0)
