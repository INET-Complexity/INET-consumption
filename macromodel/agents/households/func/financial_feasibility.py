"""Pure Stage 5 household financial-feasibility plans and policy."""

from dataclasses import dataclass, fields, replace

import numpy as np


def gross_liquidation_sale(net_sale: np.ndarray, sale_cost: np.ndarray, available_ifa: np.ndarray) -> np.ndarray:
    """Gross illiquid sale that delivers ``net_sale`` in cash after the sale cost.

    The cost is added once when anything is sold, and the sale is capped by
    the illiquid assets available. Shared by the reservation and the
    settlement check so both use identical arithmetic.
    """
    net_sale = np.asarray(net_sale, dtype=float)
    return np.minimum(net_sale + np.where(net_sale > 0.0, sale_cost, 0.0), available_ifa)


def forced_liquidation_cash(plan: "PostGrantFeasiblePlan") -> np.ndarray:
    """Cash the reserved illiquid sale adds to this period's ordinary cash needs.

    The gross reservation less the trade cost and less the part sold to pay for
    a property purchase, which the purchase spends.
    """
    if plan.reserved_liquidation_total is None:
        raise RuntimeError("Liquidation cash requires an executable liquidation reservation.")
    cash = np.asarray(plan.reserved_liquidation_total, dtype=float)
    for deduction in (plan.liquidation_sale_cost, plan.purchase_liquidation_total):
        if deduction is not None:
            cash = cash - np.asarray(deduction, dtype=float)
    return np.maximum(cash, 0.0)


def _freeze_plan_arrays(instance: object) -> None:
    """Defensively copy and freeze every array carried across Stage 5 stages."""
    for data_field in fields(instance):
        value = getattr(instance, data_field.name)
        if value is None:
            continue
        array = np.asarray(value).copy()
        array.setflags(write=False)
        object.__setattr__(instance, data_field.name, array)


@dataclass(frozen=True)
class PreGrantFeasiblePlan:
    """Provisional Stage 5 plan constructed before consumer-credit clearing."""

    liquidity_shortfall_before_repair: np.ndarray
    funded_from_liquid_assets: np.ndarray
    residual_shortfall_after_lfa: np.ndarray
    credit_requested: np.ndarray | None = None
    planned_liquidation_total: np.ndarray | None = None

    def __post_init__(self) -> None:
        _freeze_plan_arrays(self)


@dataclass(frozen=True)
class PostGrantFeasiblePlan:
    """Authoritative Stage 5 plan after consumer-credit clearing."""

    credit_granted: np.ndarray
    credit_rationing_gap: np.ndarray
    planned_liquidation_total: np.ndarray
    residual_shortfall_after_granted_credit: np.ndarray
    funded_from_liquid_assets: np.ndarray | None = None
    granted_consumer_credit_by_bank_and_household: np.ndarray | None = None
    consumer_debt_liability_booking: np.ndarray | None = None
    bank_consumer_loan_asset_booking: np.ndarray | None = None
    consumption_before_floor: np.ndarray | None = None
    residual_shortfall_before_floor: np.ndarray | None = None
    consumption_after_floor: np.ndarray | None = None
    consumption_cut_amount: np.ndarray | None = None
    remaining_subsistence_shortfall: np.ndarray | None = None
    floor_support: np.ndarray | None = None
    unpaid_consumer_service: np.ndarray | None = None
    unpaid_mortgage_service: np.ndarray | None = None
    investment_after_floor: np.ndarray | None = None
    early_consumer_repayment_capacity: np.ndarray | None = None
    floor_binding: np.ndarray | None = None
    post_liquidation_lfa: np.ndarray | None = None
    post_liquidation_ifa: np.ndarray | None = None
    settled_liquidation_total: np.ndarray | None = None
    residual_shortfall_after_lfa: np.ndarray | None = None
    reserved_liquidation_total: np.ndarray | None = None
    liquidation_reservation_ifa: np.ndarray | None = None
    liquidation_sale_cost: np.ndarray | None = None
    purchase_liquidation_total: np.ndarray | None = None

    def __post_init__(self) -> None:
        _freeze_plan_arrays(self)


class HouseholdFinancialFeasibility:
    """Own Stage 5 plan construction without mutating household balances."""

    @staticmethod
    def _non_negative(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        return np.where(np.isfinite(array), np.maximum(array, 0.0), 0.0)

    def build_pre_grant_plan(
        self,
        *,
        liquidity_shortfall_before_repair: np.ndarray,
        funded_from_liquid_assets: np.ndarray,
        residual_shortfall_after_lfa: np.ndarray,
    ) -> PreGrantFeasiblePlan:
        return PreGrantFeasiblePlan(
            liquidity_shortfall_before_repair=self._non_negative(liquidity_shortfall_before_repair).copy(),
            funded_from_liquid_assets=self._non_negative(funded_from_liquid_assets).copy(),
            residual_shortfall_after_lfa=self._non_negative(residual_shortfall_after_lfa).copy(),
        )

    def with_credit_requested(self, plan: PreGrantFeasiblePlan, credit_requested: np.ndarray) -> PreGrantFeasiblePlan:
        return replace(plan, credit_requested=self._non_negative(credit_requested).copy())

    def with_planned_liquidation(
        self,
        plan: PreGrantFeasiblePlan,
        *,
        planned_liquidation_total: np.ndarray,
        available_illiquid_assets: np.ndarray,
    ) -> PreGrantFeasiblePlan:
        liquidation = self._non_negative(planned_liquidation_total)
        available = self._non_negative(available_illiquid_assets)
        if liquidation.shape != available.shape:
            raise ValueError("planned liquidation and available illiquid assets must have the same shape")
        return replace(plan, planned_liquidation_total=np.minimum(liquidation, available).copy())

    def build_post_grant_plan(
        self,
        plan: PreGrantFeasiblePlan,
        *,
        credit_granted: np.ndarray,
        granted_consumer_credit_by_bank_and_household: np.ndarray | None = None,
    ) -> PostGrantFeasiblePlan:
        if plan.credit_requested is None or plan.planned_liquidation_total is None:
            raise RuntimeError("Post-grant reconciliation requires complete pre-grant credit and liquidation fields.")
        requested = self._non_negative(plan.credit_requested)
        granted = self._non_negative(credit_granted)
        liquidation = self._non_negative(plan.planned_liquidation_total)
        residual = self._non_negative(plan.residual_shortfall_after_lfa)
        if not (requested.shape == granted.shape == liquidation.shape == residual.shape):
            raise ValueError("All household feasibility vectors must have the same shape.")

        matrix = None
        liabilities = None
        bank_assets = None
        if granted_consumer_credit_by_bank_and_household is not None:
            matrix = np.asarray(granted_consumer_credit_by_bank_and_household, dtype=float)
            if matrix.ndim != 2 or matrix.shape[1] != granted.shape[0]:
                raise ValueError("Granted-credit settlement must be a bank-by-household matrix.")
            if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
                raise RuntimeError("Granted-credit settlement must be finite and non-negative.")
            liabilities = matrix.sum(axis=0)
            if not np.allclose(liabilities, granted, rtol=1e-10, atol=1e-8):
                raise RuntimeError("Granted-credit settlement does not reconcile with household credit granted.")
            bank_assets = matrix.sum(axis=1)

        return PostGrantFeasiblePlan(
            credit_granted=granted.copy(),
            credit_rationing_gap=np.maximum(requested - granted, 0.0).copy(),
            planned_liquidation_total=liquidation.copy(),
            residual_shortfall_after_granted_credit=np.maximum(residual - granted - liquidation, 0.0).copy(),
            funded_from_liquid_assets=plan.funded_from_liquid_assets.copy(),
            granted_consumer_credit_by_bank_and_household=None if matrix is None else matrix.copy(),
            consumer_debt_liability_booking=None if liabilities is None else liabilities.copy(),
            bank_consumer_loan_asset_booking=None if bank_assets is None else bank_assets.copy(),
            residual_shortfall_after_lfa=residual.copy(),
        )

    def reserve_executable_liquidation(
        self,
        plan: PostGrantFeasiblePlan,
        *,
        available_pre_stage4_ifa: np.ndarray,
        sale_cost: np.ndarray | None = None,
        purchase_liquidation: np.ndarray | None = None,
    ) -> PostGrantFeasiblePlan:
        """Reserve the Stage 5 liquidation before floors and goods demand are fixed.

        ``planned_liquidation_total`` stays the net forced need (capped by IFA).
        ``purchase_liquidation_total`` is the net sale that pays for a completed
        property purchase. ``reserved_liquidation_total`` is the gross sale of
        both, which also pays the illiquid-trade cost once;
        ``liquidation_sale_cost`` is that cost, at most the gross sale. Only the
        forced part, net of the cost, is cash for this period's other needs.
        """
        if plan.reserved_liquidation_total is not None:
            raise RuntimeError("Stage 5 executable liquidation has already been reserved for this period.")
        available = self._non_negative(available_pre_stage4_ifa)
        planned = self._non_negative(plan.planned_liquidation_total)
        granted = self._non_negative(plan.credit_granted)
        residual_after_lfa = plan.residual_shortfall_after_lfa
        if residual_after_lfa is None:
            raise RuntimeError("Stage 5 liquidation reservation requires residual_shortfall_after_lfa.")
        residual = self._non_negative(residual_after_lfa)
        if not (available.shape == planned.shape == granted.shape == residual.shape):
            raise ValueError("Stage 5 liquidation reservation requires equal-length household vectors.")
        cost = np.zeros_like(available) if sale_cost is None else self._non_negative(sale_cost)
        if cost.shape != available.shape:
            raise ValueError("Stage 5 liquidation sale cost must be a household vector.")
        purchase = (
            np.zeros_like(available) if purchase_liquidation is None else self._non_negative(purchase_liquidation)
        )
        if purchase.shape != available.shape:
            raise ValueError("Stage 5 purchase liquidation must be a household vector.")
        net = np.minimum(planned, available)
        reserved = gross_liquidation_sale(net + purchase, cost, available)
        cost_paid = np.minimum(np.where(reserved > 0.0, cost, 0.0), reserved)
        # The completion rule only lets a purchase through if the illiquid
        # assets cover the forced sale, the purchase sale and the cost.
        if np.any((purchase > 0.0) & (reserved - cost_paid - net < purchase - (1e-6 + 1e-9 * purchase))):
            raise RuntimeError("Illiquid assets cannot fund the sale reserved for a completed property purchase.")
        cash = np.maximum(reserved - cost_paid - purchase, 0.0)
        return replace(
            plan,
            planned_liquidation_total=net.copy(),
            reserved_liquidation_total=reserved.copy(),
            liquidation_sale_cost=cost_paid.copy(),
            purchase_liquidation_total=purchase.copy(),
            liquidation_reservation_ifa=available.copy(),
            residual_shortfall_after_granted_credit=np.maximum(residual - granted - cash, 0.0).copy(),
        )

    def refresh_cash_needs(
        self,
        plan: PostGrantFeasiblePlan,
        *,
        cash_uses: np.ndarray,
        cash_income: np.ndarray,
        available_lfa: np.ndarray,
    ) -> PostGrantFeasiblePlan:
        """Reallocate opening liquidity against current nominal uses and fixed finance.

        The loan and liquidation authorities are immutable. Their proceeds are
        counted here once; neither operation is booked by this refresh.
        """
        shape = np.asarray(plan.credit_granted).shape
        if len(shape) != 1:
            raise ValueError("Credit granted must be a household vector.")
        if plan.reserved_liquidation_total is None:
            raise RuntimeError("Post-labour refresh requires executable liquidation authority.")
        arrays = {}
        for name, values in (
            ("cash_uses", cash_uses),
            ("cash_income", cash_income),
            ("available_lfa", available_lfa),
            ("credit_granted", plan.credit_granted),
            ("reserved_liquidation_total", plan.reserved_liquidation_total),
        ):
            array = np.asarray(values, dtype=float)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite household vector.")
            if name in ("cash_uses", "credit_granted", "reserved_liquidation_total") and np.any(array < 0.0):
                raise ValueError(f"{name} must be non-negative.")
            arrays[name] = array
        if plan.consumption_after_floor is not None:
            raise RuntimeError("Cash needs must be refreshed before floor/payment settlement.")
        liquidation_cash = forced_liquidation_cash(plan)
        need = np.maximum(arrays["cash_uses"] - arrays["cash_income"], 0.0)
        funded = np.minimum(need, np.maximum(arrays["available_lfa"], 0.0))
        residual = need - funded
        return replace(
            plan,
            funded_from_liquid_assets=funded,
            residual_shortfall_after_lfa=residual,
            residual_shortfall_after_granted_credit=np.maximum(
                residual - arrays["credit_granted"] - liquidation_cash, 0.0
            ),
        )

    def settle_consumption_floor(
        self,
        plan: PostGrantFeasiblePlan,
        *,
        consumption_before_floor: np.ndarray,
        subsistence_floor: np.ndarray,
        cash_rent: np.ndarray,
        resources: np.ndarray,
        scheduled_consumer_service: np.ndarray,
        scheduled_mortgage_service: np.ndarray,
        investment: np.ndarray,
    ) -> PostGrantFeasiblePlan:
        """Allocate resources in the Stage 5 payment order, without writing household state.

        All inputs are nominal cash amounts per household. ``resources`` (A) is
        income without support plus liquid assets, granted consumer credit and
        the reserved liquidation net of its cost. The order is: goods up to the
        floor F and cash rent R; debt service O; goods above the floor up to
        desired goods G; investment V.

            S   = max(F + R - A, 0)             A1 = A + S - F - R
            P   = min(O, A1)                    U  = O - P
            C^f = F + min(max(G - F, 0), A1 - P)
            V^p = min(V, A1 - P - (C^f - F))

        Support S covers the goods floor and rent only. Unpaid service U is
        suspended on consumer credit first, then on the mortgage.
        """
        desired = self._non_negative(consumption_before_floor)
        floor = self._non_negative(subsistence_floor)
        rent = self._non_negative(cash_rent)
        consumer_service = self._non_negative(scheduled_consumer_service)
        mortgage_service = self._non_negative(scheduled_mortgage_service)
        desired_investment = self._non_negative(investment)
        available = np.asarray(resources, dtype=float)
        residual_before_floor = self._non_negative(plan.residual_shortfall_after_granted_credit)
        shape = residual_before_floor.shape
        for values in (desired, floor, rent, consumer_service, mortgage_service, desired_investment, available):
            if values.shape != shape:
                raise ValueError("Stage 5 consumption-floor inputs must be equal-length household vectors.")
        if not np.all(np.isfinite(available)):
            raise ValueError("Stage 5 consumption-floor resources must be finite.")

        support = np.maximum(floor + rent - available, 0.0)
        after_floor_and_rent = np.maximum(available + support - floor - rent, 0.0)
        service = consumer_service + mortgage_service
        service_paid = np.minimum(service, after_floor_and_rent)
        unpaid_service = service - service_paid
        unpaid_consumer = np.minimum(unpaid_service, consumer_service)
        unpaid_mortgage = unpaid_service - unpaid_consumer
        after_service = after_floor_and_rent - service_paid
        consumption_above_floor = np.minimum(np.maximum(desired - floor, 0.0), after_service)
        consumption_after_floor = floor + consumption_above_floor
        investment_after_floor = np.minimum(desired_investment, after_service - consumption_above_floor)
        consumption_cut_amount = np.maximum(desired - consumption_after_floor, 0.0)
        floor_top_up = np.maximum(consumption_after_floor - desired, 0.0)
        return replace(
            plan,
            consumption_before_floor=desired,
            residual_shortfall_before_floor=residual_before_floor,
            consumption_after_floor=consumption_after_floor,
            consumption_cut_amount=consumption_cut_amount,
            remaining_subsistence_shortfall=support + unpaid_consumer + unpaid_mortgage,
            floor_support=support,
            unpaid_consumer_service=unpaid_consumer,
            unpaid_mortgage_service=unpaid_mortgage,
            investment_after_floor=investment_after_floor,
            floor_binding=(consumption_cut_amount + floor_top_up) > 0.0,
        )
