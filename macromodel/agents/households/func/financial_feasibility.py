"""Pure Stage 5 household financial-feasibility plans and policy."""

from dataclasses import dataclass, fields, replace

import numpy as np

# Relative margin by which the plan rounds discretionary spending down. A plan that spends all its resources
# would otherwise close at about -1e-9 after the realised ledger re-adds the same flows in another order.
PLAN_ROUND_DOWN = 1e-12


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
    # Part of planned_liquidation_total that repays the opening overdraft.
    overdraft_liquidation_total: np.ndarray | None = None

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
    # Minimum-income allocation (support off): rent and goods floor the household's own resources leave unpaid,
    # and the negative part of those resources, max(-A, 0), that nobody covers. The floor settlement sets all
    # three; they are zero whenever support is enabled.
    unpaid_rent: np.ndarray | None = None
    unmet_floor: np.ndarray | None = None
    resource_deficit: np.ndarray | None = None
    # Overdraft interest due on the opening negative liquid balance, and the part the allocation leaves
    # unpaid; the unpaid part becomes consumer interest arrears at the household's bank.
    scheduled_overdraft_interest: np.ndarray | None = None
    unpaid_overdraft_interest: np.ndarray | None = None
    # Opening overdraft max(D - LFA, 0) the allocation repays as a senior use; the rest stays overdrawn.
    overdraft_repayment: np.ndarray | None = None
    # Part of planned_liquidation_total (net) that repays the opening overdraft.
    overdraft_liquidation_total: np.ndarray | None = None

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
        overdraft_liquidation_total: np.ndarray | None = None,
    ) -> PreGrantFeasiblePlan:
        liquidation = self._non_negative(planned_liquidation_total)
        available = self._non_negative(available_illiquid_assets)
        if liquidation.shape != available.shape:
            raise ValueError("planned liquidation and available illiquid assets must have the same shape")
        planned = np.minimum(liquidation, available)
        overdraft = (
            np.zeros_like(planned)
            if overdraft_liquidation_total is None
            else np.minimum(self._non_negative(overdraft_liquidation_total), planned)
        )
        if overdraft.shape != planned.shape:
            raise ValueError("overdraft liquidation must be a household vector")
        return replace(plan, planned_liquidation_total=planned.copy(), overdraft_liquidation_total=overdraft.copy())

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
            overdraft_liquidation_total=(
                None if plan.overdraft_liquidation_total is None else plan.overdraft_liquidation_total.copy()
            ),
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
        own_sale_surplus: np.ndarray | None = None,
        minimum_sale_size: np.ndarray | None = None,
    ) -> PostGrantFeasiblePlan:
        """Reserve the Stage 5 liquidation before floors and goods demand are fixed.

        The forced need was planned before housing clearing. ``own_sale_surplus`` is
        the household's own property-sale surplus max(-X, 0), cash of this period
        that covers the shortfall first: the planned sale is netted by it, so
        illiquid assets are sold only for the residual gap (paper financing order).
        ``planned_liquidation_total`` becomes that net forced need (capped by IFA).
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
        surplus = np.zeros_like(available) if own_sale_surplus is None else self._non_negative(own_sale_surplus)
        if surplus.shape != available.shape:
            raise ValueError("Stage 5 own-sale surplus must be a household vector.")
        net = np.minimum(np.maximum(planned - surplus, 0.0), available)
        reserved = gross_liquidation_sale(net + purchase, cost, available)
        cost_paid = np.minimum(np.where(reserved > 0.0, cost, 0.0), reserved)
        # Minimum sale size (paper fixed-cost logic): a forced sale is not executed when the cash it
        # must raise does not exceed the fixed trade cost (paying the cost to raise less than the cost is
        # a dust sale), nor when its gross amount delivers no cash after the cost. The shortfall then
        # stays in residual_shortfall_after_granted_credit and is handled as an unpaid obligation or
        # overdraft, like any shortfall the illiquid assets cannot cover. A sale that pays for a completed
        # property purchase is never skipped here: the completion rule has already sized it.
        # Absolute minimum: a gross sale below ``minimum_sale_size`` is not executed either.
        minimum = np.zeros_like(available) if minimum_sale_size is None else self._non_negative(minimum_sale_size)
        if minimum.shape != available.shape:
            raise ValueError("Stage 5 minimum sale size must be a household vector.")
        dust_sale = (
            (purchase <= 0.0)
            & (reserved > 0.0)
            & ((net <= cost) | (reserved - cost_paid <= 0.0) | (reserved < minimum))
        )
        net = np.where(dust_sale, 0.0, net)
        reserved = np.where(dust_sale, 0.0, reserved)
        cost_paid = np.where(dust_sale, 0.0, cost_paid)
        # The completion rule only lets a purchase through if the illiquid
        # assets cover the forced sale, the purchase sale and the cost.
        if np.any((purchase > 0.0) & (reserved - cost_paid - net < purchase - (1e-6 + 1e-9 * purchase))):
            raise RuntimeError("Illiquid assets cannot fund the sale reserved for a completed property purchase.")
        cash = np.maximum(reserved - cost_paid - purchase, 0.0)
        overdraft = (
            None if plan.overdraft_liquidation_total is None else np.minimum(plan.overdraft_liquidation_total, net)
        )
        return replace(
            plan,
            planned_liquidation_total=net.copy(),
            overdraft_liquidation_total=overdraft,
            reserved_liquidation_total=reserved.copy(),
            liquidation_sale_cost=cost_paid.copy(),
            purchase_liquidation_total=purchase.copy(),
            liquidation_reservation_ifa=available.copy(),
            residual_shortfall_after_granted_credit=np.maximum(residual - granted - cash - surplus, 0.0).copy(),
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
        # An opening overdraft is a use to be repaid, not zero cash: residual = max(uses - income - (LFA - D), 0).
        overdraft = np.maximum(-arrays["available_lfa"], 0.0)
        need = np.maximum(arrays["cash_uses"] + overdraft - arrays["cash_income"], 0.0)
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
        support_exempt_resources: np.ndarray | None = None,
        support_enabled: bool = True,
        scheduled_overdraft_interest: np.ndarray | None = None,
        opening_overdraft: np.ndarray | None = None,
    ) -> PostGrantFeasiblePlan:
        """Allocate resources in the Stage 5 payment order, without writing household state.

        All inputs are nominal cash amounts per household. ``resources`` (A) is
        income without support plus liquid assets, granted consumer credit and
        the reserved liquidation net of its cost. The order is: goods up to the
        floor F and cash rent R; debt service O; the opening overdraft B;
        goods above the floor up to desired goods G; investment V.

            S   = max(F + R - A, 0)             A1 = A + S - F - R
            P   = min(O, A1)                    U  = O - P
            C^f = F + min(max(G - F, 0), A1 - P)
            V^p = min(V, A1 - P - (C^f - F))

        Support S covers the goods floor and rent only. Unpaid service U is
        suspended on consumer credit first, then on the mortgage.

        O includes the overdraft interest due on the opening negative liquid
        balance (``scheduled_overdraft_interest``, default zero). It is paid
        after consumer and mortgage service, so their suspension is unchanged.
        The unpaid part ``unpaid_overdraft_interest`` is not charged to the
        account: the caller books it as consumer interest arrears at the
        household's bank, the unpaid-service path of consumer credit.

        The opening overdraft B = max(D - LFA, 0) (``opening_overdraft``,
        default zero) is a senior use: A counts liquid assets as
        max(LFA - D, 0), and B is repaid right after debt service, before goods
        above the floor and investment,

            B^p = min(B, A1 - P)                C^f = F + min(max(G - F, 0), A1 - P - B^p)
            V^p = min(V, A1 - P - B^p - (C^f - F))

        so the household's own resources are A - B = income + (LFA - D) + K +
        liquidation: LFA - D without the zero floor. Support never covers B: it is set from
        A, which excludes it. The part B - B^p stays overdrawn.

        ``support_exempt_resources`` (E, default zero) is the part of A that the
        support test does not count: S = max(F + R - (A - E), 0). The household
        still spends E, so A1 = A + S - F - R is unchanged in form. With support,
        resources A < 0 are covered too: S = F + R - (A - E) > F + R.

        ``support_enabled=False`` (minimum-income mode, PLAN.md section 5.4): no
        support, and rent is paid before the floor from the non-negative part of
        the resources A+ = max(A, 0). A negative A is not covered by anyone here:
        it is recorded as ``resource_deficit`` = max(-A, 0) and stays in the
        household's realised cash ledger. E plays no role.

            S   = 0
            Rp  = min(R, A+)                    unpaid_rent = R - Rp
            Fp  = min(F, A+ - Rp)               unmet_floor = F - Fp
            A1  = A+ - Rp - Fp
            C^f = Fp + min(max(G - F, 0), A1 - P), P and V^p as above

        An unmet floor leaves A1 = 0, so nothing is bought above it. With the
        goods floor switched off the caller passes F = 0 and C^f <= G.

        The whole allocation is made from the rounded-down resources
        A - m, m = ``PLAN_ROUND_DOWN`` x (|A| + F + R), so that a plan spending
        all resources closes at LFA >= m rather than at a floating-point residue
        below zero. Support absorbs m (S + A - m still covers F + R exactly);
        unpaid amounts change by at most m. ``resource_deficit`` is max(-A, 0) of
        the unrounded A, and a cut no larger than m does not make the floor binding.
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
        overdraft_interest = (
            np.zeros(shape)
            if scheduled_overdraft_interest is None
            else self._non_negative(scheduled_overdraft_interest)
        )
        overdraft = np.zeros(shape) if opening_overdraft is None else self._non_negative(opening_overdraft)
        for values in (
            desired,
            floor,
            rent,
            consumer_service,
            mortgage_service,
            desired_investment,
            available,
            overdraft_interest,
            overdraft,
        ):
            if values.shape != shape:
                raise ValueError("Stage 5 consumption-floor inputs must be equal-length household vectors.")
        if not np.all(np.isfinite(available)):
            raise ValueError("Stage 5 consumption-floor resources must be finite.")
        exempt = (
            np.zeros(shape) if support_exempt_resources is None else np.asarray(support_exempt_resources, dtype=float)
        )
        if exempt.shape != shape or not np.all(np.isfinite(exempt)) or np.any(exempt < 0.0):
            raise ValueError("Support-exempt resources must be finite, non-negative household values.")
        if not isinstance(support_enabled, (bool, np.bool_)):
            raise TypeError("support_enabled must be a boolean switch, not a household vector.")

        margin = PLAN_ROUND_DOWN * (np.abs(available) + floor + rent)
        rounded = available - margin
        if support_enabled:
            support = np.maximum(floor + rent - (rounded - exempt), 0.0)
            after_floor_and_rent = np.maximum(rounded + support - floor - rent, 0.0)
            floor_paid = floor
            unpaid_rent = np.zeros(shape)
            unmet_floor = np.zeros(shape)
            resource_deficit = np.zeros(shape)
        else:
            support = np.zeros(shape)
            own_resources = np.maximum(rounded, 0.0)
            resource_deficit = np.maximum(-available, 0.0)
            rent_paid = np.minimum(rent, own_resources)
            floor_paid = np.minimum(floor, own_resources - rent_paid)
            unpaid_rent = rent - rent_paid
            unmet_floor = floor - floor_paid
            after_floor_and_rent = own_resources - rent_paid - floor_paid
        service = consumer_service + mortgage_service
        service_paid = np.minimum(service, after_floor_and_rent)
        unpaid_service = service - service_paid
        unpaid_consumer = np.minimum(unpaid_service, consumer_service)
        unpaid_mortgage = unpaid_service - unpaid_consumer
        overdraft_interest_paid = np.minimum(overdraft_interest, after_floor_and_rent - service_paid)
        unpaid_overdraft_interest = overdraft_interest - overdraft_interest_paid
        after_service = after_floor_and_rent - service_paid - overdraft_interest_paid
        overdraft_repayment = np.minimum(overdraft, after_service)
        after_service = after_service - overdraft_repayment
        consumption_above_floor = np.minimum(np.maximum(desired - floor, 0.0), after_service)
        consumption_after_floor = floor_paid + consumption_above_floor
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
            # A cut of the order of the round-down margin (2m allows for rounding) is not a binding floor.
            floor_binding=(consumption_cut_amount + floor_top_up) > 2.0 * margin,
            unpaid_rent=unpaid_rent,
            unmet_floor=unmet_floor,
            resource_deficit=resource_deficit,
            scheduled_overdraft_interest=overdraft_interest,
            unpaid_overdraft_interest=unpaid_overdraft_interest,
            overdraft_repayment=overdraft_repayment,
        )
