"""Pure Stage 5 household financial-feasibility plans and policy."""

from dataclasses import dataclass, fields, replace

import numpy as np


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
    early_consumer_repayment_capacity: np.ndarray | None = None
    floor_binding: np.ndarray | None = None
    post_liquidation_lfa: np.ndarray | None = None
    post_liquidation_ifa: np.ndarray | None = None
    settled_liquidation_total: np.ndarray | None = None
    residual_shortfall_after_lfa: np.ndarray | None = None
    reserved_liquidation_total: np.ndarray | None = None
    liquidation_reservation_ifa: np.ndarray | None = None

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
    ) -> PostGrantFeasiblePlan:
        """Reserve the Stage 5 liquidation before floors and goods demand are fixed."""
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
        reserved = np.minimum(planned, available)
        return replace(
            plan,
            planned_liquidation_total=reserved.copy(),
            reserved_liquidation_total=reserved.copy(),
            liquidation_reservation_ifa=available.copy(),
            residual_shortfall_after_granted_credit=np.maximum(residual - granted - reserved, 0.0).copy(),
        )

    def settle_consumption_floor(
        self,
        plan: PostGrantFeasiblePlan,
        *,
        consumption_before_floor: np.ndarray,
        subsistence_floor: np.ndarray,
    ) -> PostGrantFeasiblePlan:
        """Apply the Stage 5 floor policy without writing household state."""
        consumption_before = np.asarray(consumption_before_floor, dtype=float)
        floor = np.asarray(subsistence_floor, dtype=float)
        residual_shortfall = np.asarray(plan.residual_shortfall_after_granted_credit, dtype=float)
        if not (consumption_before.shape == floor.shape == residual_shortfall.shape):
            raise ValueError("Stage 5 consumption-floor inputs must be equal-length household vectors.")

        cleaned_consumption_before = self._non_negative(consumption_before)
        cleaned_floor = self._non_negative(floor)
        residual_before_floor = self._non_negative(residual_shortfall)
        maximum_floor_cut = np.maximum(cleaned_consumption_before - cleaned_floor, 0.0)
        consumption_cut_amount = np.minimum(residual_before_floor, maximum_floor_cut)
        consumption_before_support = cleaned_consumption_before - consumption_cut_amount
        residual_after_floor_cut = residual_before_floor - consumption_cut_amount
        floor_top_up = np.maximum(cleaned_floor - consumption_before_support, 0.0)
        consumption_after_floor = consumption_before_support + floor_top_up
        remaining_subsistence_shortfall = floor_top_up + residual_after_floor_cut
        return replace(
            plan,
            consumption_before_floor=cleaned_consumption_before,
            residual_shortfall_before_floor=residual_before_floor,
            consumption_after_floor=consumption_after_floor,
            consumption_cut_amount=consumption_cut_amount,
            remaining_subsistence_shortfall=remaining_subsistence_shortfall,
            floor_binding=(consumption_cut_amount + floor_top_up) > 0.0,
        )
