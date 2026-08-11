"""Pure Stage 5 household financial-feasibility plans and policy."""

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class PreGrantFeasiblePlan:
    """Provisional Stage 5 plan constructed before consumer-credit clearing."""

    liquidity_shortfall_before_repair: np.ndarray
    funded_from_liquid_assets: np.ndarray
    residual_shortfall_after_lfa: np.ndarray
    credit_requested: np.ndarray | None = None
    planned_liquidation_total: np.ndarray | None = None


@dataclass
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
        )
