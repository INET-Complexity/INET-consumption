from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class TargetCreditSetter(ABC):
    """Abstract base class for determining firms' target credit requirements.

    This class defines strategies for calculating optimal credit demand based on:
    - Estimated deposits (available liquid funds)
    - Existing overdrafts that must be repaired/refinanced
    - Unconstrained costs for intermediate inputs (working capital needs)
    - Unconstrained costs for capital inputs (investment needs)

    The setter returns ordinary activity credit only:
    - short-term activity credit for intermediate inputs and direct TFP spending
    - long-term activity credit for capital inputs and technical investment

    `Firms.compute_target_credit` adds scheduled-principal rollover and
    existing-overdraft refinance buckets around this activity demand.
    """

    @abstractmethod
    def compute_target_credit(
        self,
        internal_cash: np.ndarray,
        existing_overdraft: np.ndarray,
        expected_sales: np.ndarray,
        hard_obligations: np.ndarray,
        unconstrained_target_intermediate_inputs_costs: np.ndarray,
        unconstrained_target_capital_inputs_costs: np.ndarray,
        planned_technical_investment_costs: np.ndarray,
        planned_tfp_investment_costs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate target short-term and long-term credit for each firm.

        Args:
            internal_cash (np.ndarray): Non-negative liquid funds available before new credit
            existing_overdraft (np.ndarray): Negative deposit balances that must be repaired/refinanced
            expected_sales (np.ndarray): Expected current-period sales proceeds
            hard_obligations (np.ndarray): Wages, taxes, interest, and scheduled debt service
            unconstrained_target_intermediate_inputs_costs (np.ndarray): Desired spending
                on intermediate inputs (materials, supplies) without financial constraints
            unconstrained_target_capital_inputs_costs (np.ndarray): Desired spending
                on capital inputs (machinery, equipment) without financial constraints
            planned_technical_investment_costs (np.ndarray): Desired technical-investment goods budget
            planned_tfp_investment_costs (np.ndarray): Desired direct-TFP cash spending

        Returns:
            Tuple[np.ndarray, np.ndarray]: Ordinary activity credit amounts.
                First array is short-term activity credit for intermediate inputs
                and direct TFP spending. Second array is long-term activity credit
                for capital inputs and technical investment.
        """
        pass


class DefaultTargetCreditSetter(TargetCreditSetter):
    """Default implementation of credit demand calculation.

    This class implements a pro-rata activity credit demand strategy that:
    1. Applies internal cash plus expected sales to hard obligations.
    2. Treats existing negative deposits as overdraft liabilities that must be
       repaired/refinanced before new spending.
    3. Allocates remaining internal funds pro-rata across planned activity.
    4. Requests activity credit for the uncovered category costs.
    """

    def compute_target_credit(
        self,
        internal_cash: np.ndarray,
        existing_overdraft: np.ndarray,
        expected_sales: np.ndarray,
        hard_obligations: np.ndarray,
        unconstrained_target_intermediate_inputs_costs: np.ndarray,
        unconstrained_target_capital_inputs_costs: np.ndarray,
        planned_technical_investment_costs: np.ndarray,
        planned_tfp_investment_costs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate credit demand using the default pro-rata activity strategy.

        The method follows these steps:
        1. Apply internal cash plus expected sales to hard obligations.
        2. Reserve enough liquidity to repair existing overdrafts.
        3. Allocate remaining finance pro-rata across planned activity.
        4. Request ST/LT credit for uncovered activity costs.

        Args:
            internal_cash (np.ndarray): Non-negative liquid funds available before new credit
            existing_overdraft (np.ndarray): Negative deposit balances that must be repaired/refinanced
            expected_sales (np.ndarray): Expected current-period sales proceeds
            hard_obligations (np.ndarray): Wages, taxes, interest, and scheduled debt service
            unconstrained_target_intermediate_inputs_costs (np.ndarray): Desired intermediate
                input spending without financial constraints
            unconstrained_target_capital_inputs_costs (np.ndarray): Desired capital
                input spending without financial constraints
            planned_technical_investment_costs (np.ndarray): Desired technical-investment goods budget
            planned_tfp_investment_costs (np.ndarray): Desired direct-TFP cash spending

        Returns:
            Tuple[np.ndarray, np.ndarray]: Ordinary activity credit amounts.
                First array is short-term activity credit for intermediate inputs
                and direct TFP spending. Second array is long-term activity credit
                for capital inputs and technical investment.
        """
        # Expected sales remain a planning preview, but the current activity budget
        # should only use cash that is actually available before the goods market.
        available_for_activity = np.maximum(0.0, internal_cash - hard_obligations - existing_overdraft)
        total_activity_budget = (
            unconstrained_target_intermediate_inputs_costs
            + planned_tfp_investment_costs
            + unconstrained_target_capital_inputs_costs
            + planned_technical_investment_costs
        )
        covered_activity_budget = np.minimum(available_for_activity, total_activity_budget)
        activity_scale = np.divide(
            covered_activity_budget,
            total_activity_budget,
            out=np.ones_like(total_activity_budget),
            where=total_activity_budget > 0.0,
        )

        uncovered_intermediate_costs = unconstrained_target_intermediate_inputs_costs * (1.0 - activity_scale)
        uncovered_tfp_costs = planned_tfp_investment_costs * (1.0 - activity_scale)
        uncovered_capital_costs = unconstrained_target_capital_inputs_costs * (1.0 - activity_scale)
        uncovered_technical_costs = planned_technical_investment_costs * (1.0 - activity_scale)
        target_short_term_credit = uncovered_intermediate_costs + uncovered_tfp_costs
        target_long_term_credit = uncovered_capital_costs + uncovered_technical_costs
        return target_short_term_credit, target_long_term_credit


class SimpleTargetCreditSetter(TargetCreditSetter):
    """Simplified implementation of credit demand calculation.

    This class implements a basic credit demand strategy where:
    - Remaining activity finance is allocated pro-rata after hard obligations
      and existing overdrafts
    - Short-term and long-term credit are requested for uncovered activity costs

    This approach is useful for:
    - Model testing and validation
    - Scenarios where firms rely primarily on equity financing
    - Simplified economic models without complex credit markets
    """

    def compute_target_credit(
        self,
        internal_cash: np.ndarray,
        existing_overdraft: np.ndarray,
        expected_sales: np.ndarray,
        hard_obligations: np.ndarray,
        unconstrained_target_intermediate_inputs_costs: np.ndarray,
        unconstrained_target_capital_inputs_costs: np.ndarray,
        planned_technical_investment_costs: np.ndarray,
        planned_tfp_investment_costs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate credit demand using the simplified strategy.

        Requests activity credit after hard obligations and existing overdrafts.

        Args:
            internal_cash (np.ndarray): Non-negative liquid funds available before new credit
            existing_overdraft (np.ndarray): Negative deposit balances that must be repaired/refinanced
            expected_sales (np.ndarray): Expected current-period sales proceeds
            hard_obligations (np.ndarray): Wages, taxes, interest, and scheduled debt service
            unconstrained_target_intermediate_inputs_costs (np.ndarray): Desired intermediate
                input spending
            unconstrained_target_capital_inputs_costs (np.ndarray): Desired capital
                input spending
            planned_technical_investment_costs (np.ndarray): Desired technical-investment goods budget
            planned_tfp_investment_costs (np.ndarray): Desired direct-TFP cash spending

        Returns:
            Tuple[np.ndarray, np.ndarray]: Ordinary activity credit amounts.
                First array is short-term activity credit for intermediate inputs
                and direct TFP spending. Second array is long-term activity credit
                for capital inputs and technical investment.
        """
        # Expected sales are not treated as spendable finance for the current
        # activity budget because they arrive after the goods market clears.
        available_for_activity = np.maximum(0.0, internal_cash - hard_obligations - existing_overdraft)
        total_activity_budget = (
            unconstrained_target_intermediate_inputs_costs
            + planned_tfp_investment_costs
            + unconstrained_target_capital_inputs_costs
            + planned_technical_investment_costs
        )
        covered_activity_budget = np.minimum(available_for_activity, total_activity_budget)
        activity_scale = np.divide(
            covered_activity_budget,
            total_activity_budget,
            out=np.ones_like(total_activity_budget),
            where=total_activity_budget > 0.0,
        )
        target_short_term_credit = (unconstrained_target_intermediate_inputs_costs + planned_tfp_investment_costs) * (
            1.0 - activity_scale
        )
        target_long_term_credit = (unconstrained_target_capital_inputs_costs + planned_technical_investment_costs) * (
            1.0 - activity_scale
        )
        return target_short_term_credit, target_long_term_credit
