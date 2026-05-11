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

    The credit demand is split into:
    - Short-term credit: total short-term demand, including scheduled-principal
      rollover, existing-overdraft repair that cannot be covered by
      post-obligation cash, and ordinary working-capital needs
    - Long-term credit: primarily for capital investments

    The setter returns only total short-term and long-term targets. `Firms.compute_target_credit`
    records the accounting split between `target_debt_rollover_credit`,
    `target_overdraft_refinance_credit`, and `ordinary_target_short_term_credit`
    from the same pro forma cash waterfall.
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
            Tuple[np.ndarray, np.ndarray]: Target short-term and long-term credit amounts
                First array is total short-term credit, including principal
                rollover, overdraft repair, and ordinary working-capital credit.
                Second array is long-term credit for investments
        """
        pass


class DefaultTargetCreditSetter(TargetCreditSetter):
    """Default implementation of credit demand calculation.

    This class implements a hierarchical credit demand strategy that:
    1. Applies internal cash plus expected sales to hard obligations.
    2. Treats existing negative deposits as overdraft liabilities that must be
       repaired/refinanced before new spending.
    3. Requests short-term credit for the remaining working-capital budget.
    4. Applies remaining internal funds to capital and technical investment.
    5. Requests long-term credit for the remaining investment budget.

    This approach prioritizes working capital needs over investment financing,
    reflecting typical business financial management practices.
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
        """Calculate credit demand using the default hierarchical strategy.

        The method follows these steps:
        1. Apply internal cash plus expected sales to hard obligations.
        2. Request short-term credit for the remaining working-capital budget.
        3. Apply remaining internal funds to capital and technical investment.
        4. Request long-term credit for the remaining investment budget.

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
            Tuple[np.ndarray, np.ndarray]: Target short-term and long-term credit amounts
                First array is total short-term credit, including overdraft repair
                and ordinary working-capital credit.
                Second array is long-term credit for investments
        """
        available_after_hard_obligations = internal_cash + expected_sales - hard_obligations - existing_overdraft
        working_capital_budget = (
            unconstrained_target_intermediate_inputs_costs + planned_tfp_investment_costs
        )
        investment_budget = unconstrained_target_capital_inputs_costs + planned_technical_investment_costs

        target_short_term_credit = np.maximum(0.0, working_capital_budget - available_after_hard_obligations)
        remaining_internal_finance = np.maximum(0.0, available_after_hard_obligations - working_capital_budget)
        target_long_term_credit = np.maximum(0.0, investment_budget - remaining_internal_finance)
        return target_short_term_credit, target_long_term_credit


class SimpleTargetCreditSetter(TargetCreditSetter):
    """Simplified implementation of credit demand calculation.

    This class implements a basic credit demand strategy where:
    - Short-term credit is requested for hard-obligation cash shortfalls after
      accounting for existing overdrafts
    - Long-term credit is requested for the remaining aggregate cash shortfall

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

        Requests short-term credit for the hard-obligation/overdraft cash shortfall
        and long-term credit for the remaining aggregate pro forma cash shortfall.

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
            Tuple[np.ndarray, np.ndarray]: Target short-term and long-term credit amounts
                First array is short-term overdraft-repair credit.
                Second array is the remaining aggregate pro forma shortfall.
        """
        available_after_hard_obligations = internal_cash + expected_sales - hard_obligations - existing_overdraft
        target_short_term_credit = np.maximum(0.0, -available_after_hard_obligations)
        remaining_internal_finance = np.maximum(0.0, available_after_hard_obligations)
        total_budget = (
            unconstrained_target_intermediate_inputs_costs
            + unconstrained_target_capital_inputs_costs
            + planned_technical_investment_costs
            + planned_tfp_investment_costs
        )
        target_long_term_credit = np.maximum(0.0, total_budget - remaining_internal_finance)
        return target_short_term_credit, target_long_term_credit
