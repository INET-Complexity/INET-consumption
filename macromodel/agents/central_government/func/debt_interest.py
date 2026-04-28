"""Public-debt interest-rate rules for the central government."""

import numpy as np


class CurrentPolicyRateDebtInterest:
    """Price the full public-debt stock at the current policy rate."""

    def __init__(self, **_: object):
        pass

    @staticmethod
    def compute_interest_rate(
        current_policy_rate: float,
        previous_debt_interest_rate: float,
        time_unit: int,
    ) -> float:
        return max(0.0, current_policy_rate)


class SmoothedPolicyRateDebtInterest:
    """Smooth public-debt repricing toward the current policy rate."""

    def __init__(self, average_maturity_years: float = 9.0, smoothing: float | None = None):
        if average_maturity_years <= 0.0:
            raise ValueError("average_maturity_years must be positive.")
        if smoothing is not None and not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1).")

        self.average_maturity_years = average_maturity_years
        self.smoothing = smoothing

    def _smoothing(self, time_unit: int) -> float:
        if self.smoothing is not None:
            return self.smoothing
        if time_unit <= 0:
            raise ValueError("time_unit must be positive.")
        return max(0.0, 1.0 - time_unit / (12.0 * self.average_maturity_years))

    def compute_interest_rate(
        self,
        current_policy_rate: float,
        previous_debt_interest_rate: float,
        time_unit: int,
    ) -> float:
        if np.isnan(previous_debt_interest_rate):
            return max(0.0, current_policy_rate)

        smoothing = self._smoothing(time_unit=time_unit)
        smoothed_rate = smoothing * previous_debt_interest_rate + (1.0 - smoothing) * current_policy_rate
        return max(0.0, smoothed_rate)
