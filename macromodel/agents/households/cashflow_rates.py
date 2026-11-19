"""Contractual payment inputs and legacy rate diagnostics for consumption."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DebtServicePayments:
    """Current/previous nominal period payments in identical monetary units.

    Income is intentionally not part of the payment record: the consumption
    evaluator owns the common behavioural-income denominator.
    """

    current_nominal: np.ndarray
    previous_nominal: np.ndarray

    def __post_init__(self):
        for name in ("current_nominal", "previous_nominal"):
            values = np.array(getattr(self, name), dtype=float, copy=True)
            if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError("Debt-service payments must be finite nonnegative household vectors.")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if self.current_nominal.shape != self.previous_nominal.shape:
            raise ValueError("Debt-service payments must have the same household shape.")

    def change_ratio(self, income_nominal):
        """Signed payment change divided by the caller's shared income base."""
        income = np.asarray(income_nominal, dtype=float)
        if income.shape != self.current_nominal.shape or not np.all(np.isfinite(income)) or np.any(income <= 0):
            raise ValueError("Debt-service income must be a matching finite positive household vector.")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            ratio = (self.current_nominal - self.previous_nominal) / income
        if not np.all(np.isfinite(ratio)):
            raise ValueError("The debt-service payment change/income ratio must be finite.")
        return ratio


@dataclass(frozen=True)
class DebtServiceRateChanges:
    """Changes in annual-simple nominal applied rates, in decimal units."""

    consumer: np.ndarray
    mortgage: np.ndarray

    def __post_init__(self):
        for name in ("consumer", "mortgage"):
            values = np.array(getattr(self, name), dtype=float, copy=True)
            if values.ndim != 1 or not np.all(np.isfinite(values)):
                raise ValueError("Debt-service rate changes must be finite household vectors.")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if self.consumer.shape != self.mortgage.shape:
            raise ValueError("Consumer and mortgage rate changes must have the same shape.")


def applied_debt_service_rate_changes(
    consumer_period_rate,
    mortgage_period_rate,
    consumer_principal,
    mortgage_principal,
    *,
    reference=None,
    annualization=4.0,
):
    """Compare applied stock rates, without an ordinary-refinancing event gate.

    Annual-simple rates match interest charged as period rate times principal;
    this is neither Fisher conversion nor compounding. `reference` contains a
    preceding planning observation. Startup assumes unchanged initial rates.
    A first debt observation has no old contract to reprice and contributes zero.
    Product averages can move with loan composition as well as contract resets.
    """
    if not np.isfinite(annualization) or annualization <= 0:
        raise ValueError("Rate annualization must be finite and positive.")
    inputs = [
        np.asarray(x, dtype=float)
        for x in (
            consumer_period_rate,
            mortgage_period_rate,
            consumer_principal,
            mortgage_principal,
        )
    ]
    if any(x.ndim != 1 or x.shape != inputs[0].shape or not np.all(np.isfinite(x)) or np.any(x < 0) for x in inputs):
        raise ValueError("Applied rates and principals must be matching finite nonnegative household vectors.")
    record = dict(
        zip(
            ("consumer_rate_annual", "mortgage_rate_annual", "consumer_principal", "mortgage_principal"),
            (annualization * inputs[0], annualization * inputs[1], inputs[2].copy(), inputs[3].copy()),
            strict=True,
        )
    )
    reference = record if reference is None else reference
    deltas = []
    for product in ("consumer", "mortgage"):
        rate_key, debt_key = product + "_rate_annual", product + "_principal"
        old_rate, old_debt = np.asarray(reference[rate_key]), np.asarray(reference[debt_key])
        if any(x.shape != inputs[0].shape or not np.all(np.isfinite(x)) or np.any(x < 0) for x in (old_rate, old_debt)):
            raise ValueError("Reference applied rates and principals must match the current household layout.")
        delta = np.where((old_debt > 0) & (record[debt_key] > 0), record[rate_key] - old_rate, 0.0)
        record[product + "_rate_change_annual"] = delta
        deltas.append(delta)
    return DebtServiceRateChanges(*deltas), record
