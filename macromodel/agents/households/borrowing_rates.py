"""Current borrowing-rate indicators applied to frozen, lagged debt exposure.

This is a behavioural consumption input, not a contractual payment calculation.
The household debt weights translate Chauvin--Muellbauer's aggregate regressor;
the consumption-income denominator and loan contracts are not modified here.
"""

import numpy as np

from macro_data.util.rates import annualized_fisher_real_rate
from macromodel.agents.households.consumption_inputs import ConsumptionInputs


def lagged_annual_income_nominal(prepared: ConsumptionInputs) -> np.ndarray:
    """Prior smoothed real income, annualised and valued at the prior CPI.

    Exclude the current planning observation. With no prior observation, use
    the sole initial observation as the startup reference. Each historical
    income/floor uses its own dated deflator before temporal smoothing.
    """
    history = prepared.histories
    prices = history.consumer_price_history
    if history.price_history_layout == "missing_current":
        prices = np.append(prices, history.current_consumer_price)
    stop = max(1, len(prices) - 1)
    start = max(0, stop - history.history_window)
    deflators = prices / history.initial_consumer_price
    observations = np.maximum(
        history.behavioural_income_nominal_history[start:stop],
        history.subsistence_nominal_history[start:stop],
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        prior_real = np.exp(np.mean(np.log(observations) - np.log(deflators[start:stop])[:, None], axis=0))
        nominal = (12.0 / history.months_per_period) * prior_real * deflators[stop - 1]
    if not np.all(np.isfinite(nominal)) or np.any(nominal <= 0):
        raise ValueError("Lagged annual borrowing-rate income must be finite and positive.")
    return nominal


def behavioural_borrowing_rate_inputs(
    prepared: ConsumptionInputs,
    household_bank_ids: np.ndarray,
    consumer_rates: np.ndarray,
    overdraft_rates: np.ndarray,
    mortgage_rates: np.ndarray,
    expected_period_inflation: float,
) -> dict[str, np.ndarray]:
    """Build the current rate index and auditable household-level operands.

    Rates are nominal per-period bank indicators. The consumer indicator is
    their equal-weight overdraft/consumer-loan mean, then converted to an
    annual effective real rate. DB exposure includes consumer principal and
    an opening overdraft (negative deposits); MR uses mortgage principal.
    No ledger or contractual schedule is written.
    """
    bank_rates = [np.asarray(x, dtype=float) for x in (consumer_rates, overdraft_rates, mortgage_rates)]
    if any(x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)) or np.any(x < 0) for x in bank_rates):
        raise ValueError("Current borrowing rates must be nonempty finite nonnegative bank vectors.")
    if any(x.shape != bank_rates[0].shape for x in bank_rates[1:]):
        raise ValueError("Current borrowing-rate products must have the same bank dimensions.")
    opening = prepared.opening_snapshot.nominal
    ids = np.asarray(household_bank_ids)
    if ids.shape != opening.consumer_principal.shape or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Borrowing-rate bank IDs must be an integer vector matching households.")
    overdraft = np.maximum(-opening.liquid_financial_assets, 0.0)
    consumer_exposure = opening.consumer_principal + overdraft
    active = (consumer_exposure > 0) | (opening.mortgage_principal > 0)
    if np.any(ids < -1) or np.any(ids >= bank_rates[0].size) or np.any((ids == -1) & active):
        raise ValueError("Every indebted household must have a valid borrowing-rate bank ID.")
    safe_ids = np.maximum(ids, 0)
    proper_consumer, overdraft_nominal, mortgage_nominal = [x[safe_ids] for x in bank_rates]
    consumer_nominal = 0.5 * (proper_consumer + overdraft_nominal)
    periods = 12.0 / prepared.histories.months_per_period
    consumer_real = annualized_fisher_real_rate(consumer_nominal, expected_period_inflation, periods)
    mortgage_real = annualized_fisher_real_rate(mortgage_nominal, expected_period_inflation, periods)
    income = lagged_annual_income_nominal(prepared)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        index = (consumer_real * consumer_exposure + mortgage_real * opening.mortgage_principal) / income
    if not np.all(np.isfinite(index)):
        raise ValueError("Behavioural borrowing-rate index must be finite.")
    return {
        "real_borrowing_rate": index,
        "consumer_rate_nominal": consumer_nominal,
        "proper_consumer_rate_nominal": proper_consumer,
        "overdraft_rate_nominal": overdraft_nominal,
        "mortgage_rate_nominal": mortgage_nominal,
        "consumer_rate_real": consumer_real,
        "mortgage_rate_real": mortgage_real,
        "consumer_principal": opening.consumer_principal,
        "overdraft_principal": overdraft,
        "consumer_exposure": consumer_exposure,
        "mortgage_principal": opening.mortgage_principal,
        "lagged_annual_income_nominal": income,
    }
