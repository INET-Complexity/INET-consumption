"""Government-consumption forecasting diagnostics."""

from __future__ import annotations

import numpy as np

from macromodel.forecaster.forecaster import ManualAutoregForecaster


def estimate_government_consumption_ar(model, country_code: str) -> dict[str, float | int]:
    """Estimate the notebook's AR(1) diagnostic on historic real government consumption."""
    country = model.countries[country_code]
    government = country.government_entities
    exogenous_before = country.exogenous.national_accounts_before[
        "Real Government Consumption (Value)"
    ].values.flatten()
    historic_ppi = np.asarray(country.economy.ts.historic("ppi"), dtype=float).flatten()
    historic_consumption = np.concatenate(
        (
            exogenous_before[-country.forecasting_window :],
            np.asarray(government.ts.historic("total_consumption"), dtype=float).flatten() / historic_ppi,
        )
    )
    series = np.log(historic_consumption)
    series = series[np.isfinite(series)]
    if len(series) < 2:
        raise ValueError("At least two finite observations are required for the AR diagnostic.")
    estimate = ManualAutoregForecaster.rfvar3(series, np.ones((len(series), 1)))
    return {
        "phi": float(estimate["By"][0][0][0]),
        "intercept": float(np.asarray(estimate["Bx"]).reshape(-1)[0]),
        "residual_std": float(np.sqrt(np.var(estimate["u"]))),
        "n_observations": len(series) - 1,
    }
