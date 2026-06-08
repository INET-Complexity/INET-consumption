"""Pre-hooks for household income MPC experiments.

The MPC experiment stages an equal nominal household income shock on the target
country. The country applies the staged amount to expected income during
planning and to realised income during accounting, then clears it at the end of
the iteration. This keeps the MPC denominator aligned with the state variable
that is shocked.
"""

import logging
from typing import Callable

import numpy as np

from macromodel.simulation import Simulation


def create_household_income_shock_hook(
    country_code: str,
    target_year: int,
    target_month: int,
    shock_fraction_of_median_income: float = 0.01,
) -> Callable[[Simulation, int, int], None]:
    """Create a one-time household income shock for MPC experiments.

    The hook stages an equal nominal addition to household income. The country
    applies it to expected income when forming plans and to realised household
    income when recording the period outcome.

    Parameters
    ----------
    country_code:
        Country whose household sector receives the staged shock.
    target_year, target_month:
        Calendar date passed by ``Simulation.run_prehooks``. The experiment
        runner converts zero-based simulation periods to this date.
    shock_fraction_of_median_income:
        Shock amount per household as a fraction of the current median positive
        household income. The shock is equal in nominal value for all households.
    """
    if shock_fraction_of_median_income <= 0:
        raise ValueError("shock_fraction_of_median_income must be strictly positive.")

    applied = [False]

    def household_income_shock_hook(simulation: Simulation, year: int, month: int) -> None:
        if applied[0]:
            return
        if year != target_year or month != target_month:
            return
        if country_code not in simulation.countries:
            raise ValueError(
                "Household income shock hook cannot find country "
                f"'{country_code}'. Available countries: {list(simulation.countries.keys())}."
            )

        country = simulation.countries[country_code]
        # Use realised household income from the current pre-shock state to set
        # the equal nominal shock size. The country applies the staged shock to
        # both expected income planning and realised income accounting.
        pre_shock_income = np.asarray(country.households.ts.current("income"), dtype=float)
        positive_income = pre_shock_income[np.isfinite(pre_shock_income) & (pre_shock_income > 0.0)]
        if positive_income.size == 0:
            raise ValueError("Cannot compute household income shock: no positive household incomes.")

        shock_amount = float(shock_fraction_of_median_income * np.median(positive_income))
        shock = np.full(pre_shock_income.shape, shock_amount, dtype=float)
        country.stage_household_income_shock(shock)
        applied[0] = True

        logging.info(
            "Household income shock staged for %s at %s-%s: amount=%s per household",
            country_code,
            year,
            month,
            shock_amount,
        )

    return household_income_shock_hook


create_household_expected_income_shock_hook = create_household_income_shock_hook
