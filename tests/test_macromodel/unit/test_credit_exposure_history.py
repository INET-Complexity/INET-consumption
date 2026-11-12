"""Regression checks for settlement exposure history across model periods."""

import numpy as np

from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.simulation import Simulation


def test_firm_credit_exposure_records_one_settled_value_per_period(datawrapper):
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()}, seed=12, t_max=3)
    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)
    firms = simulation.countries["FRA"].firms
    initial_exposure = np.asarray(firms.ts.current("total_credit_exposure")).copy()
    expected_history = [initial_exposure.item()]

    for period in range(1, 4):
        simulation.iterate()
        expected_history.append(float(firms.compute_total_credit_exposure().sum()))
        history = np.asarray(firms.ts.historic("total_credit_exposure")).reshape(-1)

        assert len(history) == period + 1
        np.testing.assert_allclose(history, expected_history)
        np.testing.assert_array_equal(firms.ts.initial("total_credit_exposure"), initial_exposure)

    # Ensure the regression checks genuine outstanding exposure, not only zeros.
    assert np.any(np.asarray(expected_history) > 0.0)
