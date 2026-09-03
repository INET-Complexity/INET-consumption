import numpy as np


class TestIndividuals:
    def test__individuals_states(self, test_individuals):
        assert test_individuals is not None
        for state in [
            "Gender",
            "Age",
            "Education",
            "Activity Status",
            "Employment Industry",
            "Income",
            "Employee Income",
            "Income from Unemployment Benefits",
            "Corresponding Household ID",
            "Corresponding Firm ID",
            "Wage Rate",
        ]:
            assert state in test_individuals.states.keys()

    def test__wage_rate_state_is_seeded_from_individual_employee_income(self, test_individuals):
        """The carried wage rate starts equal to the individual's own wage.

        It is a worker attribute, not a time series, and not a firm-level value,
        so the initial cross-worker dispersion in the data must survive into it.
        """
        wage_rate = test_individuals.states["Wage Rate"]
        employee_income = np.nan_to_num(test_individuals.states["Employee Income"].astype(float), nan=0.0)

        np.testing.assert_allclose(wage_rate, employee_income)
        assert np.all(np.isfinite(wage_rate))
        assert np.all(wage_rate >= 0.0)

    def test__wage_rate_state_is_writeable_and_independent(self, test_individuals):
        """Mutating the carried rate must not write through to Employee Income."""
        before = test_individuals.states["Employee Income"].copy()
        test_individuals.states["Wage Rate"][:] += 1.0

        np.testing.assert_allclose(test_individuals.states["Employee Income"], before)

    def test__wage_rate_state_is_not_a_time_series(self, test_individuals):
        """The authoritative carried rate lives on the worker, not in ts."""
        assert "wage_rate" not in test_individuals.ts.get_keys()

    def test__individuals_ts(self, test_individuals):
        for ts_key in [
            "n_individuals",
            "employee_income",
            "income_from_unemployment_benefits",
            "income",
            "labour_inputs",
            "reservation_wages",
        ]:
            assert ts_key in test_individuals.ts.get_keys()
