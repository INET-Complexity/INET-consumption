import numpy as np

from macromodel.agents.individuals.func.reservation_wages import (
    DefaultReservationWageSetter,
)
from macromodel.agents.individuals.individual_properties import ActivityStatus


class TestReservationWageSetter:
    def test__compute_reservation_wages(self):
        assert np.allclose(
            DefaultReservationWageSetter(unemployed_reservation_wage_timespan=12)
            .compute_reservation_wages(
                historic_wages=np.array([[10.0, 5.0, 2.0]]),
                current_individuals_activity=np.array(
                    [
                        ActivityStatus.EMPLOYED,
                        ActivityStatus.EMPLOYED,
                        ActivityStatus.UNEMPLOYED,
                    ]
                ),
                current_unemployment_benefits_by_individual=2.0,
            )
            .astype(float),
            np.array([10, 5.0, 2.0]),
        )

    def test__benefit_per_person_is_read_for_the_unemployed_only(self):
        activity = np.array([ActivityStatus.EMPLOYED, ActivityStatus.UNEMPLOYED, ActivityStatus.UNEMPLOYED])
        wages = DefaultReservationWageSetter(unemployed_reservation_wage_timespan=2).compute_reservation_wages(
            historic_wages=np.array([[10.0, 8.0, 0.0], [10.0, 0.0, 0.0]]),
            current_individuals_activity=activity,
            current_unemployment_benefits_by_individual=np.array([99.0, 1.0, 3.0]),
        )
        assert np.allclose(wages.astype(float), [10.0, 4.0, 3.0])
