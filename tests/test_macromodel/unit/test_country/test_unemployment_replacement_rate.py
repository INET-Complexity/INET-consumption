"""Replacement-rate unemployment benefit: rate derived from opening data, t0 unemployed keep their benefit,
separated workers receive rate x contract wage at job loss."""

from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.agents.central_government.func.social_benefits import (
    ConstantSocialBenefitsSetter,
    ReplacementRateSocialBenefitsSetter,
)
from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.configurations.central_government_configuration import CentralGovernmentConfiguration
from macromodel.country.country import Country
from macromodel.timeseries import TimeSeries

E, U, N = ActivityStatus.EMPLOYED, ActivityStatus.UNEMPLOYED, ActivityStatus.NOT_ECONOMICALLY_ACTIVE


def make_country(setter=None, with_weights=True):
    """Two employed (labour 1, 2; pay 10, 20 -> 10 per labour unit), two unemployed (labour 1.5, 0.5 ->
    predicted pay 15, 5; opening benefits 9, 3 -> rate 0.6), one inactive."""
    country = Country.__new__(Country)
    country.central_government = SimpleNamespace(
        functions={"social_benefits": setter or ReplacementRateSocialBenefitsSetter()},
        distribute_unemployment_benefits_to_individuals=lambda **kwargs: "flat-path",
    )
    states = {
        "Activity Status": np.array([E, E, U, U, N]),
        "Wage Rate": np.array([10.0, 20.0, 0.0, 0.0, 0.0]),
    }
    if with_weights:
        states["Benefit Weight"] = np.array([1.0, 2.0, 1.5, 0.5, 0.0])
    country.individuals = SimpleNamespace(
        states=states,
        ts=TimeSeries(
            labour_inputs=np.array([1.0, 2.0, 1.5, 0.5, 0.0]),
            employee_income=np.array([10.0, 20.0, 0.0, 0.0, 0.0]),
            income_from_unemployment_benefits=np.array([0.0, 0.0, 9.0, 3.0, 0.0]),
        ),
    )
    country._initialize_unemployment_benefit_base()
    return country


def test_rate_is_derived_from_opening_data():
    country = make_country()
    assert country._unemployment_benefit_rate == pytest.approx(0.6)


def test_t0_unemployed_keep_opening_benefit_and_employed_carry_rate_times_contract_wage():
    country = make_country()
    np.testing.assert_allclose(country._unemployment_benefit_base, [6.0, 12.0, 9.0, 3.0, 0.0])
    np.testing.assert_allclose(country._distribute_unemployment_benefits(), [0.0, 0.0, 9.0, 3.0, 0.0])


def test_separated_worker_receives_rate_times_last_contract_wage():
    country = make_country()
    country.individuals.states["Activity Status"][0] = U  # separated in clearing, before this quarter's wage update
    country.individuals.states["Wage Rate"][1] = 25.0  # incumbent indexed this quarter
    country._update_unemployment_benefit_base()
    paid = country._distribute_unemployment_benefits()
    assert paid[0] == pytest.approx(0.6 * 10.0)  # last contract wage, not updated after separation
    assert paid[1] == 0.0
    assert country._unemployment_benefit_base[1] == pytest.approx(0.6 * 25.0)
    np.testing.assert_allclose(paid[2:4], [9.0, 3.0])  # t0 unemployed unchanged


def test_rehired_worker_base_follows_new_contract_wage():
    country = make_country()
    country.individuals.states["Activity Status"][2] = E
    country.individuals.states["Wage Rate"][2] = 18.0
    country._update_unemployment_benefit_base()
    assert country._unemployment_benefit_base[2] == pytest.approx(0.6 * 18.0)
    assert country._distribute_unemployment_benefits()[2] == 0.0


def test_reservation_benefit_is_the_individual_base():
    country = make_country()
    out = country._unemployment_benefit_by_individual()
    np.testing.assert_allclose(out, country._unemployment_benefit_base)
    out[0] = -1.0
    assert country._unemployment_benefit_base[0] == pytest.approx(6.0)  # a copy


def test_constant_setter_keeps_the_flat_path():
    country = make_country(setter=ConstantSocialBenefitsSetter())
    assert country._unemployment_benefit_base is None
    assert country._distribute_unemployment_benefits() == "flat-path"


def test_replacement_rate_requires_worker_productivity_data():
    with pytest.raises(ValueError, match="Benefit Weight"):
        make_country(with_weights=False)


def test_configuration_accepts_the_setter_name():
    cfg = CentralGovernmentConfiguration.model_validate(
        {"functions": {"social_benefits": {"name": "ReplacementRateSocialBenefitsSetter"}}}
    )
    assert cfg.functions.social_benefits.name == "ReplacementRateSocialBenefitsSetter"
