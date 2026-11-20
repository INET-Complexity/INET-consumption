"""Characterisation tests pinning the current unemployment-benefit and pension rules.

These tests document behaviour as of main 7573b7c3, before the replacement-rate
revision. They are not a statement that this behaviour is correct.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.agents.central_government.central_government import CentralGovernment
from macromodel.agents.central_government.func.social_benefits import ConstantSocialBenefitsSetter
from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.country.country import Country

E, U, N = ActivityStatus.EMPLOYED, ActivityStatus.UNEMPLOYED, ActivityStatus.NOT_ECONOMICALLY_ACTIVE


def make_government(ub_level=10.0, pension_budget=60.0):
    government = CentralGovernment.__new__(CentralGovernment)
    values = {"unemployment_benefits_by_individual": [ub_level], "public_pension_benefits": [pension_budget]}
    government.ts = SimpleNamespace(current=lambda key: values[key])
    return government


# 1. Constant setter ignores inflation, growth and unemployment.
@pytest.mark.parametrize(
    "inflation, growth, unemployment",
    [(np.array([0.0]), 0.0, 0.0), (np.array([0.05, 0.1]), 0.03, 0.25), (np.array([-0.02]), -0.05, 0.9)],
)
def test_constant_setter_returns_previous_values(inflation, growth, unemployment):
    setter = ConstantSocialBenefitsSetter()
    kwargs = dict(
        benefit_indexation_inflation=inflation,
        current_estimated_growth=growth,
        current_unemployment_rate=unemployment,
        model=None,
    )
    assert setter.compute_unemployment_benefits(prev_unemployment_benefits=7.5, **kwargs) == 7.5
    assert setter.compute_regular_transfer_to_households(prev_regular_transfer_to_households=3.25, **kwargs) == 3.25


# 2. Unemployment benefit distribution: level x Benefit Weight for the unemployed only.
def test_unemployment_benefits_weighted_by_benefit_weight():
    status = np.array([E, U, N, U])
    weights = np.array([2.0, 0.5, 3.0, 1.5])
    benefits = make_government(ub_level=10.0).distribute_unemployment_benefits_to_individuals(status, weights)
    np.testing.assert_allclose(benefits, [0.0, 5.0, 0.0, 15.0])
    assert benefits.dtype == float


def test_unemployment_benefits_flat_level_without_weights():
    status = np.array([E, U, N, U])
    benefits = make_government(ub_level=10.0).distribute_unemployment_benefits_to_individuals(status)
    np.testing.assert_allclose(benefits, [0.0, 10.0, 0.0, 10.0])


def test_unemployment_benefit_level_does_not_depend_on_past_wage():
    # The current rule has no wage input: identical weights give identical benefits.
    status = np.array([U, U])
    benefits = make_government(ub_level=4.0).distribute_unemployment_benefits_to_individuals(
        status, np.array([1.0, 1.0])
    )
    assert benefits[0] == benefits[1] == 4.0


# 3. Public pension distribution: budget x w / sum(w) over eligible.
def test_pension_budget_split_over_eligible_weights():
    eligibility = np.array([True, False, True, True])
    weights = np.array([1.0, 5.0, 2.0, -3.0])  # ineligible and negative weights get nothing
    pensions = make_government(pension_budget=60.0).distribute_public_pension_benefits_to_individuals(
        eligibility, weights
    )
    np.testing.assert_allclose(pensions, [20.0, 0.0, 40.0, 0.0])
    assert pensions.sum() == pytest.approx(60.0)


@pytest.mark.parametrize("budget", [0.0, -5.0])
def test_pension_nonpositive_budget_pays_zero(budget):
    pensions = make_government(pension_budget=budget).distribute_public_pension_benefits_to_individuals(
        np.array([True, True]), np.array([1.0, 1.0])
    )
    np.testing.assert_array_equal(pensions, [0.0, 0.0])


def test_pension_positive_budget_without_eligible_weight_raises():
    with pytest.raises(RuntimeError, match="no eligible weighted recipients"):
        make_government(pension_budget=1.0).distribute_public_pension_benefits_to_individuals(
            np.array([False, True]), np.array([1.0, 0.0])
        )


# 4. Country._unemployment_benefit_by_individual
def make_country(ub_level=10.0, weights=None, cpi=1.2, individual_benefits=None, households=None):
    country = Country.__new__(Country)
    country.central_government = SimpleNamespace(
        ts=SimpleNamespace(current=lambda key: {"unemployment_benefits_by_individual": [ub_level]}[key])
    )
    states = {} if weights is None else {"Benefit Weight": weights}
    if households is not None:
        states["Corresponding Household ID"] = households
    country.individuals = SimpleNamespace(
        states=states,
        ts=SimpleNamespace(current=lambda key: {"income_from_unemployment_benefits": individual_benefits}[key]),
    )
    country.economy = SimpleNamespace(current_consumer_price_level=lambda: cpi)
    n_households = 0 if households is None else int(households.max()) + 2  # one empty household
    country.households = SimpleNamespace(
        ts=SimpleNamespace(current=lambda key: {"n_households": n_households}[key]),
        aggregate_individual_amount=lambda individual_amount, corr_households: np.bincount(
            corr_households, weights=individual_amount, minlength=n_households
        ),
    )
    return country


def test_country_benefit_by_individual_without_weights_is_scalar_level():
    assert make_country(ub_level=10.0)._unemployment_benefit_by_individual() == 10.0


def test_country_benefit_by_individual_with_weights_for_everyone():
    # Returned for every person (employed included): the would-be benefit if unemployed.
    result = make_country(ub_level=10.0, weights=np.array([0.0, 0.5, 2.0]))._unemployment_benefit_by_individual()
    np.testing.assert_allclose(result, [0.0, 5.0, 20.0])


# 5. Household aggregation in current CPI.
def test_household_unemployment_benefits_are_cpi_times_household_sum():
    individual = np.array([0.0, 5.0, 15.0, 2.0])
    households = np.array([0, 0, 1, 1])
    country = make_country(cpi=1.2, individual_benefits=individual, households=households)
    result = country._household_unemployment_benefits_from_individuals()
    np.testing.assert_allclose(result, [6.0, 20.4, 0.0])


# 6. Fiscal identity: government UB spending (country.py, total_unemployment_benefits =
# CPI x sum of individual receipts) equals the sum of household UB receipts.
def test_government_ub_spending_equals_household_receipts():
    individual = np.array([3.0, 0.0, 7.0, 1.5, 4.0])
    households = np.array([2, 0, 0, 1, 2])
    country = make_country(cpi=1.37, individual_benefits=individual, households=households)
    household_receipts = country._household_unemployment_benefits_from_individuals().sum()
    government_spending = country.economy.current_consumer_price_level() * individual.sum()
    assert household_receipts == pytest.approx(government_spending)
