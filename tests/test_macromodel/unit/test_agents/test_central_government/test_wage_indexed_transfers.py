"""Average-wage indexation of pensions and other transfers (regular_transfer_indexation)."""

from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.agents.central_government.central_government import CentralGovernment
from macromodel.agents.central_government.func.social_benefits import (
    ConstantSocialBenefitsSetter,
    ReplacementRateSocialBenefitsSetter,
)
from macromodel.country.country import Country
from macromodel.timeseries import TimeSeries


def make_government(indexation="none", setter_cls=ConstantSocialBenefitsSetter, periods=1):
    government = CentralGovernment.__new__(CentralGovernment)
    government.ts = TimeSeries(
        unemployment_benefits_by_individual=[10.0],
        total_other_benefits=[20.0],
        public_pension_benefits=[60.0],
    )
    government.functions = {"social_benefits": setter_cls(regular_transfer_indexation=indexation)}
    government.states = {"unemployment_benefits_model": None, "other_benefits_model": None}
    for _ in range(periods - 1):
        update(government, 1.0)
    return government


def update(government, ratio):
    government.update_benefits(
        historic_benefit_indexation_inflation=[np.array([0.01])],
        exogenous_benefit_indexation_inflation=np.array([0.0]),
        current_estimated_benefit_indexation_inflation=0.01,
        current_unemployment_rate=0.1,
        current_estimated_growth=0.0,
        real_average_wage_ratio=ratio,
    )


def test_default_is_none_and_ignores_ratio():
    government = make_government()
    assert government.functions["social_benefits"].regular_transfer_indexation == "none"
    update(government, 1.3)
    assert government.ts.current("public_pension_benefits")[0] == 60.0
    assert government.ts.current("total_other_benefits")[0] == 20.0


@pytest.mark.parametrize("setter_cls", [ConstantSocialBenefitsSetter, ReplacementRateSocialBenefitsSetter])
def test_average_wage_scales_t0_level_by_ratio(setter_cls):
    government = make_government("average_wage", setter_cls)
    update(government, 1.25)
    assert government.ts.current("public_pension_benefits")[0] == pytest.approx(75.0)
    assert government.ts.current("total_other_benefits")[0] == pytest.approx(25.0)
    # Level form against t0: a later ratio does not compound on the previous quarter.
    update(government, 0.9)
    assert government.ts.current("public_pension_benefits")[0] == pytest.approx(54.0)
    assert government.ts.current("total_other_benefits")[0] == pytest.approx(18.0)


def test_average_wage_leaves_unemployment_benefit_level_unchanged():
    government = make_government("average_wage")
    update(government, 1.5)
    assert government.ts.current("unemployment_benefits_by_individual")[0] == 10.0


def test_invalid_indexation_rejected():
    with pytest.raises(ValueError, match="regular_transfer_indexation"):
        ConstantSocialBenefitsSetter(regular_transfer_indexation="gdp")


def make_country(initial_wpc, initial_n, current_wpc, current_n):
    country = Country.__new__(Country)
    country.firms = SimpleNamespace(
        ts=TimeSeries(
            real_wage_per_capita=[np.array(initial_wpc, float)], number_of_employees=[np.array(initial_n, float)]
        )
    )
    country.firms.ts.real_wage_per_capita.append(np.array(current_wpc, float))
    country.firms.ts.number_of_employees.append(np.array(current_n, float))
    return country


def test_ratio_is_employment_weighted_average_wage():
    # t0 average (2*10 + 4*30)/40 = 3.5; now (3*10 + 5*10)/20 = 4.0.
    country = make_country([2.0, 4.0], [10, 30], [3.0, 5.0], [10, 10])
    assert country._real_average_wage_ratio() == pytest.approx(4.0 / 3.5)


def test_ratio_ignores_firms_without_employees():
    country = make_country([2.0, 99.0], [10, 0], [3.0, 99.0], [10, 0])
    assert country._real_average_wage_ratio() == pytest.approx(1.5)


def test_ratio_requires_positive_employment():
    with pytest.raises(RuntimeError, match="Average-wage transfer index"):
        make_country([2.0], [10], [3.0], [0])._real_average_wage_ratio()
