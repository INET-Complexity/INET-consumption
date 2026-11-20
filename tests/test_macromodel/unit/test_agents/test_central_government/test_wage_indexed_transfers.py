"""Contract-wage indexation of pensions and other transfers (regular_transfer_indexation)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.agents.central_government.central_government import CentralGovernment
from macromodel.agents.central_government.func.social_benefits import (
    ConstantSocialBenefitsSetter,
    DefaultSocialBenefitsSetter,
    GrowthSocialBenefitsSetter,
    ReplacementRateSocialBenefitsSetter,
)
from macromodel.agents.individuals.individual_properties import ActivityStatus
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


@pytest.mark.parametrize("setter_cls", [GrowthSocialBenefitsSetter, DefaultSocialBenefitsSetter])
def test_every_setter_accepts_indexation(setter_cls):
    assert setter_cls(regular_transfer_indexation="average_wage").regular_transfer_indexation == "average_wage"
    assert setter_cls().regular_transfer_indexation == "none"


E, U = ActivityStatus.EMPLOYED, ActivityStatus.UNEMPLOYED


def make_country(status, firm, wage_rate, labour, n_employees_t0):
    country = Country.__new__(Country)
    country.individuals = SimpleNamespace(
        states={
            "Activity Status": np.array(status),
            "Corresponding Firm ID": np.array(firm),
            "Wage Rate": np.array(wage_rate, float),
        },
        ts=TimeSeries(labour_inputs=np.array(labour, float)),
    )
    country.firms = SimpleNamespace(ts=TimeSeries(number_of_employees=np.array(n_employees_t0, float)))
    return country


def test_contract_index_is_one_at_first_call_and_uses_rate_per_labour_input():
    # Firm 0: rates per labour input 2 (worker 0: 4/2) and 2 (worker 1: 2/1) -> 2. Firm 1: 6/2 = 3.
    country = make_country([E, E, E], [0, 0, 1], [4.0, 2.0, 6.0], [2.0, 1.0, 2.0], [2, 1])
    assert country._contract_wage_index_ratio() == pytest.approx(1.0)
    country.individuals.states["Wage Rate"] = np.array([4.4, 2.2, 6.0])  # firm 0 rates +10%
    # Fixed t0 weights (2, 1): (2*2.2 + 1*3) / (2*2 + 1*3) = 7.4 / 7.
    assert country._contract_wage_index_ratio() == pytest.approx(7.4 / 7.0)


def test_contract_index_ignores_who_is_unemployed():
    country = make_country([E, E, E], [0, 0, 1], [4.0, 2.0, 6.0], [2.0, 1.0, 2.0], [2, 1])
    country._contract_wage_index_ratio()
    # A worker of firm 0 loses the job; no rate changes. Firm 0's rate per labour input stays 2.
    country.individuals.states["Activity Status"] = np.array([E, U, E])
    assert country._contract_wage_index_ratio() == pytest.approx(1.0)


def test_contract_index_drops_firms_without_employees():
    country = make_country([E, E], [0, 1], [2.0, 3.0], [1.0, 1.0], [1, 1])
    country._contract_wage_index_ratio()
    country.individuals.states["Activity Status"] = np.array([U, E])
    country.individuals.states["Wage Rate"] = np.array([2.0, 3.3])
    assert country._contract_wage_index_ratio() == pytest.approx(1.1)


def test_contract_index_requires_employment():
    country = make_country([E], [0], [2.0], [1.0], [1])
    country._contract_wage_index_ratio()
    country.individuals.states["Activity Status"] = np.array([U])
    with pytest.raises(RuntimeError, match="Contract-wage transfer index"):
        country._contract_wage_index_ratio()


def test_index_not_computed_when_indexation_off(monkeypatch):
    # Fix for the review: the default path must not depend on the index (it may raise with no employment).
    calls = []
    monkeypatch.setattr(Country, "_contract_wage_index_ratio", lambda self: calls.append(1) or 1.0)
    government = make_government("none")
    assert not government.uses_wage_indexed_transfers()
    assert make_government("average_wage").uses_wage_indexed_transfers()
    import inspect

    source = inspect.getsource(Country._update_benefit_planning_metrics)
    assert "uses_wage_indexed_transfers()" in source and "else 1.0" in source
    assert calls == []


def test_preset_reaches_setter():
    """The preset key names the setter parameter (no notebook imports: CI lacks plotly/environs)."""
    import sys

    root = Path(__file__).resolve().parents[5]
    sys.path.insert(0, str(root / "run_model"))
    from config import SCENARIO_PRESETS

    from macromodel.configurations import load_country_configuration
    from macromodel.util.function_mapping import functions_from_model

    ((key, value),) = SCENARIO_PRESETS["wage_indexed_transfers"].items()
    assert key == "central_government.functions.social_benefits.parameters.['regular_transfer_indexation']"
    config = load_country_configuration(root / "run_model/config/country_config_FRA.yaml", country_iso3="FRA")
    config.central_government.functions.social_benefits.parameters["regular_transfer_indexation"] = value
    setter = functions_from_model(config.central_government.functions, loc="macromodel.agents.central_government")[
        "social_benefits"
    ]
    assert setter.regular_transfer_indexation == "average_wage"
