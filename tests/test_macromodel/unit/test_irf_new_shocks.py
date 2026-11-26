from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.agents.central_government.central_government import CentralGovernment
from macromodel.utils.prehooks.irf_shocks import ShockSpec, create_irf_shock_hook


def _households_simulation(seed=3, n=8):
    households = SimpleNamespace(ts=SimpleNamespace(current=lambda _name: n), states={})
    return SimpleNamespace(countries={"FRA": SimpleNamespace(households=households)}, random_seed=seed)


def test_total_income_uncertainty_deterministic_scales_both_components_then_clears():
    spec = ShockSpec(name="total", kind="total_income_uncertainty", period=1, magnitude=0.25, mode="multiplicative")
    sim = _households_simulation()
    hook = create_irf_shock_hook(country_code="FRA", initial_year=2020, time_unit=3, spec=spec)

    hook(sim, 2020, 4)
    factors = sim.countries["FRA"].households.states["income_uncertainty_shock_factors"]
    assert factors["common"] == pytest.approx(1.25)
    np.testing.assert_allclose(factors["individual"], 1.25)

    hook(sim, 2020, 7)
    cleared = sim.countries["FRA"].households.states["income_uncertainty_shock_factors"]
    assert cleared["common"] == 1.0
    np.testing.assert_array_equal(cleared["individual"], np.ones(8))


def test_total_income_uncertainty_random_draw_is_common_to_both_components():
    spec = ShockSpec(
        name="total-random",
        kind="total_income_uncertainty",
        period=1,
        magnitude=0.25,
        mode="multiplicative",
        random_log_sd=0.1,
    )
    sim = _households_simulation()
    create_irf_shock_hook(country_code="FRA", initial_year=2020, time_unit=3, spec=spec)(sim, 2020, 4)
    factors = sim.countries["FRA"].households.states["income_uncertainty_shock_factors"]
    assert factors["common"] != 1.0
    np.testing.assert_allclose(factors["individual"], factors["common"])


class _Country:
    def __init__(self, incomes):
        self.households = SimpleNamespace(ts=SimpleNamespace(current=lambda _name: np.asarray(incomes, dtype=float)))
        self.staged = None

    def stage_household_income_shock(self, shock, *, exempt_from_support_test=False):
        self.staged = (np.asarray(shock), exempt_from_support_test)


def test_household_transfer_pays_once_at_shock_period_as_share_of_median_income():
    spec = ShockSpec(name="transfer", kind="household_transfer", period=2, magnitude=0.1, mode="multiplicative")
    country = _Country([100.0, 200.0, 300.0, -5.0])
    sim = SimpleNamespace(countries={"FRA": country})
    hook = create_irf_shock_hook(country_code="FRA", initial_year=2020, time_unit=3, spec=spec)

    hook(sim, 2020, 4)  # period 1: nothing
    assert country.staged is None
    hook(sim, 2020, 7)  # period 2
    shock, exempt = country.staged
    np.testing.assert_allclose(shock, 20.0)  # 10% of median positive income (200)
    assert exempt is True
    country.staged = None
    hook(sim, 2020, 7)  # one-off: not staged twice
    assert country.staged is None


@pytest.mark.parametrize(
    "overrides",
    [{"duration": 2}, {"mode": "additive"}, {"magnitude": -0.1}],
)
def test_household_transfer_rejects_invalid_specs(overrides):
    params = dict(name="transfer", kind="household_transfer", period=2, magnitude=0.1, mode="multiplicative")
    params.update(overrides)
    with pytest.raises(ValueError, match="Household-transfer"):
        create_irf_shock_hook(country_code="FRA", initial_year=2020, time_unit=3, spec=ShockSpec(**params))


def test_one_off_transfers_enter_government_deficit():
    values = {
        "total_unemployment_benefits": 1.0,
        "total_public_pension_benefits": 2.0,
        "total_other_social_transfers": 3.0,
        "total_necessity_support": 0.0,
        "total_minimum_income": 0.0,
        "total_one_off_transfers": 7.0,
        "revenue": 10.0,
    }
    government = SimpleNamespace(ts=SimpleNamespace(current=lambda name: [values[name]]))
    deficit = CentralGovernment.compute_deficit(
        government,
        current_ind_activity=None,
        current_cpi=1.0,
        current_government_nominal_amount_spent=np.array([4.0]),
        interest_payments_on_debt=0.5,
    )
    assert deficit[0] == pytest.approx(1 + 2 + 3 + 7 + 4 + 0.5 - 10)
