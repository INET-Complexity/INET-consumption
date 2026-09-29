"""Increment 2: output identities at the existing allocation boundary."""

from copy import deepcopy

import h5py
import numpy as np
import pytest

from macromodel.agents.households.func.consumption import CreditAugmentedConsumption
from macromodel.agents.households.households_ts import realised_consumption_outcomes


def test_initial_outputs_follow_household_basis(test_households, datawrapper):
    ts = test_households.ts
    vat = datawrapper.synthetic_countries["FRA"].tax_data.value_added_tax
    expected_cash = (1 + vat) * ts.initial("consumption") + np.maximum(ts.initial("rent"), 0)
    np.testing.assert_allclose(ts.initial("consumption_cash_expenditure"), expected_cash)
    np.testing.assert_allclose(
        ts.initial("consumption_including_housing"), expected_cash + np.maximum(ts.initial("rent_imputed"), 0)
    )
    assert np.isnan(ts.initial("target_consumption_total_mpc")).all()
    assert np.isnan(ts.initial("target_consumption_calibrated_total")).all()


@pytest.mark.parametrize("cacf", [False, True])
def test_realised_outputs_tenure_changes_investment_and_hdf5(test_households, tmp_path, cacf):
    hh = test_households
    if cacf:
        hh.functions["consumption"] = CreditAugmentedConsumption()
    ts = hh.ts
    n = ts.current("n_households")
    j = hh.n_industries
    # Renter, owner, social renter, no flow, malformed negative observations.
    rents = np.resize([20.0, 0.0, 5.0, 0.0, -2.0], n)
    imputed = np.resize([0.0, 30.0, 0.0, 0.0, -3.0], n)
    expected_history = [ts.current("consumption_cash_expenditure").copy()]
    total_history = [ts.current("consumption_including_housing").copy()]
    state = {key: deepcopy(ts.current(key)) for key in ("income", "liquid_financial_assets", "debt")}
    for period in range(2):
        # Swap tenure in period two. Targets may be zero or rationed; no planned
        # housing diagnostic is used to construct these realised outcomes.
        cash_rent, housing = (rents, imputed) if period == 0 else (imputed, rents)
        ts.rent.append(cash_rent)
        ts.rent_imputed.append(housing)
        target = np.full((n, j), 10.0)
        target[::3] = 0.0
        spending = np.full((n, j), 14.0)
        spending[1::3] = 7.0
        ts.override_current("target_consumption", target)
        ts.override_current("nominal_amount_spent_in_lcu", spending)
        hh.update_consumption_and_investment(tau_vat=0.2, tau_cf=0.1, tau_vat_on_investment=0.05)
        goods = np.minimum(spending, target).sum(axis=1)
        cash = 1.2 * goods + np.maximum(cash_rent, 0)
        total = cash + np.maximum(housing, 0)
        expected_history.append(cash)
        total_history.append(total)
        np.testing.assert_allclose(ts.current("consumption"), goods)
        np.testing.assert_allclose(ts.current("consumption_cash_expenditure"), cash)
        np.testing.assert_allclose(ts.current("consumption_including_housing"), total)
        np.testing.assert_allclose(ts.current("investment"), spending - np.minimum(spending, target))
        np.testing.assert_allclose(ts.current("total_consumption"), [1.2 * goods.sum()])
        assert total.sum() == pytest.approx(cash.sum() + np.maximum(housing, 0).sum())
        for key, value in state.items():
            np.testing.assert_array_equal(ts.current(key), value)
    path = tmp_path / "outcomes.h5"
    with h5py.File(path, "w") as handle:
        ts.write_to_h5("households", handle.create_group("FRA"))
    with h5py.File(path, "r") as handle:
        group = handle["FRA/households"]
        assert group["consumption_cash_expenditure"].shape == (3, n)
        assert group["total_consumption"].shape == (3, 1)
        np.testing.assert_allclose(group["consumption_cash_expenditure"], expected_history)
        np.testing.assert_allclose(group["consumption_including_housing"], total_history)


@pytest.mark.parametrize("flow", ["rent", "rent_imputed"])
@pytest.mark.parametrize("bad", [np.array([np.nan, 0]), np.array([np.inf, 0]), np.zeros((2, 1)), np.zeros(1)])
def test_realised_housing_rejects_malformed_flows(flow, bad):
    kwargs = {"rent": np.zeros(2), "rent_imputed": np.zeros(2)}
    kwargs[flow] = bad
    with pytest.raises(ValueError, match=flow):
        realised_consumption_outcomes(np.ones(2), 0.2, **kwargs)


def test_non_cacf_target_unavailable_in_both_passes(test_households):
    ts = test_households.ts
    test_households._append_target_consumption_diagnostics(None)
    test_households._append_target_consumption_diagnostics(None, replace_current=True)
    for key in ("target_consumption_calibrated_total", "target_consumption_total_mpc"):
        assert len(ts.historic(key)) == 2
        assert np.isnan(ts.current(key)).all()
    np.testing.assert_array_equal(ts.current("formula_implied_mpc"), 0.0)


@pytest.mark.parametrize("vat", [0.0, 0.2])
def test_country_initial_vat_override_refreshes_outputs(datawrapper, vat):
    from macromodel.configurations import CountryConfiguration, ExchangeRatesConfiguration
    from macromodel.country import Country
    from macromodel.exchange_rates import ExchangeRates

    config = CountryConfiguration()
    config.central_government.tax_overrides.value_added_tax_rate = vat
    rates = ExchangeRates.from_data(
        exchange_rates_data=datawrapper.exchange_rates,
        exchange_rate_config=ExchangeRatesConfiguration(),
        initial_year=datawrapper.configuration.year,
        country_names=["FRA"],
    )
    country = Country.from_pickled_country(
        synthetic_country=datawrapper.synthetic_countries["FRA"],
        country_configuration=config,
        exchange_rates=rates,
        country_name="FRA",
        all_country_names=["FRA", "ROW"],
        industries=datawrapper.industries,
        initial_year=datawrapper.configuration.year,
        t_max=2,
        time_unit=datawrapper.time_unit,
        running_multiple_countries=False,
        emission_factors_usd=np.array([datawrapper.emission_factors[k] for k in ("coal", "gas", "oil")]),
    )
    ts = country.households.ts
    cash = (1 + vat) * ts.current("consumption") + np.maximum(ts.current("rent"), 0)
    np.testing.assert_allclose(ts.current("consumption_cash_expenditure"), cash)
    np.testing.assert_allclose(
        ts.current("consumption_including_housing"), cash + np.maximum(ts.current("rent_imputed"), 0)
    )
    assert len(ts.consumption_cash_expenditure) == len(ts.consumption) == 1
