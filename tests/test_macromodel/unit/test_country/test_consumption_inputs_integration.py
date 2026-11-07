"""Live Country -> Households adapter dispatch, lifecycle and accounting contracts."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from macromodel.agents.households.func.consumption import CreditAugmentedConsumption
from macromodel.configurations import CountryConfiguration, SimulationConfiguration, load_country_configuration
from macromodel.simulation import Simulation


@pytest.fixture
def aligned_country(datawrapper, monkeypatch):
    root = Path(__file__).resolve().parents[4]
    production = load_country_configuration(root / "run_model/config/country_config_FRA.yaml", country_iso3="FRA")
    config = CountryConfiguration()
    config.households = production.households
    config.households.functions.wealth.parameters["frm_coefficients_path"] = str(
        root / "run_model/data/raw_data/portfolio/FR_portfolio_frm_coefficients.json"
    )
    config.banks.parameters.household_consumption_loan_maturity = (
        production.banks.parameters.household_consumption_loan_maturity
    )
    config.banks.parameters.requires_household_underwriting_snapshot = True
    config.consumer_credit = production.consumer_credit
    config.credit_market = production.credit_market
    simulation = Simulation.from_datawrapper(
        datawrapper=datawrapper,
        simulation_configuration=SimulationConfiguration(country_configurations={"FRA": config}, seed=15),
    )
    country = simulation.countries["FRA"]
    monkeypatch.setattr(country.economy, "current_expected_consumer_period_inflation", lambda: 0.0)
    return country, config


def test_both_live_passes_share_opening_stocks_recompute_income_and_preserve_accounting(aligned_country, monkeypatch):
    country, _ = aligned_country
    hh = country.households
    rule = hh.functions["consumption"]
    assert isinstance(rule, CreditAugmentedConsumption)
    assert rule.uses_aligned_behavioural_income and rule.uses_income_belief_learning
    fields = (
        "liquid_financial_assets",
        "illiquid_financial_assets",
        "wealth_main_residence",
        "mortgage_debt",
        "consumption_loan_debt",
    )
    opening = {key: hh.ts.current(key).copy() for key in fields}
    for key in fields:
        hh.ts.dicts[key].insert(0, np.full_like(opening[key], 7.0))  # distinguish t-2 from t-1
    producer = Mock(wraps=hh.current_income_belief_learning_inputs)
    monkeypatch.setattr(hh, "current_income_belief_learning_inputs", producer)
    country._set_household_target_demand(replace_current=False)
    snapshot = country._consumption_input_adapter.opening_snapshot
    epsilon = rule._household_epsilon.copy()
    first_b = hh.ts.current("target_consumption_income_base").copy()
    lengths = {key: len(hh.ts.historic(key)) for key in hh._target_consumption_diagnostic_keys()}
    for key in fields:
        hh.ts.override_current(key, np.maximum(opening[key], 0.0) + 100.0)  # intervening t values
    hh.ts.override_current("expected_income_employee", hh.ts.current("expected_income_employee") * 1.5)
    country._set_household_target_demand(replace_current=True)
    assert country._consumption_input_adapter.opening_snapshot is snapshot
    assert producer.call_count == 2
    np.testing.assert_array_equal(rule._household_epsilon, epsilon)
    assert np.any(hh.ts.current("target_consumption_income_base") != first_b)
    for key, size in lengths.items():
        assert len(hh.ts.historic(key)) == size, key
    for key, source in (
        ("opening_liquid_assets", "liquid_financial_assets"),
        ("opening_illiquid_assets", "illiquid_financial_assets"),
        ("opening_main_residence", "wealth_main_residence"),
        ("opening_mortgage_ledger_debt", "mortgage_debt"),
        ("opening_consumer_ledger_debt", "consumption_loan_debt"),
    ):
        np.testing.assert_allclose(
            hh.ts.current("target_consumption_" + key), opening[source], rtol=1e-10, atol=1e-12 * country.scale
        )
    b_hist = np.asarray(hh.ts.historic("expected_behavioural_income"))
    s_hist = np.asarray(country.economy.ts.historic("subsistence_consumption"))
    prices = np.r_[
        np.asarray(country.economy.ts.historic(country.economy.consumer_price_level_series_name())).reshape(-1),
        country.economy.current_consumer_price_level(),
    ]
    n = min(4, b_hist.shape[0])
    oracle = np.prod(
        np.maximum(b_hist[-n:], s_hist[-n:]) / (prices[-n:] / country.economy.initial_consumer_price_level())[:, None],
        axis=0,
    ) ** (1 / n)
    np.testing.assert_allclose(
        hh.ts.current("target_consumption_income_base"), oracle, rtol=1e-10, atol=1e-12 * country.scale
    )
    np.testing.assert_allclose(
        hh.ts.current("target_consumption_annual_income_base"), 4 * oracle, rtol=1e-10, atol=1e-12 * country.scale
    )
    components = rule.last_target_consumption_components
    market = np.maximum(
        0,
        components["target_consumption_calibrated_total"]
        - components["target_consumption_cash_rent"]
        - components["target_consumption_imputed_rent"],
    )
    vat = country.central_government.states["Value-added Tax"]
    np.testing.assert_allclose(
        hh.ts.current("target_consumption").sum(axis=1) * (1 + vat), market, rtol=1e-10, atol=1e-7
    )


def test_opening_principal_rate_weights_use_common_base_once(aligned_country, monkeypatch):
    country, _ = aligned_country
    h = country.households.ts.current("n_households")
    consumer, mortgage = np.full(h, 100.0), np.full(h, 200.0)
    monkeypatch.setattr(
        country.credit_market,
        "household_contractual_debt_rate_components",
        lambda **kwargs: (np.full(h, 0.01), consumer.copy(), np.full(h, 0.02), mortgage.copy()),
    )
    country._set_household_target_demand(replace_current=False)
    hh = country.households
    expected = (((1.01) ** 4 - 1) * consumer + ((1.02) ** 4 - 1) * mortgage) / hh.ts.current(
        "target_consumption_annual_income_base"
    )
    np.testing.assert_allclose(
        hh.ts.current("target_consumption_real_borrowing_rate"), expected, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_array_equal(hh.ts.current("target_consumption_opening_consumer_principal"), consumer)
    np.testing.assert_array_equal(hh.ts.current("target_consumption_opening_mortgage_principal"), mortgage)


@pytest.mark.parametrize("aligned,learning", [(False, False), (False, True), (True, False), (True, True)])
def test_all_existing_flag_combinations_dispatch_without_extra_producer_calls(
    aligned_country, monkeypatch, aligned, learning
):
    country, _ = aligned_country
    hh = country.households
    rule = hh.functions["consumption"]
    rule.uses_aligned_behavioural_income = aligned
    rule.uses_income_belief_learning = learning
    producer = Mock(wraps=hh.current_income_belief_learning_inputs)
    monkeypatch.setattr(hh, "current_income_belief_learning_inputs", producer)
    country._set_household_target_demand(replace_current=False)
    assert producer.call_count == int(learning)
    assert (country._consumption_input_adapter.opening_snapshot is not None) == (aligned and learning)


def test_reset_clears_snapshot_and_zero_growth_bypass_does_not_capture(aligned_country):
    country, config = aligned_country
    country._set_household_target_demand(replace_current=False)
    old = country._consumption_input_adapter.opening_snapshot
    initial_priors = deepcopy(country.households.states["income_belief_priors"])
    rng_before = np.random.get_state()
    cloned = deepcopy(country)
    rng_after = np.random.get_state()
    assert rng_before[0] == rng_after[0]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    assert rng_before[2:] == rng_after[2:]
    original_rng = country.households.functions["consumption"]._idiosyncratic_rng
    copied_rng = cloned.households.functions["consumption"]._idiosyncratic_rng
    original_rng_state = deepcopy(original_rng.bit_generator.state)
    assert copied_rng is not original_rng
    assert copied_rng.bit_generator.state == original_rng_state
    copied_rng.normal(size=10)
    assert original_rng.bit_generator.state == original_rng_state
    assert cloned._consumption_input_adapter is not country._consumption_input_adapter
    assert cloned._consumption_input_adapter.opening_snapshot is old  # Immutable record sharing is safe.
    cloned._consumption_input_adapter.reset()
    cloned.households.functions["consumption"]._household_epsilon[:] = 123.0
    assert country._consumption_input_adapter.opening_snapshot is old
    assert not np.shares_memory(
        cloned.households.functions["consumption"]._household_epsilon,
        country.households.functions["consumption"]._household_epsilon,
    )
    original_lfa = country.households.ts.current("liquid_financial_assets").copy()
    cloned.households.ts.current("liquid_financial_assets")[:] += 1
    np.testing.assert_array_equal(country.households.ts.current("liquid_financial_assets"), original_lfa)
    country.reset(config)
    assert country._consumption_input_adapter.opening_snapshot is None
    assert country.households.functions["consumption"]._household_epsilon is None
    for key, value in initial_priors.items():
        np.testing.assert_array_equal(country.households.states["income_belief_priors"][key], value)
    country.assume_zero_growth = True
    country._set_household_target_demand(replace_current=False)
    assert country._consumption_input_adapter.opening_snapshot is None
    country.assume_zero_growth = False
    # A skipped ordinary first pass must not manufacture a replacement snapshot.
    with pytest.raises(ValueError, match="captured opening"):
        country._set_household_target_demand(replace_current=True)
    country.reset(config)
    country._set_household_target_demand(replace_current=False)
    assert country._consumption_input_adapter.opening_snapshot.household_layout_key != old.household_layout_key


def test_duplicate_backward_and_same_size_layout_changes_fail(aligned_country):
    country, _ = aligned_country
    country._set_household_target_demand(replace_current=False)
    snapshot = country._consumption_input_adapter.opening_snapshot
    with pytest.raises(ValueError, match="duplicate/backward"):
        country._set_household_target_demand(replace_current=False)
    assert country._consumption_input_adapter.opening_snapshot is snapshot
    members = country.households.states["corr_individuals"]
    members[0], members[1] = members[1], members[0]
    with pytest.raises(ValueError, match="matching captured"):
        country._set_household_target_demand(replace_current=True)


def test_unit_mismatch_is_rejected_instead_of_inferred_from_values(aligned_country):
    country, _ = aligned_country
    country.scale *= 2
    with pytest.raises(ValueError, match="unit metadata"):
        country._set_household_target_demand(replace_current=False)


@pytest.mark.parametrize("name", ["income_override", "lagged_income_override"])
def test_integrated_ordinary_overrides_fail_explicitly(aligned_country, name):
    country, _ = aligned_country
    with pytest.raises(ValueError, match="rebuilt prepared histories"):
        country.households.compute_target_consumption(
            expected_inflation=0.0,
            current_cpi=country.economy.current_consumer_price_level(),
            initial_cpi=country.economy.initial_consumer_price_level(),
            exogenous_total_consumption=0.0,
            per_capita_unemployment_benefits=0.0,
            tau_vat=0.0,
            assume_zero_growth=False,
            **{name: np.ones(country.households.ts.current("n_households"))},
        )
