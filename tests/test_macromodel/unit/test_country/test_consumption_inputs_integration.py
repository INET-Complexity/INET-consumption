"""Live Country -> Households adapter dispatch, lifecycle and accounting contracts."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from macromodel.agents.households.func.consumption import CreditAugmentedConsumption
from macromodel.configurations import CountryConfiguration, SimulationConfiguration, load_country_configuration
from macromodel.simulation import Simulation


def _aligned_simulation(datawrapper, monkeypatch):
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
    return simulation, config


@pytest.fixture
def aligned_country(datawrapper, monkeypatch):
    simulation, config = _aligned_simulation(datawrapper, monkeypatch)
    return simulation.countries["FRA"], config


def test_deposit_repricing_does_not_raise_nonproperty_permanent_income(aligned_country, monkeypatch):
    """A quote change may alter current income, but not forecast earnings."""
    country, _ = aligned_country
    hh = country.households
    monkeypatch.setattr(country, "_common_permanent_income_terms", lambda: (0.03, 0.01))
    deposits = hh.ts.current("liquid_financial_assets")
    hh.ts.override_current("liquid_financial_assets", np.maximum(deposits, country.scale))
    country._set_household_target_demand(replace_current=False)
    before_pi = hh.ts.current("target_consumption_log_permanent_income_proxy_real").copy()
    before_base = hh.ts.current("target_consumption_income_base").copy()
    before_receipts = hh.ts.current("expected_behavioural_income_deposit_receipts").copy()
    before_cash_income = hh.ts.current("income").copy()
    opening = country._consumption_input_adapter.opening_snapshot

    quote = country.banks.ts.current("interest_rate_on_household_deposits")
    country.banks.ts.override_current("interest_rate_on_household_deposits", np.maximum(quote, 0.0) + 0.0025)
    country._set_household_target_demand(replace_current=True)

    assert np.all(hh.ts.current("expected_behavioural_income_deposit_receipts") > before_receipts)
    assert np.any(hh.ts.current("target_consumption_income_base") > before_base)
    np.testing.assert_array_equal(hh.ts.current("income"), before_cash_income)
    assert country._consumption_input_adapter.opening_snapshot is opening
    np.testing.assert_allclose(
        hh.ts.current("target_consumption_log_permanent_income_proxy_real"),
        before_pi,
        rtol=0.0,
        atol=1e-12,
    )


def test_current_borrowing_quotes_reduce_budget_without_repricing_contracts(aligned_country, monkeypatch):
    country, _ = aligned_country
    hh = country.households
    monkeypatch.setattr(country, "_common_permanent_income_terms", lambda: (0.03, 0.01))
    hh.ts.override_current("liquid_financial_assets", np.maximum(hh.ts.current("liquid_financial_assets"), 0))
    country._set_household_target_demand(replace_current=False)
    keys = (
        "target_consumption_income_base",
        "target_consumption_log_permanent_income_proxy_real",
        "target_consumption_borrowing_lagged_annual_income_nominal",
        "target_consumption_borrowing_consumer_rate_contractual_nominal",
        "target_consumption_borrowing_mortgage_rate_contractual_nominal",
        "expected_behavioural_income_borrower_interest",
    )
    unchanged = {key: hh.ts.current(key).copy() for key in keys}
    budget = hh.ts.current("cacf_real_consumption_budget").copy()
    rate = hh.ts.current("target_consumption_borrowing_real_borrowing_rate").copy()
    snapshot = country._consumption_input_adapter.opening_snapshot
    lengths = {
        key: len(values) for key, values in hh.ts.dicts.items() if key.startswith("target_consumption_borrowing_")
    }
    for key in (
        "interest_rates_on_household_consumption_loans",
        "overdraft_rate_on_household_deposits",
        "interest_rates_on_mortgages",
    ):
        country.banks.ts.override_current(key, country.banks.ts.current(key) + 0.0025)
    country._set_household_target_demand(replace_current=True)
    assert country._consumption_input_adapter.opening_snapshot is snapshot
    for key, values in unchanged.items():
        np.testing.assert_array_equal(hh.ts.current(key), values)
    assert lengths == {key: len(hh.ts.dicts[key]) for key in lengths}
    after_rate = hh.ts.current("target_consumption_borrowing_real_borrowing_rate")
    assert np.all(after_rate >= rate) and np.any(after_rate > rate)
    after_budget = hh.ts.current("cacf_real_consumption_budget")
    assert np.all(after_budget <= budget + 1e-8)
    assert np.any(after_budget < budget - 1e-8)


@pytest.mark.parametrize("component", ["income_rental", "expected_income_dividend_distributions"])
def test_property_receipts_do_not_change_permanent_income_or_uncertainty(aligned_country, monkeypatch, component):
    country, _ = aligned_country
    hh = country.households
    monkeypatch.setattr(country, "_common_permanent_income_terms", lambda: (0.03, 0.01))
    country._set_household_target_demand(replace_current=False)
    before_pi = hh.ts.current("target_consumption_log_permanent_income_proxy_real").copy()
    before_uncertainty = hh.ts.current("target_consumption_uncertainty").copy()
    before_base = hh.ts.current("target_consumption_income_base").copy()
    before_cash = hh.ts.current("income").copy()
    before_beliefs = deepcopy(hh.states["income_belief_runtime_state"])
    hh.ts.override_current(component, hh.ts.current(component) + country.scale)

    country._set_household_target_demand(replace_current=True)

    assert np.any(hh.ts.current("target_consumption_income_base") > before_base)
    np.testing.assert_array_equal(hh.ts.current("income_bridge_property"), np.zeros_like(before_pi))
    np.testing.assert_array_equal(hh.ts.current("target_consumption_log_permanent_income_proxy_real"), before_pi)
    np.testing.assert_array_equal(hh.ts.current("target_consumption_uncertainty"), before_uncertainty)
    np.testing.assert_array_equal(hh.ts.current("income"), before_cash)
    for key, value in before_beliefs.items():
        np.testing.assert_array_equal(hh.states["income_belief_runtime_state"][key], value)


@pytest.mark.parametrize("component", ["expected_income_employee", "expected_income_social_transfers"])
def test_nonproperty_income_increase_raises_permanent_income(aligned_country, monkeypatch, component):
    country, _ = aligned_country
    hh = country.households
    monkeypatch.setattr(country, "_common_permanent_income_terms", lambda: (0.03, 0.01))
    country._set_household_target_demand(replace_current=False)
    before_pi = hh.ts.current("target_consumption_log_permanent_income_proxy_real").copy()
    before_n = hh.ts.current("income_bridge_current").copy()
    hh.ts.override_current(component, hh.ts.current(component) + country.scale)

    country._set_household_target_demand(replace_current=True)

    assert np.any(hh.ts.current("income_bridge_current") > before_n)
    assert np.any(hh.ts.current("target_consumption_log_permanent_income_proxy_real") > before_pi)
    assert np.all(np.isfinite(hh.ts.current("target_consumption_log_permanent_income_proxy_real")))


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
    bank_count = country.banks.ts.current("n_banks")
    country.banks.ts.override_current("interest_rates_on_household_consumption_loans", np.full(bank_count, 0.01))
    country.banks.ts.override_current("overdraft_rate_on_household_deposits", np.full(bank_count, 0.03))
    country.banks.ts.override_current("interest_rates_on_mortgages", np.full(bank_count, 0.04))
    country.households.ts.override_current("liquid_financial_assets", np.zeros(h))
    prior_income = country.households.ts.current("expected_behavioural_income").copy()
    country._set_household_target_demand(replace_current=False)
    hh = country.households
    # Equal consumer/overdraft blend, plus current mortgage quotes. Opening
    # nominal principals and prior income use the same nominal price units.
    expected = (((1.02) ** 4 - 1) * consumer + ((1.04) ** 4 - 1) * mortgage) / (4 * prior_income)
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


def test_mortgage_payment_change_enters_cashflow_and_is_frozen_between_passes(aligned_country, monkeypatch):
    country, _ = aligned_country
    hh = country.households
    monkeypatch.setattr(country, "_common_permanent_income_terms", lambda: (0.03, 0.01))
    original = country.credit_market.preview_opening_contractual_household_service
    before_income = hh.ts.current("income").copy()
    before_expected_income = hh.ts.current("expected_income").copy()
    before_assets = hh.ts.current("liquid_financial_assets").copy()
    loan_book = {key: country.credit_market.states[key].copy() for key in ("cons_loans", "mort_loans")}
    extra = np.ones(hh.ts.current("n_households")) * country.scale

    def changed_schedule():
        components = original()
        components["mortgage_principal_nominal"] += extra
        return components

    monkeypatch.setattr(country.credit_market, "preview_opening_contractual_household_service", changed_schedule)
    country._set_household_target_demand(replace_current=False)
    prefix = "consumption_debt_service_"
    np.testing.assert_allclose(hh.ts.current(prefix + "change_nominal"), extra)
    expected = -0.75 * extra / hh.ts.current(prefix + "income_nominal")
    np.testing.assert_allclose(hh.ts.current("target_consumption_debt_service_cashflow"), expected)
    lengths = {key: len(v) for key, v in hh.ts.dicts.items() if key.startswith(prefix)}
    monkeypatch.setattr(
        country.credit_market,
        "preview_opening_contractual_household_service",
        lambda: (_ for _ in ()).throw(AssertionError("second pass must reuse its payment snapshot")),
    )
    country._set_household_target_demand(replace_current=True)
    np.testing.assert_allclose(hh.ts.current("target_consumption_debt_service_cashflow"), expected)
    assert lengths == {key: len(hh.ts.dicts[key]) for key in lengths}
    np.testing.assert_array_equal(hh.ts.current("income"), before_income)
    np.testing.assert_array_equal(hh.ts.current("expected_income"), before_expected_income)
    np.testing.assert_array_equal(hh.ts.current("liquid_financial_assets"), before_assets)
    for key, values in loan_book.items():
        np.testing.assert_array_equal(country.credit_market.states[key], values)


def test_quote_hike_does_not_reprice_fixed_contract_cashflow(aligned_country, monkeypatch):
    country, _ = aligned_country
    hh = country.households
    monkeypatch.setattr(country, "_common_permanent_income_terms", lambda: (0.03, 0.01))
    country._set_household_target_demand(replace_current=False)
    before = hh.ts.current("target_consumption_interest_rate_cashflow").copy()
    for key in ("interest_rates_on_household_consumption_loans", "interest_rates_on_mortgages"):
        country.banks.ts.override_current(key, country.banks.ts.current(key) + 0.0025)
    country._set_household_target_demand(replace_current=True)
    np.testing.assert_array_equal(hh.ts.current("target_consumption_interest_rate_cashflow"), before)


def test_variable_overdraft_interest_is_included_in_contractual_payment_change(aligned_country, monkeypatch):
    country, _ = aligned_country
    hh = country.households
    monkeypatch.setattr(country, "_common_permanent_income_terms", lambda: (0.03, 0.01))
    assets = hh.ts.current("liquid_financial_assets").copy()
    assets[0] = -100 * country.scale
    hh.ts.override_current("liquid_financial_assets", assets)
    # Seed the observed opening obligation before the variable-rate reset.
    for name, values in country._contractual_debt_service_components().items():
        hh.ts["consumption_debt_service_" + name] = values.copy()
    rate = country.banks.ts.current("overdraft_rate_on_household_deposits").copy()
    country.banks.ts.override_current("overdraft_rate_on_household_deposits", rate + 0.0025)
    country._set_household_target_demand(replace_current=False)
    expected_change = 0.0025 * np.maximum(-assets, 0)
    np.testing.assert_allclose(hh.ts.current("consumption_debt_service_change_nominal"), expected_change, atol=1e-7)
    np.testing.assert_allclose(
        hh.ts.current("target_consumption_debt_service_cashflow"),
        -0.75 * expected_change / hh.ts.current("consumption_debt_service_income_nominal"),
        atol=1e-12,
    )


def test_payment_normalizer_is_the_shared_smoothed_behavioural_income(aligned_country, monkeypatch):
    country, _ = aligned_country
    hh = country.households
    monkeypatch.setattr(country, "_common_permanent_income_terms", lambda: (0.03, 0.01))
    country._set_household_target_demand(replace_current=False)
    hh.ts.override_current("expected_income_employee", hh.ts.current("expected_income_employee") + country.scale)
    country._set_household_target_demand(replace_current=True)
    deflator = country.economy.current_consumer_price_level() / country.economy.initial_consumer_price_level()
    expected = hh.ts.current("target_consumption_annual_income_base") / 4 * deflator
    np.testing.assert_allclose(hh.ts.current("consumption_debt_service_income_nominal"), expected)
    np.testing.assert_allclose(hh.ts.current("target_consumption_debt_service_income_nominal"), expected)
    np.testing.assert_allclose(expected, hh.ts.current("target_consumption_real_income") * deflator)
    assert np.any(~np.isclose(expected, hh.ts.current("expected_behavioural_income_total_raw")))
    assert np.any(~np.isclose(expected, hh.ts.current("expected_behavioural_income")))


def test_financing_reserves_scheduled_service_once_after_behavioral_payment_response(aligned_country, monkeypatch):
    country, _ = aligned_country
    hh = country.households
    monkeypatch.setattr(country, "_common_permanent_income_terms", lambda: (0.03, 0.01))
    previous = hh.ts.current("consumption_debt_service_total_nominal").copy()
    # A lower previous obligation creates a nonzero behavioural response this period.
    hh.ts.override_current("consumption_debt_service_total_nominal", previous * 0.5)
    before_income = hh.ts.current("expected_income").copy()
    country._set_household_target_demand(replace_current=False)
    consumer, mortgage = country.credit_market.preview_opening_household_service()
    expected = (
        hh.consumption_cash_factor() * hh.ts.current("target_consumption").sum(axis=1)
        + hh.current_consumption_cash_rent()
        + consumer
        + mortgage
        + hh.target_investment_cash()
        - hh.current_financing_income()
    )
    np.testing.assert_allclose(hh.ts.current("liquidity_shortfall"), expected, atol=1e-6)
    np.testing.assert_array_equal(hh.ts.current("expected_income"), before_income)
    assert np.any(hh.ts.current("target_consumption_debt_service_cashflow") < 0)


def test_initial_payment_history_anchors_contractual_obligation_and_cash_income(aligned_country):
    country, _ = aligned_country
    hh = country.households
    prefix = "consumption_debt_service_"
    components = country._contractual_debt_service_components()
    for name, values in components.items():
        assert len(hh.ts.dicts[prefix + name]) == 1
        np.testing.assert_array_equal(hh.ts.current(prefix + name), values)
    np.testing.assert_array_equal(hh.ts.current(prefix + "previous_total_nominal"), components["total_nominal"])
    np.testing.assert_array_equal(hh.ts.current(prefix + "change_nominal"), np.zeros(hh.ts.current("n_households")))
    expected_income = np.maximum(
        hh.ts.current("expected_behavioural_income"), country.economy.ts.current("subsistence_consumption")
    )
    np.testing.assert_array_equal(hh.ts.current(prefix + "income_nominal"), expected_income)


# ---- income_denominator / income_denominator_window drive the prepared path ---------------------------


def _record_prepared_inputs(country, monkeypatch, *, force_window=None):
    """Capture ConsumptionInputs from the live adapter; optionally force the window (the old hard-coded 4)."""
    adapter, original, captured = (
        country._consumption_input_adapter,
        country._consumption_input_adapter.collect_inputs,
        [],
    )

    def collect(*args, **kwargs):
        if force_window is not None:
            kwargs["history_window"] = force_window
        captured.append(original(*args, **kwargs))
        return captured[-1]

    monkeypatch.setattr(adapter, "collect_inputs", collect)
    return captured


def test_paper_yaml_denominator_keys_reach_the_consumption_rule():
    root = Path(__file__).resolve().parents[4]
    production = load_country_configuration(root / "run_model/config/country_config_FRA.yaml", country_iso3="FRA")
    parameters = production.households.functions.consumption.parameters
    assert (parameters["income_denominator"], parameters["income_denominator_window"]) == ("geometric_average", 4)


def _run_periods(datawrapper, monkeypatch, periods=3, *, mode=None, window=None, force_window=None):
    simulation, _ = _aligned_simulation(datawrapper, monkeypatch)
    country = simulation.countries["FRA"]
    rule = country.households.functions["consumption"]
    if mode is not None:
        rule.income_denominator, rule.income_denominator_window = mode, window
    captured = _record_prepared_inputs(country, monkeypatch, force_window=force_window)
    for t in range(periods):
        simulation.iterate(t)
    return rule, captured


def test_current_yaml_values_reproduce_the_previous_fixed_window(datawrapper, monkeypatch):
    """geometric_average / 4 is what the prepared path always used: the inputs are identical."""
    rule, configured = _run_periods(datawrapper, monkeypatch)
    assert (rule.income_denominator, rule.income_denominator_window) == ("geometric_average", 4)
    _, fixed = _run_periods(datawrapper, monkeypatch, force_window=4)
    assert len(configured) == len(fixed) > 1
    assert configured[-1].window_row_count > 1
    for a, b in zip(configured, fixed):
        assert a.histories.history_window == b.histories.history_window == 4
        for name in ("income_base_real", "annual_income_base_real", "pi_log_ratio_rebased"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


@pytest.mark.parametrize("mode, window, expected", [("current", 4, 1), ("geometric_average", 2, 2)])
def test_denominator_mode_and_window_are_honoured(datawrapper, monkeypatch, mode, window, expected):
    _, captured = _run_periods(datawrapper, monkeypatch, mode=mode, window=window)
    assert captured[-1].histories.history_window == captured[-1].window_row_count == expected
