import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from macro_data.configuration.countries import Country as CountryName
from macromodel.agents.firms.func.productivity_investment_planner import SimpleProductivityInvestmentPlanner
from macromodel.configurations import CountryConfiguration, GoodsMarketConfiguration, SimulationConfiguration
from macromodel.simulation import (
    Simulation,
    check_compatibility,
    get_compatibility_mismatches,
    resolve_goods_market_configuration,
)
from macromodel.utils.prehooks.productivity_subsidy import create_productivity_subsidy_hook


def _goods_market_with_buyer_minimum_fill_micro(value: float) -> GoodsMarketConfiguration:
    configuration = GoodsMarketConfiguration()
    configuration.functions.clearing.parameters["buyer_minimum_fill_micro"] = value
    return configuration


def test_country_configuration_accepts_goods_market():
    configuration = CountryConfiguration(
        goods_market={
            "functions": {
                "clearing": {
                    "parameters": {
                        "buyer_minimum_fill_micro": 0.75,
                    },
                },
            },
        },
    )

    assert configuration.goods_market.functions.clearing.parameters["buyer_minimum_fill_micro"] == 0.75


def test_goods_market_configuration_can_be_set_from_country_config():
    country_configuration = CountryConfiguration(
        goods_market=_goods_market_with_buyer_minimum_fill_micro(0.25),
    )
    simulation_configuration = SimulationConfiguration(country_configurations={"FRA": country_configuration})

    resolved_configuration = resolve_goods_market_configuration(simulation_configuration)

    assert resolved_configuration.functions.clearing.parameters["buyer_minimum_fill_micro"] == 0.25


def test_simulation_goods_market_configuration_remains_supported():
    simulation_goods_market = _goods_market_with_buyer_minimum_fill_micro(0.4)
    simulation_configuration = SimulationConfiguration(
        country_configurations={"FRA": CountryConfiguration()},
        goods_market_configuration=simulation_goods_market,
    )

    resolved_configuration = resolve_goods_market_configuration(simulation_configuration)

    assert resolved_configuration.functions.clearing.parameters["buyer_minimum_fill_micro"] == 0.4


def test_default_country_goods_market_configurations_do_not_conflict_with_one_override():
    default_country_configuration = CountryConfiguration()
    france_configuration = CountryConfiguration(
        goods_market=_goods_market_with_buyer_minimum_fill_micro(0.25),
    )
    simulation_configuration = SimulationConfiguration(
        country_configurations={
            "FRA": france_configuration,
            "ESP": default_country_configuration,
        },
    )

    resolved_configuration = resolve_goods_market_configuration(simulation_configuration)

    assert resolved_configuration.functions.clearing.parameters["buyer_minimum_fill_micro"] == 0.25


def test_iterate_runs_credit_and_feasibility_before_single_labour_clear(monkeypatch):
    events = []

    class _TS:
        def current(self, _key):
            return [0.0]

    class _Country:
        country_name = "FRA"
        economy = SimpleNamespace(ts=_TS())

        def __getattr__(self, name):
            if name.startswith(("initialisation", "estimation", "target", "update", "prepare", "clear", "process")):

                def _method(**_kwargs):
                    events.append(f"country.{name}")

                return _method
            raise AttributeError(name)

    class _ExchangeRates:
        def get_current_exchange_rates_from_usd_to_lcu(self, **_kwargs):
            events.append("exchange_rates.get")
            return 1.0

    class _RegionalAggregator:
        def sync_central_banks(self, _countries):
            events.append("regional.sync")

    class _GoodsMarket:
        def prepare(self):
            events.append("goods.prepare")

        def clear(self):
            events.append("goods.clear")

        def record(self):
            events.append("goods.record")

    class _RestOfWorld:
        def update_planning_metrics(self, **_kwargs):
            events.append("row.update_planning_metrics")

        def record_bought_goods(self):
            events.append("row.record_bought_goods")

    class _Timestep:
        year = 2014
        month = 1

        def step(self):
            events.append("timestep.step")

    monkeypatch.setattr(Simulation, "production_price_index", property(lambda _self: 1.0))
    monkeypatch.setattr(Simulation, "total_real_production", property(lambda _self: 1.0))
    monkeypatch.setattr(Simulation, "aggregate_nominal_production", property(lambda _self: 1.0))

    country = _Country()
    simulation = Simulation(
        countries={"FRA": country},
        rest_of_the_world=_RestOfWorld(),
        goods_market=_GoodsMarket(),
        exchange_rates=_ExchangeRates(),
        timestep=_Timestep(),
        configuration=SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()}),
        initial_year=2014,
        regional_aggregator=_RegionalAggregator(),
    )

    simulation.iterate()

    assert events.count("country.clear_labour_market") == 1
    assert events.index("regional.sync") < events.index("country.prepare_credit_market_clearing")
    assert events.index("country.clear_credit_market") < events.index("country.process_housing_market_clearing")
    assert events.index("country.process_credit_market_clearing") < events.index(
        "country.prepare_post_credit_feasible_activity_plan"
    )
    assert events.index("country.prepare_post_credit_feasible_activity_plan") < events.index(
        "country.clear_labour_market"
    )
    assert events.index("country.clear_labour_market") < events.index("country.update_post_labour_planning_metrics")
    assert events.index("country.update_post_labour_planning_metrics") < events.index(
        "country.prepare_goods_market_clearing"
    )


def test_different_non_default_country_goods_market_configurations_raise():
    france_configuration = CountryConfiguration(
        goods_market=_goods_market_with_buyer_minimum_fill_micro(0.25),
    )
    spain_configuration = CountryConfiguration(
        goods_market=_goods_market_with_buyer_minimum_fill_micro(0.5),
    )
    simulation_configuration = SimulationConfiguration(
        country_configurations={
            "FRA": france_configuration,
            "ESP": spain_configuration,
        },
    )

    with pytest.raises(ValueError, match="non-default country-level goods_market configurations must match"):
        resolve_goods_market_configuration(simulation_configuration)


def test_conflicting_simulation_and_country_goods_market_configurations_raise():
    country_configuration = CountryConfiguration(
        goods_market=_goods_market_with_buyer_minimum_fill_micro(0.25),
    )
    simulation_configuration = SimulationConfiguration(
        country_configurations={"FRA": country_configuration},
        goods_market_configuration=_goods_market_with_buyer_minimum_fill_micro(0.5),
    )

    with pytest.raises(ValueError, match="set it at either the simulation level or the country level"):
        resolve_goods_market_configuration(simulation_configuration)


def test_from_datawrapper_uses_country_goods_market_configuration(datawrapper):
    country_configuration = CountryConfiguration(
        goods_market=_goods_market_with_buyer_minimum_fill_micro(0.25),
    )
    simulation_configuration = SimulationConfiguration(country_configurations={"FRA": country_configuration})

    simulation = Simulation.from_datawrapper(
        datawrapper=datawrapper,
        simulation_configuration=simulation_configuration,
    )

    assert simulation.goods_market.functions["clearing"].buyer_minimum_fill_micro == 0.25
    assert (
        simulation.configuration.goods_market_configuration.functions.clearing.parameters["buyer_minimum_fill_micro"]
        == 0.25
    )


def test_reset_uses_country_goods_market_configuration(datawrapper):
    simulation_configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})
    simulation = Simulation.from_datawrapper(
        datawrapper=datawrapper,
        simulation_configuration=simulation_configuration,
    )

    reset_country_configuration = CountryConfiguration(
        goods_market=_goods_market_with_buyer_minimum_fill_micro(0.25),
    )
    reset_configuration = SimulationConfiguration(country_configurations={"FRA": reset_country_configuration})

    simulation.reset(reset_configuration)

    assert simulation.goods_market.functions["clearing"].buyer_minimum_fill_micro == 0.25
    assert (
        simulation.configuration.goods_market_configuration.functions.clearing.parameters["buyer_minimum_fill_micro"]
        == 0.25
    )


@pytest.mark.parametrize("seed", [0, 100, 150, 200, 145])
def test_simulation(datawrapper, seed):
    """Test the simulation."""
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})

    configuration.seed = seed

    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)

    assert set(simulation.countries.keys()) == {"FRA"}

    households = simulation.countries["FRA"].households
    individuals = simulation.countries["FRA"].individuals

    n_individuals = individuals.n_individuals
    households_lengths = [len(corr_ind) for corr_ind in households.states["corr_individuals"]]
    assert n_individuals == sum(households_lengths)
    # no empty households
    assert all(households_lengths)

    for _ in range(10):
        simulation.iterate()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        simulation.save(save_dir=tmp, file_name="simulation_long.h5")
        simulation.shallow_hdf_save(save_dir=tmp, file_name="simulation_shallow.h5")
        dicts = simulation.shallow_df_dict()
        assert "FRA" in dicts

    france = simulation.countries[CountryName("FRA")]

    shallow_output = france.shallow_output()

    gross_output = shallow_output["Gross Output"]

    france_datawrapper = datawrapper.synthetic_countries[CountryName("FRA")]
    france_datawrapper_firms = france_datawrapper.firms

    firm_data = france_datawrapper_firms.firm_data
    firms_output_lcu = firm_data.groupby("Industry").apply(
        lambda x: (x["Production"] * x["Price"]).sum(), include_groups=False
    )

    assert gross_output.loc[0] == pytest.approx(firms_output_lcu.sum(), rel=1e-4)

    assert True


@pytest.mark.parametrize("seed", [0, 100])
def test_all_industries(allind_datawrapper, seed):
    n_industries = allind_datawrapper.n_industries
    configuration = SimulationConfiguration(
        country_configurations={"FRA": CountryConfiguration.n_industry_default(n_industries=n_industries)}
    )

    configuration.seed = seed

    simulation = Simulation.from_datawrapper(datawrapper=allind_datawrapper, simulation_configuration=configuration)

    for _ in range(3):
        simulation.iterate()

    assert True


def test_canadian_disagg(can_disagg_datawrapper):
    n_industries = can_disagg_datawrapper.n_industries
    firms_bundled_industries = ["B05a", "B05b", "B05c", "C19"]
    industries = can_disagg_datawrapper.industries
    firms_energy_bundle = [list(industries).index(ind) for ind in firms_bundled_industries]

    firms_substitution_bundles = [firms_energy_bundle]

    # Household energy bundle with only B05a and C19 for testing
    household_bundled_industries = ["B05a", "C19"]
    household_energy_bundle = [list(industries).index(ind) for ind in household_bundled_industries]
    household_substitution_bundles = [household_energy_bundle]

    configuration = SimulationConfiguration(
        country_configurations={
            "CAN": CountryConfiguration.n_industry_default(
                n_industries=n_industries,
                firms_bundles=firms_substitution_bundles,
                household_bundles=household_substitution_bundles,
            )
        }
    )

    assert configuration.country_configurations["CAN"].firms.functions.production.name == "BundledLeontief"
    assert (
        configuration.country_configurations["CAN"].households.functions.consumption.name == "CESHouseholdConsumption"
    )

    assert configuration.country_configurations["CAN"].firms.functions.production.name == "BundledLeontief"

    configuration.seed = 0
    simulation = Simulation.from_datawrapper(datawrapper=can_disagg_datawrapper, simulation_configuration=configuration)

    for _ in range(3):
        simulation.iterate()

    shallow_output = simulation.countries["CAN"].shallow_output()

    keys = [
        "Firm Input Emissions",
        "Firm Capital Emissions",
        "Household Consumption Emissions",
        "Household Investment Emissions",
        "Government Emissions",
    ]

    for key in keys:
        assert np.all(shallow_output[key] > 0)

    assert True


def test_can_provincial(can_provincial_datawrapper):
    n_industries = can_provincial_datawrapper.n_industries

    all_provs = can_provincial_datawrapper.synthetic_countries.keys()

    configuration = SimulationConfiguration(
        country_configurations={
            province: CountryConfiguration.n_industry_default(n_industries=n_industries) for province in all_provs
        }
    )

    configuration.seed = 0

    simulation = Simulation.from_datawrapper(
        datawrapper=can_provincial_datawrapper, simulation_configuration=configuration
    )

    for _ in range(3):
        simulation.iterate()

    simulation.countries["CAN_AB"].shallow_output()

    assert True


def test_tfp_growth_with_investment(datawrapper):
    """Test that TFP growth mechanism works with productivity investment.

    Creates two simulations with identical seeds:
    1. Control: No TFP growth (all parameters set to zero/disabled)
    2. Treatment: TFP growth enabled with high investment effectiveness and low hurdle rate

    Verifies that firms in the treatment simulation have higher TFP after several periods.
    """
    # Base configuration for control (no TFP growth)
    config_no_growth = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})
    config_no_growth.seed = 0  # Fixed seed for reproducibility

    # Disable TFP growth in control
    config_no_growth.country_configurations["FRA"].firms.parameters.tfp_base_growth_rate = 0.0
    config_no_growth.country_configurations["FRA"].firms.parameters.tfp_investment_elasticity = 0.0

    # Configuration for treatment (with TFP growth)
    config_with_growth = deepcopy(config_no_growth)

    # Enable TFP growth with favorable parameters
    config_with_growth.country_configurations["FRA"].firms.parameters.tfp_base_growth_rate = 0.001  # 0.1% base growth
    config_with_growth.country_configurations["FRA"].firms.parameters.tfp_investment_elasticity = 0.5  # High elasticity

    # Set productivity investment planner parameters
    config_with_growth.country_configurations[
        "FRA"
    ].firms.functions.productivity_investment_planner.name = "SimpleProductivityInvestmentPlanner"
    config_with_growth.country_configurations["FRA"].firms.functions.productivity_investment_planner.parameters = {
        "n_firms": config_with_growth.country_configurations["FRA"].firms.n_firms,
        "hurdle_rate": 1e-5,  # Very low hurdle rate (almost no discounting)
        "investment_effectiveness": 0.5,  # High effectiveness
        "investment_elasticity": 0.5,  # Match the TFP elasticity
        "max_investment_fraction": 0.2,  # Allow up to 20% of available cash
    }

    # Also configure the productivity growth function
    config_with_growth.country_configurations["FRA"].firms.functions.productivity_growth.name = "SimpleTFPGrowth"
    config_with_growth.country_configurations["FRA"].firms.functions.productivity_growth.parameters = {
        "investment_effectiveness": 0.5,  # High effectiveness for growth calculation
    }

    # Create simulations
    sim_no_growth = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=config_no_growth)

    sim_with_growth = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=config_with_growth)

    # Get initial TFP values (should be identical)
    initial_tfp_no_growth = sim_no_growth.countries["FRA"].firms.states["tfp_multiplier"].copy()
    initial_tfp_with_growth = sim_with_growth.countries["FRA"].firms.states["tfp_multiplier"].copy()

    # Verify initial TFP values are the same (both should be 1.0)
    np.testing.assert_array_almost_equal(initial_tfp_no_growth, initial_tfp_with_growth)
    np.testing.assert_array_almost_equal(initial_tfp_no_growth, np.ones_like(initial_tfp_no_growth))

    # Run both simulations for several periods
    n_periods = 10
    for _ in range(n_periods):
        sim_no_growth.iterate()
        sim_with_growth.iterate()

    # Get final TFP values
    final_tfp_no_growth = sim_no_growth.countries["FRA"].firms.states["tfp_multiplier"]
    final_tfp_with_growth = sim_with_growth.countries["FRA"].firms.states["tfp_multiplier"]

    # Verify that TFP in the growth simulation is higher
    # Control should remain at 1.0 (no growth)
    np.testing.assert_array_almost_equal(final_tfp_no_growth, np.ones_like(final_tfp_no_growth))

    # Treatment should have TFP > 1.0 for at least most firms
    assert np.mean(final_tfp_with_growth) > 1.0, "Average TFP should be greater than 1.0 with growth enabled"
    assert np.sum(final_tfp_with_growth > 1.0) > len(final_tfp_with_growth) * 0.8, "Most firms should have TFP > 1.0"

    # Verify all firms in treatment have at least as much TFP as control
    assert np.all(final_tfp_with_growth >= final_tfp_no_growth), "All firms should have TFP >= control"

    # Check that productivity investment is actually happening
    if len(sim_with_growth.countries["FRA"].firms.ts.executed_productivity_investment) > 0:
        total_investment = sum(
            inv.sum() for inv in sim_with_growth.countries["FRA"].firms.ts.executed_productivity_investment
        )
        assert total_investment > 0, (
            f"There should be positive productivity investment, first 5 elements: {total_investment[:5]}"
        )


def test_tfp_growth_applies_after_current_period_production(datawrapper):
    """TFP growth computed in one period should only affect later production."""
    config_no_growth = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})
    config_no_growth.seed = 0
    config_no_growth.country_configurations["FRA"].firms.parameters.tfp_base_growth_rate = 0.0
    config_no_growth.country_configurations["FRA"].firms.parameters.tfp_investment_elasticity = 0.0

    config_with_growth = deepcopy(config_no_growth)
    config_with_growth.country_configurations["FRA"].firms.parameters.tfp_base_growth_rate = 0.25
    config_with_growth.country_configurations["FRA"].firms.parameters.tfp_investment_elasticity = 0.0
    config_with_growth.country_configurations["FRA"].firms.functions.productivity_growth.name = "SimpleTFPGrowth"
    config_with_growth.country_configurations["FRA"].firms.functions.productivity_growth.parameters = {
        "investment_effectiveness": 0.0,
    }

    sim_no_growth = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=config_no_growth)
    sim_with_growth = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=config_with_growth)

    sim_no_growth.iterate()
    sim_with_growth.iterate()

    firms_no_growth = sim_no_growth.countries["FRA"].firms
    firms_with_growth = sim_with_growth.countries["FRA"].firms

    np.testing.assert_allclose(firms_with_growth.ts.current("production"), firms_no_growth.ts.current("production"))
    assert np.mean(firms_with_growth.states["tfp_multiplier"]) > np.mean(firms_no_growth.states["tfp_multiplier"])
    np.testing.assert_allclose(
        firms_with_growth.ts.current("tfp_multiplier"), firms_with_growth.states["tfp_multiplier"]
    )
    assert np.mean(firms_with_growth.ts.current("tfp_multiplier")) > np.mean(
        firms_with_growth.ts.prev("tfp_multiplier")
    )


def test_productivity_subsidy_uses_execution_path(datawrapper):
    """Subsidies should stage nominal investment instead of appending executed investment directly."""
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})
    configuration.seed = 0
    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)
    firms = simulation.countries["FRA"].firms
    industry_code = firms.industries[firms.states["Industry"][0]]
    initial_executed_len = len(firms.ts.executed_productivity_investment)

    hook = create_productivity_subsidy_hook(
        country_code="FRA",
        industry_code=industry_code,
        target_year=2020,
        target_month=1,
        subsidy_amount=1_000.0,
    )
    hook(simulation, 2020, 1)

    assert len(firms.ts.executed_productivity_investment) == initial_executed_len
    assert firms.states["forced_productivity_investment"].sum() == pytest.approx(1_000.0)

    simulation.iterate()

    assert firms.states["forced_productivity_investment"].sum() == pytest.approx(0.0)
    assert firms.ts.current("executed_productivity_investment").sum() >= 1_000.0
    assert firms.ts.current("executed_tfp_investment").sum() >= 1_000.0


def test_check_compatibility(datawrapper):
    """Test the compatibility check."""
    france = CountryName("FRA")
    country_data_configuration = datawrapper.configuration.country_configs[france]
    country_sim_configuration = CountryConfiguration()

    country_sim_configuration.firms.parameters.capital_inputs_utilisation_rate = 0.1
    country_sim_configuration.firms.parameters.intermediate_inputs_utilisation_rate = 0.1

    assert not check_compatibility(country_data_configuration, country_sim_configuration)


def test_capital_compensation_mode_mismatch_is_incompatible(datawrapper):
    country_data_configuration = datawrapper.configuration.country_configs[CountryName("FRA")]
    country_sim_configuration = CountryConfiguration()
    country_sim_configuration.firms.parameters.capital_compensation_accounting_mode = "surplus_pool"

    mismatches = get_compatibility_mismatches(country_data_configuration, country_sim_configuration)

    assert not check_compatibility(country_data_configuration, country_sim_configuration)
    assert any(mismatch.startswith("firms.capital_compensation_accounting_mode") for mismatch in mismatches)


def test_capital_depreciation_settings_mismatch_is_incompatible(datawrapper):
    country_data_configuration = datawrapper.configuration.country_configs[CountryName("FRA")]
    country_sim_configuration = CountryConfiguration()
    country_sim_configuration.firms.parameters.capital_depreciation_accounting_mode = "eurostat_cfc"
    country_sim_configuration.firms.parameters.capital_replacement_matrix_source = "eurostat_cfc_output"

    mismatches = get_compatibility_mismatches(country_data_configuration, country_sim_configuration)

    assert not check_compatibility(country_data_configuration, country_sim_configuration)
    assert any(mismatch.startswith("firms.capital_depreciation_accounting_mode") for mismatch in mismatches)
    assert any(mismatch.startswith("firms.capital_replacement_matrix_source") for mismatch in mismatches)


def test_capital_compensation_mode_mismatch_raises_in_simulation(datawrapper):
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})
    configuration.country_configurations["FRA"].firms.parameters.capital_compensation_accounting_mode = "surplus_pool"

    with pytest.raises(ValueError, match="capital accounting settings must match"):
        Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)


def test_random_seed(datawrapper):
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})

    configuration.seed = 0

    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)

    for i in range(3):
        simulation.iterate()

    gdp1 = np.stack(simulation.countries["FRA"].economy.ts.historic("gdp_output")).flatten()

    simulation_bis = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)

    for i in range(3):
        simulation_bis.iterate()

    gdp_bis = np.stack(simulation_bis.countries["FRA"].economy.ts.historic("gdp_output")).flatten()

    assert gdp1 == pytest.approx(gdp_bis, rel=1e-2)


def test_reset(datawrapper):
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})

    configuration.seed = 0

    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)

    for i in range(3):
        simulation.iterate()

    gdp1 = np.stack(simulation.countries["FRA"].economy.ts.historic("gdp_output")).flatten()

    simulation.reset()

    assert len(simulation.countries["FRA"].firms.ts.historic("price")) == 1

    for i in range(3):
        simulation.iterate()

    gdp2 = np.stack(simulation.countries["FRA"].economy.ts.historic("gdp_output")).flatten()

    assert gdp1 == pytest.approx(gdp2, rel=1e-2)


def test_longrun(datawrapper):
    """Test the longrun."""
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()}, t_max=200)

    configuration.seed = 0

    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)

    simulation.run()

    assert True


def test_change_config(datawrapper):
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})

    configuration.seed = 0

    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)

    for i in range(3):
        simulation.iterate()

    gdp1 = np.stack(simulation.countries["FRA"].economy.ts.historic("gdp_output")).flatten()
    new_configuration = deepcopy(simulation.configuration)

    # first just change seed
    new_configuration.seed = 1

    simulation.reset(new_configuration)

    for i in range(3):
        simulation.iterate()

    gdp2 = np.stack(simulation.countries["FRA"].economy.ts.historic("gdp_output")).flatten()

    assert np.sum(gdp1 - gdp2) != 0

    # reset seed again, check that changing params  change the output

    new_configuration.seed = 0

    # edit France config
    new_configuration.country_configurations["FRA"].firms.parameters.capital_inputs_utilisation_rate = 0.5

    # edit France config
    new_configuration.country_configurations["FRA"].firms.parameters.capital_inputs_utilisation_rate = 0.5

    original_param = new_configuration.country_configurations["FRA"].firms.functions.prices.parameters[
        "price_setting_speed_gf"
    ]

    new_configuration.country_configurations["FRA"].firms.functions.prices.parameters["price_setting_speed_gf"] = (
        1 - original_param
    )

    simulation.reset(new_configuration)

    assert len(simulation.countries["FRA"].firms.ts.historic("price")) == 1

    for i in range(3):
        simulation.iterate()

    gdp3 = np.stack(simulation.countries["FRA"].economy.ts.historic("gdp_output")).flatten()

    assert np.sum(gdp1 - gdp3) != 0


def test_reset_row_params(datawrapper):
    """Test the reset params."""
    country_sim_configuration = CountryConfiguration()

    sim_configuration = SimulationConfiguration(country_configurations={"FRA": country_sim_configuration})
    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=sim_configuration)

    for _ in range(5):
        simulation.iterate()

    values = [0.0, 1.0]

    for x in values:
        new_row_conf = deepcopy(sim_configuration.row_configuration)
        new_row_conf.functions.exports.parameters["consistency"] = x
        sim_configuration.row_configuration = new_row_conf

        simulation.reset(sim_configuration)
        row = simulation.rest_of_the_world
        func = row.functions["exports"]

        param = func.consistency

        assert param == x
        simulation.iterate()


def test_reset_firm_params(datawrapper):
    """Test the reset params."""
    country_sim_configuration = CountryConfiguration()

    def redo_configuration(
        country_conf: CountryConfiguration,
        target_inputs_capital_: float,
    ):
        new_country_conf_ = deepcopy(country_conf)
        new_country_conf_.firms.functions.target_production.parameters[
            "intermediate_inputs_target_considers_capital_inputs"
        ] = target_inputs_capital_
        return new_country_conf_

    country_sim_configuration.firms.reset_params["capital_inputs_utilisation_rate"] = 0.1
    country_sim_configuration.firms.reset_params["intermediate_inputs_utilisation_rate"] = 0.1

    sim_configuration = SimulationConfiguration(country_configurations={"FRA": country_sim_configuration})
    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=sim_configuration)

    for _ in range(5):
        simulation.iterate()

    values = np.linspace(0, 1, 10)

    for x in values:
        new_country_conf = redo_configuration(country_sim_configuration, x)
        sim_configuration.country_configurations["FRA"] = new_country_conf

        simulation.reset(sim_configuration)
        firms = simulation.countries["FRA"].firms
        func = firms.functions["target_production"]

        param = func.intermediate_inputs_target_considers_capital_inputs

        assert param == x
        simulation.iterate()


def test_target_production_inventory_fraction_does_not_reset_initial_inventory(datawrapper):
    country_sim_configuration = CountryConfiguration()
    country_sim_configuration.firms.functions.target_production.parameters["existing_inventory_fraction"] = 0.75

    assert country_sim_configuration.firms.reset_params["initial_inventory_to_input_fraction"] == 0.0
    assert check_compatibility(datawrapper.configuration.country_configs["FRA"], country_sim_configuration)


def test_alternative_labour(datawrapper):
    """Test the alternative labour."""
    country_sim_configuration = CountryConfiguration()

    country_sim_configuration.labour_market.functions.clearing.parameters["firing_speed"] = 0.8
    country_sim_configuration.labour_market.functions.clearing.parameters["hiring_speed"] = 0.8
    country_sim_configuration.labour_market.functions.clearing.parameters["individuals_quitting"] = True
    # random_firing_probability
    country_sim_configuration.labour_market.functions.clearing.parameters["random_firing_probability"] = 0.02

    sim_configuration = SimulationConfiguration(
        country_configurations={"FRA": country_sim_configuration}, seed=0, t_max=5
    )

    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=sim_configuration)

    simulation.run()

    assert True


def test_large_firing_rate(allind_datawrapper):
    country_sim_configuration = CountryConfiguration.n_industry_default(n_industries=allind_datawrapper.n_industries)

    country_sim_configuration.labour_market.functions.clearing.parameters["firing_speed"] = 0.8
    country_sim_configuration.labour_market.functions.clearing.parameters["hiring_speed"] = 0.8
    country_sim_configuration.labour_market.functions.clearing.parameters["individuals_quitting"] = True
    # random_firing_probability
    country_sim_configuration.labour_market.functions.clearing.parameters["random_firing_probability"] = 0.99

    sim_configuration = SimulationConfiguration(
        country_configurations={"FRA": country_sim_configuration}, seed=0, t_max=5
    )

    simulation = Simulation.from_datawrapper(datawrapper=allind_datawrapper, simulation_configuration=sim_configuration)

    simulation.run()

    assert True


@pytest.mark.parametrize(
    "tfp_growth_type", ["NoOpTFPGrowth", "SimpleTFPGrowth", "StochasticTFPGrowth", "SectoralTFPGrowth"]
)
@pytest.mark.parametrize("seed", [0, 100])
def test_simulation_with_tfp_growth(datawrapper, seed, tfp_growth_type):
    """Test the simulation with different TFP growth configurations."""
    # Create base configuration
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})

    # Modify the TFP growth configuration
    configuration.country_configurations["FRA"].firms.functions.productivity_growth.name = tfp_growth_type

    # Set parameters based on TFP growth type
    if tfp_growth_type == "NoOpTFPGrowth":
        # No parameters needed for NoOp
        configuration.country_configurations["FRA"].firms.functions.productivity_growth.parameters = {}
    elif tfp_growth_type == "SimpleTFPGrowth":
        # Parameters for simple TFP growth
        configuration.country_configurations["FRA"].firms.functions.productivity_growth.parameters = {
            "investment_effectiveness": 0.1
        }
        # Also set the base growth rate in parameters
        configuration.country_configurations["FRA"].firms.parameters.tfp_base_growth_rate = 0.001  # 0.1% per period
        configuration.country_configurations["FRA"].firms.parameters.tfp_investment_elasticity = 0.3
    elif tfp_growth_type == "StochasticTFPGrowth":
        # Parameters for stochastic TFP growth
        configuration.country_configurations["FRA"].firms.functions.productivity_growth.parameters = {
            "investment_effectiveness": 0.1,
            "shock_std": 0.005,  # 0.5% standard deviation for shocks
        }
        configuration.country_configurations["FRA"].firms.parameters.tfp_base_growth_rate = 0.001
        configuration.country_configurations["FRA"].firms.parameters.tfp_investment_elasticity = 0.3
    elif tfp_growth_type == "SectoralTFPGrowth":
        # Parameters for sectoral TFP growth
        configuration.country_configurations["FRA"].firms.functions.productivity_growth.parameters = {
            "investment_effectiveness": 0.1,
            "sector_base_growth": {},  # Could specify sector-specific rates here
            "sector_effectiveness": {},  # Could specify sector-specific effectiveness here
        }
        configuration.country_configurations["FRA"].firms.parameters.tfp_base_growth_rate = 0.001
        configuration.country_configurations["FRA"].firms.parameters.tfp_investment_elasticity = 0.3

    configuration.seed = seed

    # Create and run simulation
    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)

    assert set(simulation.countries.keys()) == {"FRA"}

    # Check that TFP multiplier is initialized
    firms = simulation.countries["FRA"].firms
    assert "tfp_multiplier" in firms.states
    assert np.all(firms.states["tfp_multiplier"] == 1.0)  # Should start at 1.0

    # Run simulation for several iterations
    for _ in range(5):
        simulation.iterate()

    # Check TFP behavior based on type
    final_tfp = firms.states["tfp_multiplier"]

    if tfp_growth_type == "NoOpTFPGrowth":
        # TFP should remain at 1.0 (no growth)
        assert np.allclose(final_tfp, 1.0), f"NoOpTFPGrowth should keep TFP at 1.0, got {final_tfp}"
    else:
        # For other types, TFP might change (though with small growth rates, changes could be minimal)
        # We mainly check that the simulation runs without errors
        assert np.all(final_tfp > 0), f"TFP should be positive, got {final_tfp}"
        assert np.all(np.isfinite(final_tfp)), f"TFP should be finite, got {final_tfp}"

    assert True


def test_tfp_only_investment_allocation(datawrapper, seed=42):
    """Test that investment allocation to TFP-only works correctly.

    Configure investment to go 100% to TFP, 0% to technical coefficients.
    Verify TFP multiplier improves while technical coefficient multipliers stay at 1.0.
    """
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})
    configuration.seed = seed

    # Configure for TFP-only investment allocation
    firms_config = configuration.country_configurations["FRA"].firms
    firms_config.functions.productivity_investment_planner.name = "SimpleProductivityInvestmentPlanner"
    firms_config.functions.productivity_investment_planner.parameters.update(
        {
            "tfp_investment_share": 1.0,  # 100% to TFP
            "max_investment_fraction": 0.2,  # High investment to see effects
            "investment_effectiveness": 0.3,  # High effectiveness
        }
    )
    firms_config.functions.productivity_growth.name = "SimpleTFPGrowth"
    firms_config.functions.technical_coefficients_growth.name = "NoOpTechnicalGrowth"  # Disable technical growth

    # Create and run simulation
    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)
    firms = simulation.countries["FRA"].firms

    # Store initial values
    initial_tfp = firms.states["tfp_multiplier"].copy()
    initial_intermediate_tech = firms.states["intermediate_tech_multipliers"].copy()
    initial_capital_tech = firms.states["capital_tech_multipliers"].copy()

    # Run simulation for several iterations to allow investment effects
    for _ in range(10):
        simulation.iterate()

    # Check results
    final_tfp = firms.states["tfp_multiplier"]
    final_intermediate_tech = firms.states["intermediate_tech_multipliers"]
    final_capital_tech = firms.states["capital_tech_multipliers"]

    # TFP should have improved (at least some firms should have TFP > 1.0)
    assert np.any(final_tfp > initial_tfp), "TFP should improve with TFP-only investment"
    assert np.all(final_tfp >= 1.0), "TFP multipliers should be >= 1.0"

    # Technical coefficients should remain at 1.0 (no technical investment)
    assert np.allclose(final_intermediate_tech, initial_intermediate_tech), (
        "Intermediate tech multipliers should not change with TFP-only investment"
    )
    assert np.allclose(final_capital_tech, initial_capital_tech), (
        "Capital tech multipliers should not change with TFP-only investment"
    )
    assert np.allclose(final_intermediate_tech, 1.0), "Intermediate tech multipliers should stay at 1.0"
    assert np.allclose(final_capital_tech, 1.0), "Capital tech multipliers should stay at 1.0"


def test_technical_only_investment_allocation(datawrapper, seed=42):
    """Test that investment allocation to technical coefficients-only works correctly.

    Configure investment to go 0% to TFP, 100% to technical coefficients.
    Verify technical coefficient multipliers improve while TFP stays at 1.0.
    """
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})
    configuration.seed = seed

    # Configure for technical-only investment allocation
    firms_config = configuration.country_configurations["FRA"].firms
    firms_config.functions.productivity_investment_planner.name = "SimpleProductivityInvestmentPlanner"
    firms_config.functions.productivity_investment_planner.parameters.update(
        {
            "tfp_investment_share": 0.0,  # 0% to TFP, 100% to technical
            "max_investment_fraction": 0.2,  # High investment to see effects
            "technical_investment_effectiveness": 0.3,  # High effectiveness
            "technical_diminishing_returns": 0.1,  # Low diminishing returns for faster growth
        }
    )
    firms_config.functions.productivity_growth.name = "NoOpTFPGrowth"  # Disable TFP growth
    firms_config.functions.technical_coefficients_growth.name = "SimpleTechnicalGrowth"
    firms_config.functions.technical_coefficients_growth.parameters.update(
        {
            "investment_effectiveness": 0.3,
            "diminishing_returns_factor": 0.1,
        }
    )

    # Create and run simulation
    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)
    firms = simulation.countries["FRA"].firms

    # Store initial values
    initial_tfp = firms.states["tfp_multiplier"].copy()
    initial_intermediate_tech = firms.states["intermediate_tech_multipliers"].copy()
    initial_capital_tech = firms.states["capital_tech_multipliers"].copy()

    # Store base coefficients for comparison
    base_intermediate_coeffs = firms.base_intermediate_inputs_productivity_matrix
    base_capital_coeffs = firms.base_capital_inputs_productivity_matrix

    # Run simulation for several iterations to allow investment effects
    for _ in range(10):
        simulation.iterate()

    # Check results
    final_tfp = firms.states["tfp_multiplier"]
    final_intermediate_tech = firms.states["intermediate_tech_multipliers"]
    final_capital_tech = firms.states["capital_tech_multipliers"]

    # TFP should remain at 1.0 (no TFP investment)
    assert np.allclose(final_tfp, initial_tfp), "TFP should not change with technical-only investment"
    assert np.allclose(final_tfp, 1.0), "TFP multipliers should stay at 1.0"

    # Technical coefficients should have improved
    # At least some multipliers should be > 1.0, and all should be >= 1.0
    intermediate_improved = np.any(final_intermediate_tech > initial_intermediate_tech)
    capital_improved = np.any(final_capital_tech > initial_capital_tech)

    # At least one type should have improved
    assert intermediate_improved or capital_improved, (
        "At least some technical multipliers should improve with technical-only investment"
    )

    # All multipliers should be >= 1.0 (productivity improvements)
    assert np.all(final_intermediate_tech >= 1.0), "Intermediate tech multipliers should be >= 1.0"
    assert np.all(final_capital_tech >= 1.0), "Capital tech multipliers should be >= 1.0"

    # Check effective coefficients vs base coefficients
    # Effective coefficients should be >= base coefficients (due to multipliers >= 1.0)
    effective_intermediate = firms.get_effective_intermediate_coefficients()
    effective_capital = firms.get_effective_capital_coefficients()

    # Get base coefficients for each firm's industry
    firm_industries = firms.states["Industry"]
    base_intermediate_for_firms = base_intermediate_coeffs[:, firm_industries].T
    base_capital_for_firms = base_capital_coeffs[:, firm_industries].T

    # Check that effective >= base (element-wise)
    intermediate_comparison = effective_intermediate >= base_intermediate_for_firms
    capital_comparison = effective_capital >= base_capital_for_firms

    # All should be >= base, and at least some should be > base
    assert np.all(intermediate_comparison), "All effective intermediate coeffs should be >= base"
    assert np.all(capital_comparison), "All effective capital coeffs should be >= base"

    # At least some should be strictly greater (using your suggested check)
    intermediate_some_better = (effective_intermediate > base_intermediate_for_firms).sum() > 0
    capital_some_better = (effective_capital > base_capital_for_firms).sum() > 0

    assert intermediate_some_better or capital_some_better, (
        "At least some effective coefficients should be strictly better than base coefficients"
    )


def test_technical_growth_uses_executed_investment(datawrapper, monkeypatch, seed=42):
    """Technical coefficient growth should consume realised, not merely planned, investment."""

    class TechnicalGrowthSpy:
        def __init__(self):
            self.intermediate_investment = None
            self.capital_investment = None
            self.intermediate_prices = None
            self.capital_prices = None

        def compute_intermediate_multiplier_growth(self, **kwargs):
            self.intermediate_investment = kwargs["technical_investment"].copy()
            self.intermediate_prices = kwargs["prices"].copy()
            return np.zeros_like(kwargs["current_multipliers"])

        def compute_capital_multiplier_growth(self, **kwargs):
            self.capital_investment = kwargs["technical_investment"].copy()
            self.capital_prices = kwargs["prices"].copy()
            return np.zeros_like(kwargs["current_multipliers"])

    class TFPGrowthSpy:
        def __init__(self):
            self.productivity_investment = None

        def compute_tfp_growth(self, **kwargs):
            self.productivity_investment = kwargs["productivity_investment"].copy()
            return np.zeros_like(kwargs["current_tfp"])

    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})
    configuration.seed = seed

    firms_config = configuration.country_configurations["FRA"].firms
    firms_config.functions.productivity_investment_planner.name = "SimpleProductivityInvestmentPlanner"
    firms_config.functions.productivity_investment_planner.parameters.update(
        {
            "tfp_investment_share": 0.0,
            "max_investment_fraction": 0.2,
            "technical_investment_effectiveness": 0.3,
        }
    )
    firms_config.functions.productivity_growth.name = "NoOpTFPGrowth"

    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)
    firms = simulation.countries["FRA"].firms
    technical_spy = TechnicalGrowthSpy()
    tfp_spy = TFPGrowthSpy()
    firms.functions["technical_coefficients_growth"] = technical_spy
    firms.functions["productivity_growth"] = tfp_spy
    original_execute_productivity_investment = firms.execute_productivity_investment

    def constrained_execute_productivity_investment():
        original_execute_productivity_investment()
        planned_total = firms.ts.current("planned_productivity_investment")
        planned_tfp = firms.ts.current("planned_tfp_investment")
        planned_technical = firms.ts.current("planned_technical_investment")
        execution_ratio = np.full_like(planned_total, 0.5)
        executed_total = planned_total * execution_ratio
        firms.ts.override_current("executed_productivity_investment", executed_total)
        firms.ts.override_current("executed_tfp_investment", planned_tfp * execution_ratio)
        firms.ts.override_current("direct_tfp_investment_cash_expense", planned_tfp * execution_ratio)
        firms.ts.override_current("executed_technical_investment", planned_technical * execution_ratio[:, np.newaxis])
        firms.ts.override_current(
            "real_executed_productivity_investment",
            firms.compute_real_productivity_investment(executed_total),
        )

    monkeypatch.setattr(firms, "execute_productivity_investment", constrained_execute_productivity_investment)

    simulation.iterate()

    planned = firms.ts.current("planned_technical_investment")
    executed = firms.ts.current("executed_technical_investment")
    planned_total = firms.ts.current("planned_productivity_investment")
    planned_tfp = firms.ts.current("planned_tfp_investment")
    executed_total = firms.ts.current("executed_productivity_investment")
    executed_tfp = firms.ts.current("executed_tfp_investment")
    ratio = np.divide(
        np.minimum(executed_total, planned_total),
        planned_total,
        out=np.zeros_like(executed_total),
        where=planned_total > 0,
    )
    assert planned.sum() > 0
    assert executed.sum() > 0
    assert not np.allclose(executed, planned)
    assert np.allclose(executed, planned * ratio[:, np.newaxis])
    assert np.allclose(executed_tfp, planned_tfp * ratio)
    assert np.allclose(tfp_spy.productivity_investment, 0.0)
    assert np.allclose(technical_spy.intermediate_investment, executed)
    assert np.allclose(technical_spy.capital_investment, executed)
    assert technical_spy.intermediate_prices.shape == (executed.shape[1],)
    assert technical_spy.capital_prices.shape == (executed.shape[1],)


def test_target_intensity_planner_execution_not_capped_by_net_capital(datawrapper, monkeypatch):
    """The revised direct-TFP planner executes planned TFP independent of net capital purchases."""

    class DirectPlannerFlag:
        executes_direct_tfp_independently = True

    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})
    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)
    firms = simulation.countries["FRA"].firms
    n_firms = firms.ts.current("n_firms")
    n_industries = firms.n_industries
    planned = np.full(n_firms, 100.0)

    firms.functions["productivity_investment_planner"] = DirectPlannerFlag()
    firms.ts.planned_productivity_investment.append(planned)
    firms.ts.planned_tfp_investment.append(planned)
    firms.ts.planned_technical_investment.append(np.zeros((n_firms, n_industries)))
    monkeypatch.setattr(firms, "compute_net_capital_investment_above_replacement", lambda: np.zeros(n_firms))
    monkeypatch.setattr(firms, "compute_real_productivity_investment", lambda investment: investment.copy())

    firms.execute_productivity_investment()

    assert np.allclose(firms.ts.current("net_capital_investment_above_replacement"), 0.0)
    assert np.allclose(firms.ts.current("executed_productivity_investment"), planned)
    assert np.allclose(firms.ts.current("executed_tfp_investment"), planned)


def test_firms_pass_industries_and_effective_rates_to_productivity_planner(datawrapper):
    """Firms should map bank long-term firm loan rates into firm-level planner rates."""

    class PlannerSpy:
        firm_industries = None
        effective_cost_rate = None

        def plan_productivity_investment(self, **kwargs):
            self.firm_industries = kwargs["firm_industries"].copy()
            self.effective_cost_rate = kwargs["effective_cost_rate"].copy()
            n_firms = len(kwargs["current_production"])
            n_industries = kwargs["n_industries"]
            return np.zeros(n_firms), np.zeros(n_firms), np.zeros((n_firms, n_industries))

    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})
    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)
    firms = simulation.countries["FRA"].firms
    planner_spy = PlannerSpy()
    bank_rates = np.linspace(0.01, 0.05, simulation.countries["FRA"].banks.ts.current("n_banks"))

    firms.functions["productivity_investment_planner"] = planner_spy
    firms.plan_productivity_investment(
        estimated_inflation=0.0,
        current_good_prices=np.ones(firms.n_industries),
        bank_interest_rates_on_long_term_firm_loans=bank_rates,
    )

    assert np.allclose(planner_spy.firm_industries, firms.states["Industry"])
    assert np.allclose(planner_spy.effective_cost_rate, bank_rates[firms.states["Corresponding Bank ID"]])


def test_simple_planner_execution_still_capped_by_net_capital(datawrapper, monkeypatch):
    """Existing planners keep the net-capital execution cap."""

    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})
    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)
    firms = simulation.countries["FRA"].firms
    n_firms = firms.ts.current("n_firms")
    n_industries = firms.n_industries
    planned = np.full(n_firms, 100.0)
    net_capital = np.full(n_firms, 25.0)

    firms.functions["productivity_investment_planner"] = SimpleProductivityInvestmentPlanner(n_firms=n_firms)
    firms.ts.planned_productivity_investment.append(planned)
    firms.ts.planned_tfp_investment.append(planned)
    firms.ts.planned_technical_investment.append(np.zeros((n_firms, n_industries)))
    monkeypatch.setattr(firms, "compute_net_capital_investment_above_replacement", lambda: net_capital)
    monkeypatch.setattr(firms, "compute_real_productivity_investment", lambda investment: investment.copy())

    firms.execute_productivity_investment()

    assert np.allclose(firms.ts.current("net_capital_investment_above_replacement"), net_capital)
    assert np.allclose(firms.ts.current("executed_productivity_investment"), net_capital)
    assert np.allclose(firms.ts.current("executed_tfp_investment"), net_capital)


def test_prehooks(datawrapper):
    """Test that pre-hooks execute correctly before each iteration."""
    # Track hook calls
    hook_calls = []

    def test_hook(simulation: Simulation, year: int, month: int) -> None:
        """Simple test hook that records when it's called."""
        hook_calls.append((year, month))

    # Create simulation
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})
    configuration.seed = 0

    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)

    # Register the test hook
    simulation.prehooks.append(test_hook)

    # Run for 3 iterations
    for _ in range(3):
        simulation.iterate()

    # Verify hook was called correct number of times
    assert len(hook_calls) == 3, f"Expected 3 hook calls, got {len(hook_calls)}"

    # Verify each call had year and month
    for year, month in hook_calls:
        assert isinstance(year, int), "Year should be an integer"
        assert isinstance(month, int), "Month should be an integer"
        assert 1 <= month <= 12, f"Month should be between 1 and 12, got {month}"


def test_heterogeneous_investment_effectiveness(datawrapper):
    """Test simulation with heterogeneous investment effectiveness across firms.

    This tests that we can set different investment effectiveness parameters
    for different firms and that the simulation runs correctly.
    """
    configuration = SimulationConfiguration(country_configurations={"FRA": CountryConfiguration()})
    configuration.seed = 0

    # Get n_firms from the configuration
    n_firms = configuration.country_configurations["FRA"].firms.n_firms

    # Set up productivity investment with heterogeneous parameters
    # Create a list of investment effectiveness values that vary across firms
    # Low effectiveness for first third, medium for middle third, high for last third
    investment_effectiveness_list = []
    for i in range(n_firms):
        if i < n_firms // 3:
            investment_effectiveness_list.append(0.05)  # Low
        elif i < 2 * n_firms // 3:
            investment_effectiveness_list.append(0.10)  # Medium
        else:
            investment_effectiveness_list.append(0.15)  # High

    # Configure productivity investment planner with heterogeneous parameters
    configuration.country_configurations[
        "FRA"
    ].firms.functions.productivity_investment_planner.name = "SimpleProductivityInvestmentPlanner"
    configuration.country_configurations["FRA"].firms.functions.productivity_investment_planner.parameters = {
        "n_firms": n_firms,
        "hurdle_rate": 0.10,  # Uniform
        "investment_effectiveness": investment_effectiveness_list,  # Heterogeneous
        "investment_elasticity": 0.3,  # Uniform
        "max_investment_fraction": 0.15,
        "investment_propensity": 0.5,
    }

    # Enable TFP growth to see the effects
    configuration.country_configurations["FRA"].firms.functions.productivity_growth.name = "SimpleTFPGrowth"
    configuration.country_configurations["FRA"].firms.functions.productivity_growth.parameters = {
        "investment_effectiveness": 0.1
    }
    configuration.country_configurations["FRA"].firms.parameters.tfp_base_growth_rate = 0.001

    # Create and run simulation
    simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=configuration)

    # Verify heterogeneous parameters were set correctly
    planner = simulation.countries["FRA"].firms.functions["productivity_investment_planner"]
    assert isinstance(planner.investment_effectiveness, np.ndarray)
    assert len(planner.investment_effectiveness) == n_firms
    # Check that we have the three different values
    unique_values = np.unique(planner.investment_effectiveness)
    assert len(unique_values) == 3
    assert np.allclose(sorted(unique_values), [0.05, 0.10, 0.15])

    # Store initial TFP
    initial_tfp = simulation.countries["FRA"].firms.states["tfp_multiplier"].copy()

    # Run simulation for several periods
    for _ in range(10):
        simulation.iterate()

    # Get final TFP
    final_tfp = simulation.countries["FRA"].firms.states["tfp_multiplier"]

    # Verify simulation ran successfully
    assert np.all(final_tfp >= initial_tfp), "TFP should not decrease"
    assert np.all(np.isfinite(final_tfp)), "TFP should be finite"

    # Check that at least some firms invested (if there was investment opportunity)
    if len(simulation.countries["FRA"].firms.ts.executed_productivity_investment) > 0:
        total_investment_history = simulation.countries["FRA"].firms.ts.executed_productivity_investment
        total_invested = sum(inv.sum() for inv in total_investment_history)
        # Just verify no errors - investment amount depends on profitability
        assert total_invested >= 0
