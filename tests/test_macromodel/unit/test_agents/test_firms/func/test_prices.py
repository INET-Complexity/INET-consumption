import numpy as np
import pandas as pd
import pytest

from macro_data.readers.exo_prices.exo_prices_reader import SectorExoPrices, SectorExoPricesReader
from macromodel.agents.firms.func.prices import (
    DefaultPriceSetter,
    SectorExogenousPriceSetter,
    SectorMarkupMarginalCostPriceSetter,
)

N_FIRMS = 4
INDUSTRIES = ["B05a", "B05a", "C19", "C19"]

DEFAULT_PARAMS = dict(
    price_setting_noise_std=0.0,
    price_setting_speed_gf=0.0,
    price_setting_speed_dp=0.0,
    price_setting_speed_cp=0.0,
)

PRICE_KWARGS = dict(
    prev_prices=np.ones(N_FIRMS),
    current_estimated_ppi_inflation=0.0,
    excess_demand=np.zeros(N_FIRMS),
    inventories=np.zeros(N_FIRMS),
    production=np.ones(N_FIRMS),
    prev_average_good_prices=np.array([10.0, 20.0]),
    prev_firm_prices=np.ones(N_FIRMS),
    prev_supply=np.ones(N_FIRMS),
    prev_demand=np.ones(N_FIRMS),
    current_firm_sectors=np.array([0, 0, 1, 1]),
    curr_unit_costs=np.ones(N_FIRMS),
    prev_unit_costs=np.ones(N_FIRMS),
    ppi_during=np.ones(N_FIRMS),
    current_time=4,
)


def _make_exo_prices(industries: list[str]) -> SectorExoPrices:
    """Price path that rises from 1.0 in 2013 to 2.0 in 2030."""
    df = pd.DataFrame({industry: [1.0, 2.0] for industry in industries}, index=[2013, 2030])
    reader = SectorExoPricesReader(prices=df)
    exo = SectorExoPrices.from_reader(reader, initial_year=2014)
    exo.initial_model_prices = np.ones(N_FIRMS)
    return exo


def test_price_setting_speeds_are_clipped_to_unit_interval():
    setter = DefaultPriceSetter(
        price_setting_noise_std=0.0,
        price_setting_speed_gf=2.0,
        price_setting_speed_dp=-1.0,
        price_setting_speed_cp=2.0,
    )

    prices = setter.compute_price(
        prev_prices=np.array([10.0]),
        current_estimated_ppi_inflation=0.1,
        excess_demand=np.array([0.0]),
        inventories=np.array([0.0]),
        production=np.array([0.0]),
        prev_average_good_prices=np.array([20.0]),
        prev_firm_prices=np.array([10.0]),
        prev_supply=np.array([10.0]),
        prev_demand=np.array([20.0]),
        current_firm_sectors=np.array([0]),
        curr_unit_costs=np.array([22.0]),
        prev_unit_costs=np.array([20.0]),
        ppi_during=np.array([0.0]),
        current_time=1,
    )

    np.testing.assert_allclose(prices, np.array([12.1]))


def test_default_price_setter_ignores_markup_rule_parameters():
    setter = DefaultPriceSetter(
        orbis_markup_path="unused.csv",
        markup_year=2014,
        markup_central_column="mu_op_weighted_median",
        markup_lower_column="mu_op_median_interval_low",
        markup_upper_column="mu_op_median_interval_high",
        markup_corridor_relative_cap=1.0,
        mc_smoothing_horizon=4,
        ac_smoothing_horizon=8,
        normal_output_smoothing_horizon=8,
        normal_output_capital_floor_lambda=0.25,
        demand_pull_speed=1.0,
        ac_floor_share=1.0,
        initial_cost_normalization_mode="output_weighted_robust_gap",
        initial_cost_normalization_lower_quantile=0.01,
        initial_cost_normalization_upper_quantile=0.99,
        initial_cost_normalization_min_factor=0.5,
        initial_cost_normalization_max_factor=2.0,
        initial_cost_normalization_min_valid_weight_share=0.5,
    )

    assert setter.price_setting_noise_std == pytest.approx(0.05)
    assert setter.price_setting_speed_gf == pytest.approx(1.0)
    assert setter.price_setting_speed_dp == pytest.approx(0.0)
    assert setter.price_setting_speed_cp == pytest.approx(0.0)


def test_default_price_setter_rejects_unknown_parameters():
    with pytest.raises(TypeError, match="unexpected price parameter"):
        DefaultPriceSetter(not_a_price_parameter=1)


class TestSectorExoPricesReader:
    def test_read_from_csv(self, tmp_path):
        csv = "year,B05a,C19\n2013,100.0,80.0\n2014,110.0,90.0\n2030,130.0,110.0\n"
        path = tmp_path / "firm_prices.csv"
        path.write_text(csv)

        reader = SectorExoPricesReader.read_from_raw_data(path)

        assert reader.prices is not None
        assert list(reader.prices.columns) == ["B05a", "C19"]
        assert reader.prices.loc[2014, "B05a"] == pytest.approx(110.0)

    def test_missing_file_returns_none(self, tmp_path):
        reader = SectorExoPricesReader.read_from_raw_data(tmp_path / "missing.csv")
        assert reader.prices is None

    def test_from_reader_copies_dataframe(self):
        df = pd.DataFrame({"B05a": [1.0, 2.0]}, index=[2013, 2030])
        exo = SectorExoPrices.from_reader(SectorExoPricesReader(prices=df), initial_year=2015)

        assert exo.prices is df
        assert exo.initial_year == 2015
        assert exo.initial_model_prices is None


class TestSectorExogenousPriceSetter:
    def _make_setter(self) -> SectorExogenousPriceSetter:
        setter = SectorExogenousPriceSetter(**DEFAULT_PARAMS)
        setter.overriden_industries = INDUSTRIES
        return setter

    def test_no_exo_prices_matches_default(self):
        default = DefaultPriceSetter(**DEFAULT_PARAMS)
        setter = self._make_setter()

        assert np.allclose(default.compute_price(**PRICE_KWARGS), setter.compute_price(**PRICE_KWARGS))

    def test_overrides_only_target_industry(self):
        setter = self._make_setter()
        setter.firm_exo_prices = _make_exo_prices(["B05a"])

        default_prices = DefaultPriceSetter(**DEFAULT_PARAMS).compute_price(**PRICE_KWARGS)
        result = setter.compute_price(**PRICE_KWARGS)

        assert result[0] != pytest.approx(default_prices[0])
        assert result[1] != pytest.approx(default_prices[1])
        assert result[2] == pytest.approx(default_prices[2])
        assert result[3] == pytest.approx(default_prices[3])

    def test_overrides_all_industries(self):
        setter = self._make_setter()
        setter.firm_exo_prices = _make_exo_prices(["B05a", "C19"])

        default_prices = DefaultPriceSetter(**DEFAULT_PARAMS).compute_price(**PRICE_KWARGS)
        result = setter.compute_price(**PRICE_KWARGS)

        assert not np.allclose(result, default_prices)
        assert pytest.approx(result[0]) == result[1] == result[2] == result[3]

    def test_unknown_industry_in_file_is_ignored(self):
        setter = self._make_setter()
        setter.firm_exo_prices = _make_exo_prices(["UNKNOWN_SECTOR"])

        default_prices = DefaultPriceSetter(**DEFAULT_PARAMS).compute_price(**PRICE_KWARGS)
        result = setter.compute_price(**PRICE_KWARGS)

        assert np.allclose(result, default_prices)

    def test_industry_level_initial_prices_are_aligned_to_firms(self):
        setter = self._make_setter()
        setter.firm_exo_prices = _make_exo_prices(["B05a", "C19"])
        setter.firm_exo_prices.initial_model_prices = np.array([10.0, 20.0])

        result = setter.compute_price(**PRICE_KWARGS)

        assert result[0] == pytest.approx(result[1])
        assert result[2] == pytest.approx(result[3])
        assert result[2] == pytest.approx(2 * result[0])

    def test_same_class_update_preserves_loaded_exogenous_price_state(self):
        setter = self._make_setter()
        firm_exo_prices = _make_exo_prices(["B05a"])
        setter.firm_exo_prices = firm_exo_prices
        setter.overriden_industries = INDUSTRIES

        setter.update_parameters_from_config(
            {
                "price_setting_noise_std": 0.1,
                "price_setting_speed_gf": 0.2,
                "price_setting_speed_dp": 0.3,
                "price_setting_speed_cp": 0.4,
            }
        )

        assert setter.firm_exo_prices is firm_exo_prices
        assert setter.overriden_industries == INDUSTRIES
        assert setter.price_setting_noise_std == pytest.approx(0.1)


def _write_markup_csv(path):
    path.write_text(
        "year,nace_rev_2_main_section,sum_weights,mu_op_weighted_median,"
        "mu_op_median_interval_low,mu_op_median_interval_high\n"
        "2014,A - Agriculture,1,1.20,1.00,1.50\n"
        "2014,B - Mining,1,2.00,1.50,3.00\n"
    )


class TestSectorMarkupMarginalCostPriceSetter:
    def _make_setter(self, tmp_path, **kwargs):
        path = tmp_path / "markups.csv"
        _write_markup_csv(path)
        params = dict(
            orbis_markup_path=str(path),
            markup_year=2014,
            industry_to_nace_main_section={"0": "A", "1": "B"},
            mc_smoothing_horizon=1,
            ac_smoothing_horizon=1,
            normal_output_smoothing_horizon=1,
            demand_pull_speed=1.0,
        )
        params.update(kwargs)
        return SectorMarkupMarginalCostPriceSetter(**params)

    @staticmethod
    def _cost_kwargs(n: int, mc: float | np.ndarray = 10.0, ac: float | np.ndarray | None = None):
        mc_values = np.full(n, mc, dtype=float) if np.isscalar(mc) else np.asarray(mc, dtype=float)
        ac_values = mc_values if ac is None else (np.full(n, ac, dtype=float) if np.isscalar(ac) else np.asarray(ac))
        return dict(
            pricing_material_mc=mc_values,
            pricing_effective_labour_inputs=np.ones(n),
            pricing_normal_output=np.ones(n),
            pricing_depreciation_unit_cost=ac_values - mc_values,
            wage_obligation_preview=np.zeros(n),
            producer_tax_rates=np.zeros(n),
            prev_mc_smooth=mc_values,
            prev_ac_smooth=ac_values,
            prev_normal_output=np.ones(n),
        )

    def test_tight_slack_pressure_moves_markup_regardless_of_price_position(self, tmp_path):
        setter = self._make_setter(tmp_path)
        kwargs = PRICE_KWARGS | dict(
            prev_prices=np.array([12.0, 8.0, 40.0, 10.0]),
            prev_firm_prices=np.array([12.0, 8.0, 40.0, 10.0]),
            prev_average_good_prices=np.array([10.0, 20.0]),
            prev_supply=np.array([1.0, 2.0, 1.0, 2.0]),
            prev_demand=np.array([2.0, 1.0, 2.0, 1.0]),
            curr_unit_costs=np.array([10.0, 10.0, 10.0, 10.0]),
            prev_unit_costs=np.array([10.0, 10.0, 10.0, 10.0]),
            **self._cost_kwargs(4, mc=10.0),
        )

        prices = setter.compute_price(**kwargs)

        np.testing.assert_allclose(prices, np.array([15.0, 11.0, 30.0, 17.5]))
        np.testing.assert_allclose(setter.last_pricing_markup_mu, np.array([1.5, 1.1, 3.0, 1.75]))
        np.testing.assert_allclose(
            setter.last_pricing_gate_state,
            np.array(
                [
                    setter.GATE_EXPENSIVE_TIGHT,
                    setter.GATE_CHEAP_SLACK,
                    setter.GATE_EXPENSIVE_TIGHT,
                    setter.GATE_CHEAP_SLACK,
                ],
                dtype=float,
            ),
        )

    def test_cheap_tight_moves_markup_up(self, tmp_path):
        setter = self._make_setter(tmp_path)
        kwargs = PRICE_KWARGS | dict(
            prev_prices=np.array([8.0]),
            prev_firm_prices=np.array([8.0]),
            prev_average_good_prices=np.array([10.0]),
            current_firm_sectors=np.array([0]),
            prev_supply=np.array([1.0]),
            prev_demand=np.array([2.0]),
            curr_unit_costs=np.array([10.0]),
            prev_unit_costs=np.array([10.0]),
            **self._cost_kwargs(1, mc=10.0),
        )

        prices = setter.compute_price(**kwargs)

        assert prices[0] == pytest.approx(15.0)
        assert setter.last_pricing_markup_mu[0] == pytest.approx(1.5)
        assert setter.last_pricing_gate_state[0] == setter.GATE_CHEAP_TIGHT

    def test_expensive_slack_moves_markup_down(self, tmp_path):
        setter = self._make_setter(tmp_path)
        kwargs = PRICE_KWARGS | dict(
            prev_prices=np.array([12.0]),
            prev_firm_prices=np.array([12.0]),
            prev_average_good_prices=np.array([10.0]),
            current_firm_sectors=np.array([0]),
            prev_supply=np.array([2.0]),
            prev_demand=np.array([1.0]),
            curr_unit_costs=np.array([10.0]),
            prev_unit_costs=np.array([10.0]),
            **self._cost_kwargs(1, mc=10.0),
        )

        prices = setter.compute_price(**kwargs)

        assert prices[0] == pytest.approx(11.0)
        assert setter.last_pricing_markup_mu[0] == pytest.approx(1.1)
        assert setter.last_pricing_gate_state[0] == setter.GATE_EXPENSIVE_SLACK

    def test_expensive_tight_moves_markup_up(self, tmp_path):
        setter = self._make_setter(tmp_path)
        kwargs = PRICE_KWARGS | dict(
            prev_prices=np.array([12.0]),
            prev_firm_prices=np.array([12.0]),
            prev_average_good_prices=np.array([10.0]),
            current_firm_sectors=np.array([0]),
            prev_supply=np.array([1.0]),
            prev_demand=np.array([2.0]),
            curr_unit_costs=np.array([10.0]),
            prev_unit_costs=np.array([10.0]),
            **self._cost_kwargs(1, mc=10.0),
        )

        prices = setter.compute_price(**kwargs)

        assert prices[0] == pytest.approx(15.0)
        assert setter.last_pricing_markup_mu[0] == pytest.approx(1.5)
        assert setter.last_pricing_gate_state[0] == setter.GATE_EXPENSIVE_TIGHT

    def test_cheap_slack_moves_markup_down(self, tmp_path):
        setter = self._make_setter(tmp_path)
        kwargs = PRICE_KWARGS | dict(
            prev_prices=np.array([8.0]),
            prev_firm_prices=np.array([8.0]),
            prev_average_good_prices=np.array([10.0]),
            current_firm_sectors=np.array([0]),
            prev_supply=np.array([2.0]),
            prev_demand=np.array([1.0]),
            curr_unit_costs=np.array([10.0]),
            prev_unit_costs=np.array([10.0]),
            **self._cost_kwargs(1, mc=10.0),
        )

        prices = setter.compute_price(**kwargs)

        assert prices[0] == pytest.approx(11.0)
        assert setter.last_pricing_markup_mu[0] == pytest.approx(1.1)
        assert setter.last_pricing_gate_state[0] == setter.GATE_CHEAP_SLACK

    def test_demand_pull_speed_is_clipped(self, tmp_path):
        setter = self._make_setter(tmp_path, demand_pull_speed=2.0)
        assert setter.demand_pull_speed == 1.0
        setter = self._make_setter(tmp_path, demand_pull_speed=-1.0)
        assert setter.demand_pull_speed == 0.0

    def test_relative_corridor_cap_limits_upper_bound(self, tmp_path):
        path = tmp_path / "wide.csv"
        path.write_text(
            "year,nace_rev_2_main_section,sum_weights,mu_op_weighted_median,"
            "mu_op_median_interval_low,mu_op_median_interval_high\n"
            "2014,A - Agriculture,1,1.00,0.50,3.00\n"
        )

        setter = SectorMarkupMarginalCostPriceSetter(
            orbis_markup_path=str(path),
            industry_to_nace_main_section={"0": "A"},
            markup_corridor_relative_cap=1.0,
        )

        assert setter.markup_upper_by_industry[0] == pytest.approx(2.0)

    def test_smooths_only_valid_current_mc(self, tmp_path):
        setter = self._make_setter(tmp_path, mc_smoothing_horizon=4, ac_smoothing_horizon=4)
        prices = setter.compute_price(
            **(
                PRICE_KWARGS
                | dict(
                    prev_prices=np.array([12.0, 12.0]),
                    prev_firm_prices=np.array([12.0, 12.0]),
                    prev_average_good_prices=np.array([10.0]),
                    prev_supply=np.array([1.0, 1.0]),
                    prev_demand=np.array([1.0, 1.0]),
                    current_firm_sectors=np.array([0, 0]),
                    curr_unit_costs=np.array([20.0, 0.0]),
                    prev_unit_costs=np.array([10.0, 10.0]),
                    pricing_material_mc=np.array([20.0, np.nan]),
                    pricing_effective_labour_inputs=np.ones(2),
                    pricing_normal_output=np.ones(2),
                    pricing_depreciation_unit_cost=np.zeros(2),
                    wage_obligation_preview=np.zeros(2),
                    producer_tax_rates=np.zeros(2),
                    prev_mc_smooth=np.array([10.0, 10.0]),
                    prev_ac_smooth=np.array([10.0, 10.0]),
                    prev_normal_output=np.ones(2),
                )
            )
        )

        np.testing.assert_allclose(setter.last_pricing_mc_smooth, np.array([14.0, 10.0]))
        np.testing.assert_allclose(prices, np.array([16.8, 12.0]))

    def test_wage_preview_enters_mc_through_effective_labour_inputs(self, tmp_path):
        setter = self._make_setter(tmp_path)
        prices = setter.compute_price(
            **(
                PRICE_KWARGS
                | dict(
                    prev_prices=np.array([12.0]),
                    prev_firm_prices=np.array([12.0]),
                    prev_average_good_prices=np.array([10.0]),
                    prev_supply=np.array([1.0]),
                    prev_demand=np.array([1.0]),
                    current_firm_sectors=np.array([0]),
                    pricing_material_mc=np.array([10.0]),
                    pricing_effective_labour_inputs=np.array([10.0]),
                    pricing_normal_output=np.array([5.0]),
                    pricing_depreciation_unit_cost=np.array([0.0]),
                    wage_obligation_preview=np.array([20.0]),
                    producer_tax_rates=np.array([0.0]),
                    prev_mc_smooth=np.array([14.0]),
                    prev_ac_smooth=np.array([14.0]),
                    prev_normal_output=np.array([5.0]),
                )
            )
        )

        np.testing.assert_allclose(setter.last_pricing_labour_mc, np.array([2.0]))
        np.testing.assert_allclose(setter.last_pricing_mc, np.array([12.0]))
        np.testing.assert_allclose(prices, np.array([14.4]))

    def test_realised_production_and_accounting_uc_do_not_anchor_price(self, tmp_path):
        setter = self._make_setter(tmp_path)
        prices = setter.compute_price(
            **(
                PRICE_KWARGS
                | dict(
                    prev_prices=np.array([12.0]),
                    prev_firm_prices=np.array([12.0]),
                    prev_average_good_prices=np.array([10.0]),
                    prev_supply=np.array([1.0]),
                    prev_demand=np.array([1.0]),
                    current_firm_sectors=np.array([0]),
                    production=np.array([1e-9]),
                    curr_unit_costs=np.array([1e9]),
                    prev_unit_costs=np.array([1e9]),
                    pricing_material_mc=np.array([10.0]),
                    pricing_effective_labour_inputs=np.array([5.0]),
                    pricing_normal_output=np.array([5.0]),
                    pricing_depreciation_unit_cost=np.array([0.0]),
                    wage_obligation_preview=np.array([0.0]),
                    producer_tax_rates=np.array([0.0]),
                    prev_mc_smooth=np.array([10.0]),
                    prev_ac_smooth=np.array([10.0]),
                    prev_normal_output=np.array([5.0]),
                )
            )
        )

        np.testing.assert_allclose(setter.last_pricing_mc, np.array([10.0]))
        np.testing.assert_allclose(prices, np.array([12.0]))

    def test_invalid_costs_fall_back_to_previous_price(self, tmp_path):
        setter = self._make_setter(tmp_path)
        prices = setter.compute_price(
            **(
                PRICE_KWARGS
                | dict(
                    prev_prices=np.array([24.0]),
                    prev_firm_prices=np.array([24.0]),
                    prev_average_good_prices=np.array([10.0]),
                    prev_supply=np.array([1.0]),
                    prev_demand=np.array([1.0]),
                    current_firm_sectors=np.array([0]),
                    curr_unit_costs=np.array([0.0]),
                    prev_unit_costs=np.array([0.0]),
                    pricing_material_mc=np.array([np.nan]),
                    pricing_effective_labour_inputs=np.array([np.nan]),
                    pricing_normal_output=np.array([np.nan]),
                    pricing_depreciation_unit_cost=np.array([0.0]),
                    wage_obligation_preview=np.array([np.nan]),
                    producer_tax_rates=np.array([0.0]),
                    prev_mc_smooth=np.array([np.nan]),
                    prev_ac_smooth=np.array([np.nan]),
                    prev_normal_output=np.array([np.nan]),
                )
            )
        )

        assert prices[0] == pytest.approx(24.0)
        assert setter.last_pricing_fallback_code[0] == setter.FALLBACK_PREVIOUS_PRICE

    def test_ac_floor_can_bind(self, tmp_path):
        setter = self._make_setter(tmp_path)
        prices = setter.compute_price(
            **(
                PRICE_KWARGS
                | dict(
                    prev_prices=np.array([24.0]),
                    prev_firm_prices=np.array([24.0]),
                    prev_average_good_prices=np.array([10.0]),
                    prev_supply=np.array([1.0]),
                    prev_demand=np.array([1.0]),
                    current_firm_sectors=np.array([0]),
                    curr_unit_costs=np.array([100.0]),
                    prev_unit_costs=np.array([100.0]),
                    **self._cost_kwargs(1, mc=100.0, ac=150.0),
                )
            )
        )

        assert prices[0] == pytest.approx(150.0)
        assert setter.last_pricing_ac_floor_binding[0] == 1.0

    def test_ac_above_sector_p95_uses_sector_median(self, tmp_path):
        setter = self._make_setter(tmp_path)
        prices = setter.compute_price(
            **(
                PRICE_KWARGS
                | dict(
                    prev_prices=np.array([96.0, 96.0, 96.0]),
                    prev_firm_prices=np.array([96.0, 96.0, 96.0]),
                    prev_average_good_prices=np.array([10.0]),
                    prev_supply=np.ones(3),
                    prev_demand=np.ones(3),
                    current_firm_sectors=np.array([0, 0, 0]),
                    curr_unit_costs=np.full(3, 80.0),
                    prev_unit_costs=np.full(3, 80.0),
                    **self._cost_kwargs(3, mc=np.array([80.0, 80.0, 80.0]), ac=np.array([100.0, 110.0, 1000.0])),
                )
            )
        )

        assert prices[2] == pytest.approx(110.0)
        assert setter.last_pricing_ac_fallback_binding[2] == 1.0

    def test_initial_cost_normalization_uses_initial_output_weighted_gap(self, tmp_path):
        setter = self._make_setter(
            tmp_path,
            initial_cost_normalization_mode="output_weighted_robust_gap",
            initial_cost_normalization_lower_quantile=0.0,
            initial_cost_normalization_upper_quantile=1.0,
        )
        prices = setter.compute_price(
            **(
                PRICE_KWARGS
                | dict(
                    prev_prices=np.array([24.0, 12.0, 12.0]),
                    prev_firm_prices=np.array([24.0, 12.0, 12.0]),
                    prev_average_good_prices=np.array([10.0]),
                    prev_supply=np.ones(3),
                    prev_demand=np.ones(3),
                    current_firm_sectors=np.array([0, 0, 0]),
                    curr_unit_costs=np.full(3, 10.0),
                    prev_unit_costs=np.full(3, 10.0),
                    initial_output_weights=np.array([100.0, 1.0, 1.0]),
                    **self._cost_kwargs(3, mc=10.0),
                )
            )
        )

        np.testing.assert_allclose(prices, np.full(3, 24.0))
        assert setter.initial_cost_normalization_factor == pytest.approx(2.0)
        assert setter.initial_cost_normalization_status == setter.NORMALIZATION_STATUS_APPLIED
        np.testing.assert_allclose(setter.last_pricing_cost_normalization_raw_gap, np.array([2.0, 1.0, 1.0]))
        np.testing.assert_allclose(setter.last_pricing_cost_normalization_factor, np.full(3, 2.0))
        np.testing.assert_allclose(setter.last_pricing_mc_smooth, np.full(3, 20.0))

    def test_initial_cost_normalization_is_reused_after_first_call(self, tmp_path):
        setter = self._make_setter(
            tmp_path,
            initial_cost_normalization_mode="output_weighted_robust_gap",
            initial_cost_normalization_lower_quantile=0.0,
            initial_cost_normalization_upper_quantile=1.0,
        )
        kwargs = PRICE_KWARGS | dict(
            prev_prices=np.array([24.0, 12.0, 12.0]),
            prev_firm_prices=np.array([24.0, 12.0, 12.0]),
            prev_average_good_prices=np.array([10.0]),
            prev_supply=np.ones(3),
            prev_demand=np.ones(3),
            current_firm_sectors=np.array([0, 0, 0]),
            curr_unit_costs=np.full(3, 10.0),
            prev_unit_costs=np.full(3, 10.0),
            initial_output_weights=np.array([100.0, 1.0, 1.0]),
            **self._cost_kwargs(3, mc=10.0),
        )
        setter.compute_price(**kwargs)

        prices = setter.compute_price(
            **(
                kwargs
                | dict(
                    prev_prices=np.full(3, 12.0),
                    prev_firm_prices=np.full(3, 12.0),
                    initial_output_weights=np.array([1.0, 100.0, 100.0]),
                )
            )
        )

        np.testing.assert_allclose(prices, np.full(3, 24.0))
        assert setter.initial_cost_normalization_factor == pytest.approx(2.0)
        assert np.all(np.isnan(setter.last_pricing_cost_normalization_raw_gap))

    def test_initial_cost_normalization_uses_previous_pre_tax_price(self, tmp_path):
        setter = self._make_setter(
            tmp_path,
            initial_cost_normalization_mode="output_weighted_robust_gap",
            initial_cost_normalization_lower_quantile=0.0,
            initial_cost_normalization_upper_quantile=1.0,
            initial_cost_normalization_max_factor=20.0,
        )
        prices = setter.compute_price(
            **(
                PRICE_KWARGS
                | dict(
                    prev_prices=np.array([120.0 / 0.9]),
                    prev_firm_prices=np.array([120.0 / 0.9]),
                    prev_average_good_prices=np.array([100.0]),
                    prev_supply=np.ones(1),
                    prev_demand=np.ones(1),
                    current_firm_sectors=np.array([0]),
                    curr_unit_costs=np.array([10.0]),
                    prev_unit_costs=np.array([10.0]),
                    initial_output_weights=np.array([1.0]),
                    **(self._cost_kwargs(1, mc=10.0) | {"producer_tax_rates": np.array([0.10])}),
                )
            )
        )

        assert setter.initial_cost_normalization_factor == pytest.approx(10.0)
        assert prices[0] == pytest.approx(120.0 / 0.9)

    def test_initial_cost_normalization_falls_back_without_valid_weights(self, tmp_path):
        setter = self._make_setter(
            tmp_path,
            initial_cost_normalization_mode="output_weighted_robust_gap",
        )
        prices = setter.compute_price(
            **(
                PRICE_KWARGS
                | dict(
                    prev_prices=np.array([24.0]),
                    prev_firm_prices=np.array([24.0]),
                    prev_average_good_prices=np.array([10.0]),
                    prev_supply=np.ones(1),
                    prev_demand=np.ones(1),
                    current_firm_sectors=np.array([0]),
                    curr_unit_costs=np.array([10.0]),
                    prev_unit_costs=np.array([10.0]),
                    initial_output_weights=np.array([0.0]),
                    **self._cost_kwargs(1, mc=10.0),
                )
            )
        )

        assert prices[0] == pytest.approx(12.0)
        assert setter.initial_cost_normalization_factor == pytest.approx(1.0)
        assert setter.initial_cost_normalization_status == setter.NORMALIZATION_STATUS_INVALID

    def test_initial_cost_normalization_reports_low_valid_weight_share(self, tmp_path):
        setter = self._make_setter(
            tmp_path,
            initial_cost_normalization_mode="output_weighted_robust_gap",
            initial_cost_normalization_min_valid_weight_share=0.5,
        )
        prices = setter.compute_price(
            **(
                PRICE_KWARGS
                | dict(
                    prev_prices=np.array([24.0, 24.0]),
                    prev_firm_prices=np.array([24.0, 24.0]),
                    prev_average_good_prices=np.array([10.0, 20.0]),
                    prev_supply=np.ones(2),
                    prev_demand=np.ones(2),
                    current_firm_sectors=np.array([0, 1]),
                    curr_unit_costs=np.array([10.0, 0.0]),
                    prev_unit_costs=np.array([10.0, 0.0]),
                    pricing_material_mc=np.array([10.0, np.nan]),
                    pricing_effective_labour_inputs=np.ones(2),
                    pricing_normal_output=np.ones(2),
                    pricing_depreciation_unit_cost=np.zeros(2),
                    wage_obligation_preview=np.zeros(2),
                    producer_tax_rates=np.zeros(2),
                    prev_mc_smooth=np.array([10.0, np.nan]),
                    prev_ac_smooth=np.array([10.0, np.nan]),
                    prev_normal_output=np.ones(2),
                    initial_output_weights=np.array([1.0, 100.0]),
                )
            )
        )

        np.testing.assert_allclose(prices, np.array([12.0, 24.0]))
        assert setter.initial_cost_normalization_factor == pytest.approx(1.0)
        assert setter.initial_cost_normalization_status == setter.NORMALIZATION_STATUS_LOW_VALID_WEIGHT

    def test_initial_cost_normalization_reports_clipped_factor(self, tmp_path):
        setter = self._make_setter(
            tmp_path,
            initial_cost_normalization_mode="output_weighted_robust_gap",
            initial_cost_normalization_lower_quantile=0.0,
            initial_cost_normalization_upper_quantile=1.0,
            initial_cost_normalization_max_factor=1.5,
        )
        prices = setter.compute_price(
            **(
                PRICE_KWARGS
                | dict(
                    prev_prices=np.array([24.0]),
                    prev_firm_prices=np.array([24.0]),
                    prev_average_good_prices=np.array([10.0]),
                    prev_supply=np.ones(1),
                    prev_demand=np.ones(1),
                    current_firm_sectors=np.array([0]),
                    curr_unit_costs=np.array([10.0]),
                    prev_unit_costs=np.array([10.0]),
                    initial_output_weights=np.array([1.0]),
                    **self._cost_kwargs(1, mc=10.0),
                )
            )
        )

        assert prices[0] == pytest.approx(18.0)
        assert setter.initial_cost_normalization_factor == pytest.approx(1.5)
        assert setter.initial_cost_normalization_status == setter.NORMALIZATION_STATUS_CLIPPED

    def test_positive_producer_tax_grosses_up_price(self, tmp_path):
        setter = self._make_setter(tmp_path)
        prices = setter.compute_price(
            **(
                PRICE_KWARGS
                | dict(
                    prev_prices=np.array([120.0]),
                    prev_firm_prices=np.array([120.0]),
                    prev_average_good_prices=np.array([100.0]),
                    prev_supply=np.array([1.0]),
                    prev_demand=np.array([1.0]),
                    current_firm_sectors=np.array([0]),
                    curr_unit_costs=np.array([100.0]),
                    prev_unit_costs=np.array([100.0]),
                    **(self._cost_kwargs(1, mc=100.0) | {"producer_tax_rates": np.array([0.10])}),
                )
            )
        )

        assert prices[0] == pytest.approx(120.0 / 0.9)

    def test_subsidy_does_not_mark_down_price(self, tmp_path):
        setter = self._make_setter(tmp_path)
        prices = setter.compute_price(
            **(
                PRICE_KWARGS
                | dict(
                    prev_prices=np.array([120.0]),
                    prev_firm_prices=np.array([120.0]),
                    prev_average_good_prices=np.array([100.0]),
                    prev_supply=np.array([1.0]),
                    prev_demand=np.array([1.0]),
                    current_firm_sectors=np.array([0]),
                    curr_unit_costs=np.array([100.0]),
                    prev_unit_costs=np.array([100.0]),
                    **(self._cost_kwargs(1, mc=100.0) | {"producer_tax_rates": np.array([-0.10])}),
                )
            )
        )

        assert prices[0] == pytest.approx(120.0)

    def test_tax_rate_at_or_above_one_fails(self, tmp_path):
        setter = self._make_setter(tmp_path)
        with pytest.raises(ValueError, match="Producer tax rates"):
            setter.compute_price(
                **(
                    PRICE_KWARGS
                    | dict(
                        prev_prices=np.array([120.0]),
                        prev_firm_prices=np.array([120.0]),
                        prev_average_good_prices=np.array([100.0]),
                        prev_supply=np.array([1.0]),
                        prev_demand=np.array([1.0]),
                        current_firm_sectors=np.array([0]),
                        curr_unit_costs=np.array([100.0]),
                        prev_unit_costs=np.array([100.0]),
                        **(self._cost_kwargs(1, mc=100.0) | {"producer_tax_rates": np.array([1.0])}),
                    )
                )
            )

    def test_invalid_markup_configuration_fails(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text(
            "year,nace_rev_2_main_section,sum_weights,mu_op_weighted_median,"
            "mu_op_median_interval_low,mu_op_median_interval_high\n"
            "2014,A - Agriculture,1,1.00,1.20,1.50\n"
        )

        with pytest.raises(ValueError, match="markup_lower <= markup_central"):
            SectorMarkupMarginalCostPriceSetter(
                orbis_markup_path=str(path),
                industry_to_nace_main_section={"0": "A"},
            )

    def test_invalid_initial_cost_normalization_configuration_fails(self, tmp_path):
        with pytest.raises(ValueError, match="initial_cost_normalization_lower_quantile"):
            self._make_setter(
                tmp_path,
                initial_cost_normalization_mode="output_weighted_robust_gap",
                initial_cost_normalization_lower_quantile=0.9,
                initial_cost_normalization_upper_quantile=0.1,
            )

    def test_missing_orbis_file_fails(self, tmp_path):
        with pytest.raises(ValueError, match="Orbis markup file not found"):
            SectorMarkupMarginalCostPriceSetter(orbis_markup_path=str(tmp_path / "missing.csv"))
