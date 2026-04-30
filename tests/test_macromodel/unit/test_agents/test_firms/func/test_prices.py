import numpy as np

from macromodel.agents.firms.func.prices import DefaultPriceSetter


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
