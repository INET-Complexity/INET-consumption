import numpy as np

from macromodel.agents.agent import Agent
from macromodel.configurations.goods_market_configuration import GoodsMarketConfiguration
from macromodel.markets.goods_market.func.clearing import WaterBucketGoodsMarketClearer
from macromodel.markets.goods_market.goods_market import GoodsMarket
from macromodel.markets.goods_market.value_type import ValueType
from macromodel.timeseries import TimeSeries


def _agent(country_name: str, buy_priority: int, sell_priority: int) -> Agent:
    return Agent(
        country_name=country_name,
        all_country_names=["FRA", "ROW"],
        n_industries=2,
        n_transactors_buy=1,
        n_transactors_sell=1,
        ts=TimeSeries(),
        states={},
        transactor_settings={
            "Buyer Value Type": ValueType.REAL,
            "Seller Value Type": ValueType.REAL,
            "Buyer Priority": buy_priority,
            "Seller Priority": sell_priority,
        },
    )


def _market() -> GoodsMarket:
    # Asymmetric priorities so that a buyer/seller label swap is observable.
    participants = {"FRA": [_agent("FRA", 1, 0)], "ROW": [_agent("ROW", 0, 1)]}
    n_industries = 2
    trade_props = np.full(2 * 2 * n_industries, 0.5)
    return GoodsMarket.from_data(
        n_industries=n_industries,
        configuration=GoodsMarketConfiguration(),
        goods_market_participants=participants,
        origin_trade_proportions=trade_props,
        destin_trade_proportions=trade_props,
        row_index=1,
    )


def _clearer(real_country_prioritisation: float) -> WaterBucketGoodsMarketClearer:
    return WaterBucketGoodsMarketClearer(
        real_country_prioritisation=real_country_prioritisation,
        prio_high_prio_buyers=False,
        prio_high_prio_sellers=False,
        prio_domestic_sellers=False,
        probability_keeping_previous_seller=0.0,
        price_temperature=2.0,
        trade_temperature=0.0,
        seller_selection_distribution_type="additive",
        seller_minimum_fill=0.9,
        buyer_minimum_fill_macro=0.0,
        buyer_minimum_fill_micro=0.95,
        deterministic=True,
        consider_trade_proportions=True,
        consider_buyer_priorities=True,
        additionally_available_factor=0.2,
        price_markup=0.2,
    )


class TestGoodsMarketPriorities:
    def test__buyer_priorities_come_from_buyer_priority_setting(self):
        market = _market()
        assert market.buyer_priorities["FRA"].tolist() == [1]
        assert market.buyer_priorities["ROW"].tolist() == [0]

    def test__seller_priorities_come_from_seller_priority_setting(self):
        market = _market()
        assert market.seller_priorities["FRA"].tolist() == [0]
        assert market.seller_priorities["ROW"].tolist() == [1]


class TestRealCountryPrioritisation:
    def test__stored_value_is_clamped_to_unit_interval(self):
        assert _clearer(10.0).real_country_prioritisation == 1.0
        assert _clearer(-3.0).real_country_prioritisation == 0.0
        assert _clearer(0.4).real_country_prioritisation == 0.4
