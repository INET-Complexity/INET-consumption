import numpy as np
import pandas as pd
import pytest

from macro_data.readers.exo_prices.exo_prices_reader import SectorExoPrices, SectorExoPricesReader
from macromodel.agents.firms.func.prices import DefaultPriceSetter, SectorExogenousPriceSetter

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
