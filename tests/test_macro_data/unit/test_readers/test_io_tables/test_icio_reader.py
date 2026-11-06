from functools import reduce
from itertools import product

import numpy as np
import pandas as pd
import pytest

from macro_data.readers.io_tables.icio_reader import move_housing_inputs, remove_rental_services


class TestICIOAggReader:
    def test__countries_agg(self, readers):
        assert set(readers.icio[2014].considered_countries) == {"FRA"}

    def test__industries(self, readers):
        assert len(readers.icio[2014].industries) == 18
        try:
            readers.icio[2014].iot.loc[readers.icio[2014].considered_countries[0]].loc[readers.icio[2014].industries]
        except KeyError:
            assert False, "Raised a keyerror"

    def test__industries_agg(self, readers):
        try:
            readers.icio[2014].iot.loc[readers.icio[2014].considered_countries[0]].loc[readers.icio[2014].industries]
        except KeyError:
            assert False, "Raised a keyerror"

    @pytest.mark.parametrize("country", ["FRA"])
    def test__output(self, readers, country):
        assert np.all(readers.icio[2014].get_total_output(country)) > 0

    @pytest.mark.parametrize(
        "country, symbol",
        product(["FRA"], ["Firm Fixed Capital Formation", "Household Consumption"]),
    )
    def test__column_allc(self, readers, country: str, symbol: str):
        assert len(readers.icio[2014].column_allc(country, symbol)) == len(readers.icio[2014].industries)

    def test__hh_consumption(self, readers):
        assert readers.icio[2014].get_hh_consumption_weights("FRA").sum() == pytest.approx(1)

    def test__imputed_rents_equal_the_published_imputed_rentals(self, readers):
        # Eurostat CPA_L68A for France 2014: 168,078 mio EUR per year, quarterly in the reader
        assert readers.icio[2014].imputed_rents["FRA"] == pytest.approx(168078e6 / 4)

    def test__actual_rents_are_what_was_removed_beyond_the_imputed_rents(self, readers):
        icio = readers.icio[2014]
        imputed, actual = icio.imputed_rents["FRA"], icio.actual_rents["FRA"]
        # Eurostat CPA_L68A + CPA_L68B for France 2014 is 222,148 mio EUR per year. The domestic cell is
        # smaller, so the removed actual rentals fall short of CPA_L68B (54,070 mio) but are positive.
        assert 0.0 < actual < 54070.27e6 / 4
        assert (imputed + actual) * 4 <= (168078e6 + 54070.27e6) * (1 + 1e-9)

    def test__govt_cons(self, readers):
        assert readers.icio[2014].govt_consumption_weights("FRA").sum() == pytest.approx(1)

    def test__import_export(self, readers):
        total_exports = reduce(
            lambda a, b: a + b,
            [readers.icio[2014].get_exports(country) for country in readers.icio[2014].considered_countries + ["ROW"]],
        )
        total_imports = reduce(
            lambda a, b: a + b,
            [readers.icio[2014].get_imports(country) for country in readers.icio[2014].considered_countries + ["ROW"]],
        )

        assert total_imports.sum() - total_exports.sum() == pytest.approx(0)

    # @pytest.mark.parametrize("datareader", [readers_provincial_can, readers, readers_disagg_can])
    def test__ii_use(self, all_readers):
        icio_reader = all_readers.icio[2014]
        countries = icio_reader.considered_countries
        countries_row = list(countries) + ["ROW"]

        index = icio_reader.iot.index
        ind_index = index[index.get_level_values(0).isin(countries_row)]

        use_table = icio_reader.iot.loc[ind_index, ind_index]

        supplied_ii = use_table.sum(axis=1)

        assert use_table.loc[(countries[0], "A01")].sum() == pytest.approx(supplied_ii.loc[(countries[0], "A01")])

        used_ii = use_table.sum(axis=0)

        assert np.allclose(icio_reader.iot.loc[("TOTAL", "Intermediate Inputs"), use_table.columns], used_ii, rtol=1e-2)

        supplied_ii /= icio_reader.yearly_factor

        for country in countries:
            assert np.allclose(
                icio_reader.get_intermediate_inputs_supply(country).sum(axis=1), supplied_ii.loc[country], rtol=1e-2
            )


class TestICIODisagg:
    def test__output(self, all_readers):
        icio = all_readers.icio[2014]
        output = icio.get_total_output("FRA")
        assert np.all(output > 0)
        output_series = icio.get_total_output_series("FRA")
        assert np.all(output_series > 0)
        icio.get_output_shares_dict("FRA")


@pytest.mark.parametrize(
    "cell, imputed_lcu, actual_lcu, expected",
    [
        (1_000.0, 200.0, 100.0, (100.0, 50.0)),  # rentals fit in the cell: both removed in full
        (120.0, 200.0, 100.0, (100.0, 20.0)),  # cell is smaller than the rentals: actual is squeezed first
        (40.0, 200.0, 100.0, (40.0, 0.0)),  # imputed rentals alone exceed the cell
        (0.0, 200.0, 100.0, (0.0, 0.0)),
    ],
)
def test__remove_rental_services_never_exceeds_the_cell(cell, imputed_lcu, actual_lcu, expected):
    imputed, actual = remove_rental_services(cell, imputed_lcu, actual_lcu, lcu_per_usd=2.0)
    assert (imputed, actual) == pytest.approx(expected)
    assert imputed + actual <= cell


def test_move_housing_inputs_keeps_totals_out_of_the_move():
    labels = [("FRA", "C"), ("FRA", "L"), ("TOTAL", "OUT")]
    columns = pd.MultiIndex.from_tuples(labels + [("FRA", "Household Consumption")])
    iot = pd.DataFrame(
        [[1.0, 10.0, 0.0, 5.0], [0.0, 20.0, 0.0, 2.0], [50.0, 100.0, 0.0, 0.0]],
        index=pd.MultiIndex.from_tuples(labels),
        columns=columns,
    )

    moved = move_housing_inputs(iot, "FRA", 0.5, ["C", "L"])

    assert moved.sum() == pytest.approx(15.0)
    assert moved["C"] == pytest.approx(5.0)
    assert iot.at[("FRA", "C"), ("FRA", "L")] == pytest.approx(5.0)
    assert iot.at[("FRA", "L"), ("FRA", "L")] == pytest.approx(10.0)
    assert iot.at[("FRA", "C"), ("FRA", "Household Consumption")] == pytest.approx(10.0)
    assert iot.at[("TOTAL", "OUT"), ("FRA", "L")] == pytest.approx(100.0)
