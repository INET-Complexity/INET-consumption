"""Fixed-stock rental price index (issue #197)."""

import numpy as np
import pytest

from macromodel.economy.economy import Economy
from macromodel.timeseries import TimeSeries


def make_economy():
    economy = Economy.__new__(Economy)
    economy.ts = TimeSeries(rpi=[1.0], rpi_inflation=[0.0])
    return economy


def test_rpi_compounds_the_growth_of_total_rent_over_the_same_properties():
    economy = make_economy()
    economy.compute_rental_price_index(np.array([9.5, 19.0]), np.array([10.0, 20.0]))
    economy.compute_rental_price_index(np.array([9.5, 20.9]), np.array([9.5, 19.0]))
    assert economy.ts.current("rpi_inflation")[0] == pytest.approx(30.4 / 28.5 - 1.0)
    assert economy.ts.current("rpi")[0] == pytest.approx(30.4 / 30.0)


def test_rpi_falls_when_rents_fall():
    economy = make_economy()
    economy.compute_rental_price_index(np.array([95.0]), np.array([100.0]))
    assert economy.ts.current("rpi")[0] == pytest.approx(0.95)


def test_rpi_is_flat_when_previous_rents_are_zero():
    economy = make_economy()
    economy.compute_rental_price_index(np.array([1.0]), np.array([0.0]))
    assert economy.ts.current("rpi")[0] == 1.0


def test_rpi_rejects_rents_of_different_property_sets():
    with pytest.raises(ValueError, match="same properties"):
        make_economy().compute_rental_price_index(np.array([1.0, 2.0]), np.array([1.0]))
