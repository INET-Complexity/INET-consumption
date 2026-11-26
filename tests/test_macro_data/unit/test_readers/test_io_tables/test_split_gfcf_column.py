import pandas as pd
import pytest

from macro_data.readers.io_tables.icio_reader import split_gfcf_column


def _iot():
    rows = pd.MultiIndex.from_tuples([("FRA", "A"), ("FRA", "F")])
    cols = pd.MultiIndex.from_tuples(
        [("FRA", "Fixed Capital Formation"), ("FRA", "Government Consumption"), ("FRA", "Household Consumption")]
    )
    return pd.DataFrame([[10.0, 5.0, 1.0], [30.0, 7.0, 2.0]], index=rows, columns=cols)


def test_government_gfcf_goes_to_firms_not_government_consumption():
    fractions = {"FRA": {"Firm": 0.5, "Household": 0.3, "Government": 0.2}}
    out = split_gfcf_column(["FRA"], ["A", "F"], fractions, _iot())

    assert out[("FRA", "Government Consumption")].tolist() == [5.0, 7.0]
    assert out[("FRA", "Firm Fixed Capital Formation")].tolist() == pytest.approx([7.0, 21.0])
    assert out[("FRA", "Household Fixed Capital Formation")].tolist() == pytest.approx([3.0, 9.0])
    assert ("FRA", "Fixed Capital Formation") not in out.columns


def test_total_gfcf_is_preserved():
    fractions = {"FRA": {"Firm": 0.5, "Household": 0.3, "Government": 0.2}}
    out = split_gfcf_column(["FRA"], ["A", "F"], fractions, _iot())
    total = out[("FRA", "Firm Fixed Capital Formation")] + out[("FRA", "Household Fixed Capital Formation")]
    assert total.sum() == pytest.approx(40.0)
