"""NPISH final consumption is mapped to government, not household, consumption."""

import numpy as np
import pandas as pd
import pytest

from macro_data.readers.io_tables.mappings import ICIO_AGGREGATE, ICIO_AGGREGATE_INV, ICIO_ALL, ICIO_ALL_INV


@pytest.mark.parametrize("mapping", [ICIO_AGGREGATE, ICIO_ALL])
def test_npish_mapped_to_government(mapping):
    assert "NPISH" not in mapping["Household Consumption"]
    assert "NPISH" in mapping["Government Consumption"]


@pytest.mark.parametrize("inverse", [ICIO_AGGREGATE_INV, ICIO_ALL_INV])
def test_aggregation_moves_npish_and_preserves_row_totals(inverse):
    final_demand = ["HFCE", "NPISH", "DPABR", "GGFC", "GFCF", "INVNT"]
    rng = np.random.default_rng(0)
    raw = pd.DataFrame(rng.uniform(1, 10, (3, len(final_demand))), index=["A01", "C10T12", "L"], columns=final_demand)

    agg = raw.T.groupby(lambda code: inverse[code]).sum().T

    np.testing.assert_allclose(agg["Household Consumption"], raw[["HFCE", "DPABR"]].sum(axis=1))
    np.testing.assert_allclose(agg["Government Consumption"], raw[["GGFC", "NPISH"]].sum(axis=1))
    np.testing.assert_allclose(agg.sum(axis=1), raw.sum(axis=1))


def test_imf_growth_combines_npish_with_government():
    from macro_data.readers.economic_data.imf_reader import IMFReader

    quarters = [f"2014Q{q}" for q in range(1, 5)] + ["2015Q1"]
    suffix = ", Nominal, Domestic Currency"
    levels = {
        "Gross Domestic Product": [100, 110, 121, 133.1, 146.41],
        "Households Final Consumption Expenditure": [50, 55, 60, 65, 70],
        "Non-profit Institutions Serving Households (NPISHs) Final Consumption Expenditure": [10, 10, 10, 10, 10],
        "General Government Final Consumption Expenditure": [20, 30, 40, 50, 60],
    }
    rows = [
        {
            "Indicator Name": name + suffix,
            "Country Name": "France",
            "Country Code": "FR",
            "Indicator Code": "x",
            "Attribute": "Value",
            **dict(zip(quarters, vals)),
            "Base": None,
        }
        for name, vals in levels.items()
    ]
    reader = object.__new__(IMFReader)
    reader.data = {"international_financial_statistics": pd.DataFrame(rows)}

    growth = reader.get_na_growth_rates("FRA")

    hh = np.array(levels["Households Final Consumption Expenditure"][:4], dtype=float)
    gov = np.array(levels["General Government Final Consumption Expenditure"][:4], dtype=float) + 10
    np.testing.assert_allclose(growth["HH Cons"].values, hh[1:] / hh[:-1] - 1)
    np.testing.assert_allclose(growth["Gov Cons"].values, gov[1:] / gov[:-1] - 1)
    assert "NPISH Cons" not in growth.columns
