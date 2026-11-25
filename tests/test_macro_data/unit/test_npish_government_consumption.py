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
