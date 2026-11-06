from types import SimpleNamespace

import pandas as pd
import pytest

from macro_data.readers.icio_sea_matching import _reconcile_value_added


def test_reconcile_scales_capital_stock_once_with_value_added():
    index = pd.MultiIndex.from_tuples([("FRA", "C"), ("FRA", "L")], names=["Country", "Industry"])
    df = pd.DataFrame(
        {
            "Value Added": [10.0, 20.0],
            "Labour Compensation": [6.0, 12.0],
            "Capital Compensation": [4.0, 8.0],
            "Capital Stock": [100.0, 200.0],
        },
        index=index,
    )
    new_value_added = pd.Series([5.0, 40.0], index=["C", "L"])
    icio = SimpleNamespace(
        get_value_added_series=lambda _: new_value_added,
        get_capital_scaling_value_added_series=lambda _: new_value_added,
    )
    sea = SimpleNamespace(df=df, industries=["C", "L"])

    _reconcile_value_added("FRA", icio, sea, yearly_factor=4.0)

    assert df["Capital Stock"].tolist() == pytest.approx([200.0, 1600.0])
    assert df["Capital Compensation"].tolist() == pytest.approx([8.0, 64.0])
