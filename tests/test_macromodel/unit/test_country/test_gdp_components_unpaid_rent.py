"""The GDP component listings account for unpaid rent, which is income-side GDP (landlord loss, social write-off)."""

from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.country.country import Country


def _aggregate_source(label):
    """A time series whose aggregate is a distinct, name-dependent vector."""
    return SimpleNamespace(
        ts=SimpleNamespace(get_aggregate=lambda name: np.full(3, float(sum(map(ord, label + name)))) + np.arange(3.0))
    )


def _stub():
    return SimpleNamespace(
        economy=_aggregate_source("economy"),
        central_government=_aggregate_source("government"),
        households=_aggregate_source("households"),
        government_entities=_aggregate_source("entities"),
        firms=_aggregate_source("firms"),
    )


@pytest.mark.parametrize("listing", [Country.gdp_debug_output, Country.gdp_components_df.fget])
def test_listing_has_unpaid_rent_rows_from_the_series_gdp_income_adds(listing):
    stub = _stub()
    frame = listing(stub)
    np.testing.assert_array_equal(
        frame["+Rent_Unpaid_Absorbed"], stub.households.ts.get_aggregate("total_rent_unpaid_absorbed")
    )
    np.testing.assert_array_equal(
        frame["+Social_Rent_Written_Off"], stub.central_government.ts.get_aggregate("social_rent_written_off")
    )
    np.testing.assert_array_equal(
        frame["memo_Rent_Unpaid_No_Counterparty"], stub.households.ts.get_aggregate("rent_unpaid_no_counterparty")
    )
    # They sit with the other income-side rent components, next to rent received.
    columns = list(frame.columns)
    assert columns.index("+Central_Government_Rent_Received") < columns.index("+Rent_Unpaid_Absorbed")
    assert columns.index("+Rent_Unpaid_Absorbed") < columns.index("+Social_Rent_Written_Off")
