from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.timeseries import TimeSeries
from macromodel.util.initial_consumption import initialise_hfcs_consumption


def country():
    ts = TimeSeries(
        expected_behavioural_income_raw=np.array([100.0, 200.0]),
        expected_behavioural_income=np.array([120.0, 200.0]),
        industry_consumption=np.array([66.0, 99.0]),
        rent=np.array([0.0, 30.0]),
        rent_imputed=np.array([12.0, 0.0]),
        investment=np.array([[2.0, 3.0], [4.0, 5.0]]),
        consumption=np.array([82.5, 82.5]),
        consumption_cash_expenditure=np.zeros(2),
        consumption_including_housing=np.zeros(2),
        cacf_real_consumption_budget=np.zeros(2),
        target_consumption=np.zeros((2, 2)),
        realised_household_expenditure=np.zeros(2),
    )
    return SimpleNamespace(
        households=SimpleNamespace(ts=ts),
        central_government=SimpleNamespace(states={"Value-added Tax": 0.2}),
        economy=SimpleNamespace(initial_consumer_price_level=lambda: 2.0),
    )


def test_uses_unfloored_income_effective_vat_and_preserves_sector_aggregate():
    c = country()
    h = c.households
    result = initialise_hfcs_consumption(c, np.array([0.6, 0.9]))
    np.testing.assert_allclose(h.ts.current("consumption"), [40.0, 125.0])
    np.testing.assert_allclose(h.ts.current("consumption_including_housing"), [60.0, 180.0])
    np.testing.assert_allclose(h.ts.current("cacf_real_consumption_budget"), [30.0, 90.0])
    np.testing.assert_allclose(h.ts.current("target_consumption").sum(axis=0), [66.0, 99.0])
    np.testing.assert_allclose(h.ts.current("target_consumption").sum(axis=1), [40.0, 125.0])
    np.testing.assert_allclose(h.ts.current("initial_hfcs_candidate_total"), [60.0, 180.0])
    np.testing.assert_allclose(h.ts.current("realised_household_expenditure"), [45.0, 134.0])
    assert result.goods_scale == 1.0


def test_reset_restores_final_opening_budget_and_diagnostics():
    c = country()
    initialise_hfcs_consumption(c, np.array([0.6, 0.9]))
    h = c.households
    initial = {k: np.array(v[0], copy=True) for k, v in h.ts.dicts.items()}
    for k in h.ts.dicts:
        h.ts.dicts[k].append(np.zeros_like(h.ts.current(k)))
    h.ts.reset()
    for k, v in initial.items():
        np.testing.assert_array_equal(h.ts.current(k), v)


@pytest.mark.parametrize("failure", ["income", "housing", "advanced"])
def test_rejects_infeasible_or_advanced_states_without_partial_budget_write(failure):
    c = country()
    h = c.households
    before = h.ts.current("consumption").copy()
    if failure == "income":
        h.ts.current("expected_behavioural_income_raw")[0] = 0.0
    elif failure == "housing":
        h.ts.current("rent")[0] = 1000.0
    else:
        h.ts.dicts["consumption"].append(before.copy())
    with pytest.raises(ValueError):
        initialise_hfcs_consumption(c, np.array([0.6, 0.9]))
    np.testing.assert_array_equal(h.ts.current("consumption"), before)
