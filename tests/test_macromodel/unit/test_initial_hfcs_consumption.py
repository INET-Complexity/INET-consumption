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
        initialise_hfcs_consumption(c, np.array([0.6, 0.9]), legacy_fallback=False)
    np.testing.assert_array_equal(h.ts.current("consumption"), before)


def test_default_fallback_keeps_opening_goods_cash_and_housing_without_changing_income():
    c = country()
    h = c.households
    vat = 0.2
    old_goods = h.ts.current("consumption").copy()
    housing = h.ts.current("rent") + h.ts.current("rent_imputed")
    old_total = (1 + vat) * old_goods + housing
    old_cash = (1 + vat) * old_goods + h.ts.current("rent")
    h.ts.override_current("expected_behavioural_income_raw", np.array([0.0, 200.0]))
    income = h.ts.current("expected_behavioural_income_raw").copy()
    result = initialise_hfcs_consumption(c, np.array([0.6, 0.9]))
    np.testing.assert_array_equal(result.legacy_fallback, [True, False])
    assert h.ts.current("consumption")[0] == old_goods[0]
    assert h.ts.current("consumption_including_housing")[0] == old_total[0]
    assert h.ts.current("consumption_cash_expenditure")[0] == old_cash[0]
    np.testing.assert_array_equal(h.ts.current("expected_behavioural_income_raw"), income)
    np.testing.assert_allclose(h.ts.current("target_consumption").sum(axis=0), [66.0, 99.0])
    np.testing.assert_allclose(h.ts.current("target_consumption").sum(axis=1), result.goods)
    assert np.isnan(h.ts.current("initial_hfcs_final_c_y")[0])
    assert h.ts.current("initial_hfcs_goods_scale")[0] == 1.0
    opening = {name: np.array(rows[0], copy=True) for name, rows in h.ts.dicts.items()}
    for key in h.ts.dicts:
        h.ts.dicts[key].append(np.zeros_like(h.ts.current(key)))
    h.ts.reset()
    for key, value in opening.items():
        np.testing.assert_array_equal(h.ts.current(key), value)
