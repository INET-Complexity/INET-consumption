"""ARDL government consumption rule: dlnG_t = a + r1 dlnG_{t-1} + r2 dlnG_{t-2} + b * mean(dlnY_{t-5..t-12})."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macromodel.agents.government_entities.func.consumption import ARDLGovernmentConsumptionSetter
from macromodel.configurations.government_entities_configuration import GovernmentFunctions
from macromodel.country.country import Country
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import functions_from_model


def setter(**kwargs):
    """Fixed data intercept unless a test passes trend_window."""
    kwargs.setdefault("trend_window", None)
    return ARDLGovernmentConsumptionSetter(consistency=1.0, sectoral_weights="initial_fixed", **kwargs)


def target(s, prev, g, y, pi_e=0.0, posted=None, z=None):
    return s.compute_target_consumption(
        previous_desired_government_consumption=prev,
        model=None,
        historic_total_consumption=g,
        initial_good_prices=np.ones(len(prev)),
        current_good_prices=np.ones(len(prev)),
        expected_growth=0.0,
        expected_inflation=pi_e,
        current_time=1,
        exogenous_total_consumption=None,
        forecasting_window=60,
        historic_real_gdp_growth=y,
        historic_productivity_growth=z,
        posted_price_index_by_industry=np.ones(len(prev)) if posted is None else posted,
    )


def test_real_growth_follows_the_estimated_equation():
    s = setter()
    g = np.array([100.0, 101.0, 102.5])  # growth t-2 = ln(1.01), t-1 = ln(102.5/101)
    y = np.linspace(0.001, 0.016, 16)  # latest last
    expected = (
        s.intercept
        + s.own_lag_1 * np.log(102.5 / 101.0)
        + s.own_lag_2 * np.log(1.01)
        + s.output_coefficient * y[-12:-4].mean()
    )
    assert s.real_growth(g, y) == pytest.approx(expected)


def test_same_quarter_and_recent_gdp_growth_do_not_enter():
    s = setter()
    g = np.array([100.0, 100.0, 100.0])
    y = np.zeros(16)
    base = s.real_growth(g, y)
    y_recent = y.copy()
    y_recent[-4:] = 0.05  # growth t-1..t-4: outside the 5-12 window
    assert s.real_growth(g, y_recent) == pytest.approx(base)
    assert base == pytest.approx(s.intercept)


def test_long_run_growth_matches_the_closed_form():
    s = setter()
    gy = 0.004  # constant real GDP growth per quarter
    level, dg = [100.0, 100.0, 100.0], []
    for _ in range(400):
        x = s.real_growth(np.array(level[-3:]), np.full(12, gy))
        level.append(level[-1] * np.exp(x))
        dg.append(x)
    closed = (s.intercept + s.output_coefficient * gy) / (1 - s.own_lag_1 - s.own_lag_2)
    assert dg[-1] == pytest.approx(closed, rel=1e-6)


def test_first_target_is_previous_envelope_times_real_growth_and_ignores_expected_inflation():
    s = setter()
    prev, g, y = np.array([30.0, 50.0, 20.0]), np.array([100.0, 101.0, 102.0]), np.full(12, 0.003)
    out = target(s, prev, g, y, pi_e=0.05)
    assert out.sum() == pytest.approx(prev.sum() * np.exp(s.real_growth(g, y)))
    np.testing.assert_allclose(out / out.sum(), prev / prev.sum())


@pytest.mark.parametrize(
    "g, y, match",
    [
        (np.array([100.0, 101.0]), np.zeros(12), "three positive"),
        (np.array([100.0, 0.0, 101.0]), np.zeros(12), "three positive"),
        (np.array([100.0, 101.0, 102.0]), np.zeros(11), "12 finite"),
        (np.array([100.0, 101.0, 102.0]), np.r_[np.zeros(11), np.nan], "12 finite"),
    ],
)
def test_insufficient_or_invalid_history_raises(g, y, match):
    with pytest.raises(ValueError, match=match):
        setter().real_growth(g, y)


def test_missing_gdp_history_raises():
    with pytest.raises(ValueError, match="histories"):
        setter().compute_target_consumption(
            previous_desired_government_consumption=np.ones(2),
            model=None,
            historic_total_consumption=np.ones(3),
            initial_good_prices=np.ones(2),
            current_good_prices=np.ones(2),
            expected_growth=0.0,
            expected_inflation=0.0,
            current_time=1,
            exogenous_total_consumption=None,
            forecasting_window=60,
        )


def test_invalid_lag_window_raises():
    with pytest.raises(ValueError, match="output_lag"):
        setter(output_lag_start=6, output_lag_end=5)


def test_setter_is_loaded_from_government_model_config():
    cfg = GovernmentFunctions()
    cfg.consumption.name = "ARDLGovernmentConsumptionSetter"
    cfg.consumption.parameters["sectoral_weights"] = "initial_fixed"
    functions = functions_from_model(model=cfg, loc="macromodel.agents.government_entities")
    assert isinstance(functions["consumption"], ARDLGovernmentConsumptionSetter)


def test_country_splices_growth_rates_not_levels():
    country = Country.__new__(Country)
    country.forecasting_window = 60
    pre_levels = np.array([100.0, 101.0, 102.01])  # 1% growth twice
    country.exogenous = SimpleNamespace(national_accounts_before=pd.DataFrame({"Real GDP (Value)": pre_levels}))
    # Simulated real GDP at a different level scale (x10): the junction must not create a growth rate.
    country.economy = SimpleNamespace(ts=TimeSeries(gdp_output=np.array([1000.0]), ppi=np.array([1.0])))
    country.economy.ts.gdp_output.append(np.array([1020.0]))
    country.economy.ts.ppi.append(np.array([1.0]))
    out = country._historic_real_gdp_growth()
    np.testing.assert_allclose(out, [np.log(1.01), np.log(1.01), np.log(1.02)])


def test_data_intercept_equals_the_trend_loading_at_the_data_trend():
    s = ARDLGovernmentConsumptionSetter(consistency=1.0, sectoral_weights="initial_fixed")
    assert s.trend_window == 40
    loading = 1 - s.own_lag_1 - s.own_lag_2 - s.output_coefficient
    trend = s.intercept / loading  # implied data trend, per quarter
    assert 400 * trend == pytest.approx(1.545, abs=0.01)
    assert s.current_intercept(np.full(40, trend)) == pytest.approx(s.intercept)


def test_trend_intercept_uses_trailing_productivity_growth():
    s = setter(trend_window=40)
    loading = 1 - s.own_lag_1 - s.own_lag_2 - s.output_coefficient
    z = np.r_[np.full(20, 0.05), np.full(40, 0.002)]
    assert s.current_intercept(z) == pytest.approx(loading * 0.002)
    assert s.current_intercept(np.full(5, 0.001)) == pytest.approx(loading * 0.001)  # expanding early
    assert s.current_intercept(np.array([])) == pytest.approx(s.intercept)  # no history yet
    assert s.current_intercept(None) == pytest.approx(s.intercept)


def test_trend_window_must_be_positive():
    with pytest.raises(ValueError, match="trend_window"):
        setter(trend_window=0)


def test_steady_state_government_consumption_grows_with_trend_productivity():
    """If GDP grows at productivity growth z, G grows at z: a stable share in steady state."""
    for z in (0.0015, 0.004):
        s = setter(trend_window=40)
        prev, g, y, zz = np.array([60.0, 40.0]), np.array([100.0, 100.0, 100.0]), np.full(60, z), np.full(60, z)
        totals = [target(s, prev, g, y, z=zz).sum()]
        for _ in range(300):
            totals.append(target(s, prev, g, y, z=zz).sum())
        assert np.log(totals[-1] / totals[-2]) == pytest.approx(z, rel=1e-6)


def test_planned_growth_ignores_realised_consumption_after_the_first_call():
    y, z = np.full(60, 0.003), np.full(60, 0.002)
    a, b = setter(trend_window=40), setter(trend_window=40)
    prev, g = np.array([60.0, 40.0]), np.array([100.0, 100.0, 100.0])
    target(a, prev, g, y, z=z), target(b, prev, g, y, z=z)
    ta = target(a, prev, np.array([100.0, 100.0, 100.0]), y, z=z)
    tb = target(b, prev, np.array([100.0, 100.0, 80.0]), y, z=z)  # rationing/surprise cut b's realised G
    np.testing.assert_allclose(ta, tb)


def test_nominal_target_scales_with_posted_prices_not_expected_inflation():
    y = np.full(60, 0.003)
    a, b, c = (setter(trend_window=40) for _ in range(3))
    prev, g = np.array([60.0, 40.0]), np.array([100.0, 100.0, 100.0])
    ta = target(a, prev, g, y)
    tb = target(b, prev, g, y, posted=np.array([1.2, 1.2]))
    tc = target(c, prev, g, y, pi_e=0.10)
    np.testing.assert_allclose(tb, 1.2 * ta)
    np.testing.assert_allclose(tc, ta)
    np.testing.assert_allclose(ta / ta.sum(), [0.6, 0.4])


def test_basket_index_uses_the_sectoral_weights():
    s = setter(trend_window=40)
    prev, g, y = np.array([60.0, 40.0]), np.array([100.0, 100.0, 100.0]), np.full(60, 0.0)
    base = target(setter(trend_window=40), prev, g, y).sum()
    out = target(s, prev, g, y, posted=np.array([1.5, 1.0]))
    assert out.sum() == pytest.approx(base * (0.6 * 1.5 + 0.4 * 1.0))


def test_missing_posted_prices_raises():
    with pytest.raises(ValueError, match="posted price"):
        setter().compute_target_consumption(
            previous_desired_government_consumption=np.ones(2),
            model=None,
            historic_total_consumption=np.ones(3),
            initial_good_prices=np.ones(2),
            current_good_prices=np.ones(2),
            expected_growth=0.0,
            expected_inflation=0.0,
            current_time=1,
            exogenous_total_consumption=None,
            forecasting_window=60,
            historic_real_gdp_growth=np.zeros(12),
        )


def test_country_productivity_growth_is_fixed_weight_laspeyres():
    country = Country.__new__(Country)
    tfp = TimeSeries(tfp_multiplier=np.array([1.0, 1.0]), labour_inputs=np.array([3.0, 1.0]))
    tfp.tfp_multiplier.append(np.array([1.1, 1.0]))
    tfp.labour_inputs.append(np.array([0.0, 10.0]))  # reallocation must not matter
    country.firms = SimpleNamespace(ts=tfp)
    out = country._historic_productivity_growth()
    np.testing.assert_allclose(out, [np.log((3 * 1.1 + 1.0) / 4.0)])


def test_country_productivity_growth_is_empty_at_t0():
    country = Country.__new__(Country)
    country.firms = SimpleNamespace(ts=TimeSeries(tfp_multiplier=np.array([1.0, 1.0]), labour_inputs=np.ones(2)))
    assert country._historic_productivity_growth().size == 0


def test_country_posted_price_index_by_industry():
    country = Country.__new__(Country)
    ts = TimeSeries(price=np.array([1.0, 2.0, 5.0]), production=np.array([1.0, 3.0, 2.0]))
    ts.price.append(np.array([1.2, 2.0, 5.5]))
    country.firms = SimpleNamespace(ts=ts, states={"Industry": np.array([0, 0, 1])})
    country.economy = SimpleNamespace(n_industries=3)
    out = country._posted_price_index_by_industry()
    np.testing.assert_allclose(out, [(1.2 * 1 + 2.0 * 3) / (1.0 * 1 + 2.0 * 3), 1.1, 1.0])
