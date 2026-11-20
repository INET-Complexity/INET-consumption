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
    return ARDLGovernmentConsumptionSetter(consistency=1.0, sectoral_weights="initial_fixed", **kwargs)


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


def test_target_is_previous_envelope_times_real_growth_times_expected_inflation():
    s = setter()
    prev = np.array([30.0, 50.0, 20.0])
    g = np.array([100.0, 101.0, 102.0])
    y = np.full(12, 0.003)
    out = s.compute_target_consumption(
        previous_desired_government_consumption=prev,
        model=None,
        historic_total_consumption=g,
        initial_good_prices=np.ones(3),
        current_good_prices=np.ones(3),
        expected_growth=0.5,  # ignored by this rule
        expected_inflation=0.01,
        current_time=1,
        exogenous_total_consumption=None,
        forecasting_window=60,
        historic_real_gdp_growth=y,
    )
    growth = s.real_growth(g, y)
    assert out.sum() == pytest.approx(prev.sum() * np.exp(growth) * 1.01)
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
