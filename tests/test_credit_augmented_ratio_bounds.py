"""The active YAML controls the ratios used by maps and the consumption target."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.agents.households.func.consumption import CreditAugmentedConsumption
from macromodel.agents.households.households import Households
from macromodel.configurations import load_country_configuration

ROOT = Path(__file__).resolve().parents[1]


def parameters():
    return deepcopy(
        load_country_configuration(
            ROOT / "run_model/config/country_config_FRA.yaml", "FRA"
        ).households.functions.consumption.parameters
    )


def evaluate(rule, ratios, time_unit=12):
    ratios = np.atleast_2d(ratios).astype(float)
    n = len(ratios)
    income = np.full(n, 100.0 * time_unit / 12)
    annual_income = income * 12 / time_unit
    nla, ifa, ha, lfa = ratios.T
    zero = np.zeros(n)
    return rule._evaluate_target(
        income=income,
        lagged_income=income,
        lagged_consumption=income,
        liquid_wealth=lfa * annual_income,
        illiquid_wealth=ifa * annual_income,
        housing_wealth=ha * annual_income,
        lagged_housing_wealth=ha * annual_income,
        rent=zero,
        mortgage_debt=(lfa - nla) * annual_income,
        mortgage_payment=zero,
        lagged_liquid_wealth=lfa * annual_income,
        lagged_illiquid_wealth=ifa * annual_income,
        lagged_mortgage_debt=(lfa - nla) * annual_income,
        lagged_consumption_loan_debt=zero,
        cashflow_consumer_debt=zero,
        owner_occupied=zero,
        mortgagor=zero,
        house_price_index=None,
        lagged_house_price_index=None,
        real_borrowing_rate=None,
        permanent_income_log_ratio=np.full(n, 0.2),
        consumer_debt_rate_delta=None,
        uncertainty_delta=None,
        current_cpi=1.0,
        lagged_cpi=1.0,
        initial_cpi=1.0,
        expected_inflation=0.0,
        population_scale_factor=1.0,
        time_unit=time_unit,
        lagged_real_consumption_budget=income,
        epsilon=zero,
    )


def test_outside_bounds_matches_boundary_for_map_and_entire_target():
    cfg = parameters()
    rule = CreditAugmentedConsumption(**cfg)
    cal = cfg["continuous_wealth_calibration"]
    bounds = np.array(
        [
            cal[k]
            for k in [
                "net_liquid_assets_ratio_bounds",
                "illiquid_financial_assets_ratio_bounds",
                "housing_assets_ratio_bounds",
                "liquid_financial_assets_ratio_bounds",
            ]
        ]
    )
    outside = np.array([[-20.0, -3.0, -2.0, -1.0], [30.0, 80.0, 100.0, 70.0]])
    bounded = np.clip(outside, bounds[:, 0], bounds[:, 1])
    a, target_a = evaluate(rule, outside)
    b, target_b = evaluate(rule, bounded)
    for key in [
        "alpha_2",
        "gamma_1",
        "liquid_wealth",
        "illiquid_wealth",
        "housing_wealth",
        "wealth_contribution",
        "log_long_run",
        "long_run",
    ]:
        np.testing.assert_allclose(a["target_consumption_" + key], b["target_consumption_" + key])
    np.testing.assert_allclose(target_a, target_b)
    for j, key in enumerate(["nla", "ifa", "ha", "lfa"]):
        np.testing.assert_array_equal(a["target_consumption_" + key + "_ratio_winsorized"], 1.0)
        np.testing.assert_array_equal(b["target_consumption_" + key + "_ratio_winsorized"], 0.0)
    np.testing.assert_array_equal(a["target_consumption_raw_illiquid_assets_ratio"], outside[:, 1])
    np.testing.assert_array_equal(a["target_consumption_illiquid_assets_ratio"], bounded[:, 1])


def test_yaml_maps_and_logarithmic_tail_match_independent_equation():
    cfg = parameters()
    rule = CreditAugmentedConsumption(**cfg)
    x = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 2.0, 2.0], [-1.0, 8.0, 16.0, 0.0]])
    out, _ = evaluate(rule, x)
    c = cfg["continuous_wealth_calibration"]
    score = x[:, :3] @ np.array(
        [c["weight_net_liquid_assets"], c["weight_illiquid_financial_assets"], c["weight_housing_assets"]]
    )

    def logistic(z):
        return 1 / (1 + np.exp(-z))

    alpha = 0.1 + 0.8 * logistic(
        c["alpha_2_steepness"] * ((score - c["b_raw_min"]) / (c["b_raw_max"] - c["b_raw_min"]) - c["alpha_2_midpoint"])
    )
    gamma = c["gamma_1_low"] + (c["gamma_1_high"] - c["gamma_1_low"]) * (
        1 - logistic(c["gamma_1_steepness"] * (x[:, 3] / c["lfa_raw_max"] - c["gamma_1_midpoint"]))
    )
    W = gamma * x[:, 0] + 0.022 * x[:, 1] + 0.013 * x[:, 2]
    F = W.copy()
    threshold = 0.95 * (1 - alpha)
    tail = W > threshold
    F[tail] = threshold[tail] * (1 + np.log(W[tail] / threshold[tail]))
    assert tail.any()
    np.testing.assert_allclose(out["target_consumption_alpha_2"], alpha)
    np.testing.assert_allclose(out["target_consumption_gamma_1"], gamma)
    np.testing.assert_allclose(out["target_consumption_log_long_run"], cfg["long_run_intercept"] + alpha * 0.2 + F)
    np.testing.assert_array_equal(out["target_consumption_wealth_drag_clipped"], 0.0)
    np.testing.assert_array_equal(out["target_consumption_wealth_logarithmic_tail"], tail)
    for period in [1, 3, 12]:
        other, _ = evaluate(rule, x, time_unit=period)
        np.testing.assert_allclose(other["target_consumption_log_long_run"], out["target_consumption_log_long_run"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("net_liquid_assets_ratio_bounds", [0.0, 0.0]),
        ("illiquid_financial_assets_ratio_bounds", [0.0, np.nan]),
        ("liquid_financial_assets_ratio_bounds", [2.0, 1.0]),
        ("wealth_saturation_tau", 1.1),
    ],
)
def test_invalid_bounds_and_tau_fail_at_construction(key, value):
    cfg = parameters()
    cfg["continuous_wealth_calibration"][key] = value
    with pytest.raises(ValueError):
        CreditAugmentedConsumption(**cfg)


@pytest.mark.parametrize("concept,expected", [("main_residence", [10.0, 8.0]), ("all_properties", [30.0, 25.0])])
def test_housing_caller_uses_configured_concept(concept, expected):
    obj = Households.__new__(Households)
    obj.functions = {"consumption": SimpleNamespace(continuous_wealth_calibration_housing_asset_concept=concept)}
    current = {"wealth_main_residence": np.array([10.0]), "wealth_other_properties": np.array([20.0])}
    lagged = {"wealth_main_residence": np.array([8.0]), "wealth_other_properties": np.array([17.0])}
    obj.ts = SimpleNamespace(current=lambda key: current[key], prev=lambda key: lagged[key])
    actual = obj._consumption_housing_wealth()
    np.testing.assert_allclose([actual[0][0], actual[1][0]], expected)
    override = np.array([42.0])
    assert obj._consumption_housing_wealth(override)[1] is override
