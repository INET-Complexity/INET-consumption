"""Independent target/dynamic/probe oracles for prepared consumption inputs."""

import pickle
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from macromodel.agents.households.consumption_inputs import (
    IncomeHistories,
    MonetaryUnits,
    OpeningSnapshot,
    OpeningStockValues,
    PermanentIncomePayload,
    prepare_inputs,
)
from macromodel.agents.households.func.consumption import CreditAugmentedConsumption
from macromodel.configurations import load_country_configuration

ROOT = Path(__file__).resolve().parents[5]


def configuration():
    return deepcopy(
        load_country_configuration(
            ROOT / "run_model/config/country_config_FRA.yaml", "FRA"
        ).households.functions.consumption.parameters
    )


def prepared(
    *,
    months=3,
    scale=1,
    raw=None,
    floor=None,
    prices=None,
    current=120.0,
    initial=100.0,
    values=None,
    q=0.2,
    m=None,
    pass_kind="first",
):
    units = MonetaryUnits("EUR", scale)
    raw = np.full((4, 2), 120 * months / 3 * scale) if raw is None else np.asarray(raw, float)
    floor = np.full_like(raw, 10 * months / 3 * scale) if floor is None else np.asarray(floor, float)
    prices = [100, 110, 120, current] if prices is None else prices
    h = raw.shape[1]
    values = [np.full(h, v * scale) for v in [500, 800, 1500, 100, 50, 90, 40]] if values is None else values
    opening = OpeningSnapshot(1, "run:order", units, OpeningStockValues(*values))
    pi = PermanentIncomePayload(
        1,
        pass_kind,
        "run:order",
        units,
        np.full(h, 150 * months / 3 * scale) if m is None else m,
        np.broadcast_to(q, (h,)),
    )
    histories = IncomeHistories(raw, floor, prices, current, initial, "aligned", units, units, months_per_period=months)
    return prepare_inputs(histories, opening, pi, period_id=1, pass_kind=pass_kind, household_layout_key="run:order")


def oracle(cfg, b, p, opening, *, alpha=None, gamma=None, rate=0.0, epsilon=0.0, months=3):
    annual = b * 12 / months
    x = np.column_stack(
        [
            opening.net_liquid_assets / annual,
            opening.illiquid_financial_assets / annual,
            opening.main_residence_wealth / annual,
            opening.liquid_financial_assets / annual,
        ]
    )
    cal = cfg["continuous_wealth_calibration"]
    bounds = np.array(
        [
            cal[key]
            for key in [
                "net_liquid_assets_ratio_bounds",
                "illiquid_financial_assets_ratio_bounds",
                "housing_assets_ratio_bounds",
                "liquid_financial_assets_ratio_bounds",
            ]
        ]
    )
    x = np.clip(x, bounds[:, 0], bounds[:, 1])
    score = x[:, :3] @ np.array(
        [cal["weight_net_liquid_assets"], cal["weight_illiquid_financial_assets"], cal["weight_housing_assets"]]
    )
    if alpha is None:
        alpha = 0.1 + 0.8 / (
            1
            + np.exp(
                -cal["alpha_2_steepness"]
                * ((score - cal["b_raw_min"]) / (cal["b_raw_max"] - cal["b_raw_min"]) - cal["alpha_2_midpoint"])
            )
        )
    if gamma is None:
        gamma = cal["gamma_1_low"] + (cal["gamma_1_high"] - cal["gamma_1_low"]) * (
            1
            - 1
            / (
                1
                + np.exp(
                    -cal["gamma_1_steepness"]
                    * (
                        (x[:, 3] - cal["lfa_raw_min"]) / (cal["lfa_raw_max"] - cal["lfa_raw_min"])
                        - cal["gamma_1_midpoint"]
                    )
                )
            )
        )
    w = gamma * x[:, 0] + 0.022 * x[:, 1] + 0.013 * x[:, 2]
    threshold = 0.95 * (1 - alpha)
    f = np.where(w > threshold, threshold * (1 + np.log(np.maximum(w, 1e-300) / threshold)), w)
    log_target = np.log(b) + cfg["long_run_intercept"] - 0.72 * rate + alpha * p + f + epsilon
    return log_target, alpha, gamma


def compute(rule, data, budget=None, **kwargs):
    h = data.income_base_real.size
    return rule.compute_target_consumption(
        expected_inflation=0.0,
        current_cpi=data.histories.current_consumer_price,
        initial_cpi=data.histories.initial_consumer_price,
        historic_consumption_sum=np.ones((1, h)),
        saving_rates=np.zeros(h),
        income=data.histories.behavioural_income_nominal_history[-1],
        household_benefits=np.zeros(h),
        consumption_weights=np.array([0.4, 0.6]),
        consumption_weights_by_income=np.full((5, 2), 0.5),
        exogenous_total_consumption=np.ones(h),
        current_time=1,
        take_consumption_weights_by_income_quantile=False,
        tau_vat=0.2,
        lagged_real_consumption_budget=data.income_base_real if budget is None else budget,
        time_unit=data.histories.months_per_period,
        consumption_inputs=data,
        **kwargs,
    )


def test_hand_smoothed_base_rebase_target_and_dynamic_decomposition():
    data = prepared(
        raw=[[0, 20], [-2, 40], [80, 40], [160, 160]],
        floor=[[10, 10], [40, 20], [30, 10], [160, 10]],
        prices=[100, 200, 100, 200],
        current=200,
    )
    expected = np.array([(10 * 20 * 80 * 80) ** 0.25, (20 * 20 * 40 * 80) ** 0.25])
    np.testing.assert_allclose(data.income_base_real, expected, rtol=1e-10, atol=1e-12)
    cfg = configuration()
    rule = CreditAugmentedConsumption(**cfg)
    rate = np.array([0.04, 0.1])
    budget = expected * 0.7
    out, total = rule.evaluate_prepared_target(
        data,
        lagged_real_consumption_budget=budget,
        real_borrowing_rate=rate,
        epsilon=np.full(2, 0.03),
        consumer_debt_rate_delta=0.02,
        uncertainty_delta=0.4,
        expected_inflation=0.025,
    )
    log_target, a, g = oracle(cfg, expected, data.pi_log_ratio_rebased, data.opening_real, rate=rate, epsilon=0.03)
    np.testing.assert_allclose(out["target_consumption_raw_log_target"], log_target, rtol=1e-10, atol=1e-12)
    cash = 0.02 * data.opening_real.consumer_ledger_debt / (4 * expected)
    delta = 0.56 * (log_target - np.log(budget)) - 0.003 * cash - 0.005 * 0.4
    np.testing.assert_allclose(out["target_consumption_interest_rate_cashflow_index"], cash, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(out["target_consumption_unguarded_delta_log_consumption"], delta, rtol=1e-10, atol=1e-12)
    desired = budget * np.exp(np.clip(delta, -0.5, 0.5))
    np.testing.assert_allclose(out["target_consumption_real_budget"], desired, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(total, desired * 2 * 1.025, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(out["target_consumption_real_net_liquid_assets"], data.opening_real.net_liquid_assets)


@pytest.mark.parametrize("months", [1, 3, 12])
@pytest.mark.parametrize("scale", [1, 5000])
def test_period_and_currency_invariance(months, scale):
    cfg = configuration()
    data = prepared(months=months, scale=scale)
    rule = CreditAugmentedConsumption(**cfg)
    out, _ = rule.evaluate_prepared_target(data, lagged_real_consumption_budget=data.income_base_real)
    reference = prepared()
    ref, _ = rule.evaluate_prepared_target(reference, lagged_real_consumption_budget=reference.income_base_real)
    for suffix in ["alpha_2", "gamma_1", "log_long_run", "permanent_income_log_ratio"]:
        np.testing.assert_allclose(
            out["target_consumption_" + suffix], ref["target_consumption_" + suffix], rtol=1e-10, atol=1e-12
        )
    np.testing.assert_allclose(
        out["target_consumption_raw_target_real"] / scale * 3 / months,
        ref["target_consumption_raw_target_real"],
        rtol=1e-10,
        atol=1e-12,
    )


def test_initial_anchor_fixed_residual_sigma_intercept_and_no_old_helper():
    data = prepared()
    results = []
    for intercept_shift, sigma in [(0.0, 0.1), (0.4, 0.8)]:
        cfg = configuration()
        cfg["long_run_intercept"] += intercept_shift
        cfg["idiosyncratic_sd"] = sigma
        rule = CreditAugmentedConsumption(**cfg)
        rule._geometric_average_income = Mock(side_effect=AssertionError("old helper called"))
        compute(rule, data)
        epsilon = rule._household_epsilon.copy()
        np.testing.assert_allclose(
            rule.last_target_consumption_components["target_consumption_raw_target_real"],
            data.income_base_real,
            rtol=1e-10,
            atol=1e-12,
        )
        # Historical formula probe must be a nonzero response, despite prepared operands.
        assert np.all(rule.last_formula_implied_mpc > 0)
        assert np.all(np.isfinite(epsilon))
        changed = prepared(raw=np.full((4, 2), 300.0), pass_kind="replacement")
        compute(rule, changed)
        np.testing.assert_array_equal(rule._household_epsilon, epsilon)
        results.append(epsilon)
    np.testing.assert_allclose(results[1] - results[0], -0.4, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("q", [10000.0, -10000.0])
def test_growth_and_numerical_guard_binding(q):
    data = prepared(q=q)
    rule = CreditAugmentedConsumption(**configuration())
    out, _ = rule.evaluate_prepared_target(data, lagged_real_consumption_budget=data.income_base_real)
    np.testing.assert_array_equal(out["target_consumption_raw_target_log_guarded"], 1)
    np.testing.assert_array_equal(out["target_consumption_growth_clipped"], 1)
    sign = 1 if q > 0 else -1
    np.testing.assert_allclose(
        out["target_consumption_real_budget"] / data.income_base_real, np.exp(sign * 0.5), rtol=1e-10, atol=1e-12
    )


def test_conditional_income_freezes_maps_and_fixed_yp_rebase():
    cfg = configuration()
    rule = CreditAugmentedConsumption(**cfg)
    data = prepared()
    step = np.array([20.0, 50.0])
    anchor = np.array([90.0, 120.0])
    probe = rule.probe_prepared_target(
        data,
        kind="conditional_income",
        step_real=step,
        consumption_anchor_real=anchor,
        real_consumer_rate=0.03,
        real_mortgage_rate=0.05,
    )
    base, new = probe["baseline"], probe["perturbed"]
    for name in ["alpha_2", "gamma_1"]:
        np.testing.assert_array_equal(base["target_consumption_" + name], new["target_consumption_" + name])
    b = data.income_base_real + step
    p = data.pi_log_ratio_rebased - np.log1p(step / data.income_base_real)
    rate = (0.03 * data.opening_real.consumer_principal + 0.05 * data.opening_real.mortgage_principal) / (4 * b)
    log_target, *_ = oracle(
        cfg,
        b,
        p,
        data.opening_real,
        alpha=base["target_consumption_alpha_2"],
        gamma=base["target_consumption_gamma_1"],
        rate=rate,
    )
    np.testing.assert_allclose(new["target_consumption_raw_log_target"], log_target, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        probe["raw_target_mpc"],
        anchor * np.expm1(log_target - base["target_consumption_raw_log_target"]) / step,
        rtol=1e-10,
        atol=1e-12,
    )


def test_asset_probe_updates_alpha_freezes_gamma_and_no_state_mutation():
    cfg = configuration()
    rule = CreditAugmentedConsumption(**cfg)
    data = prepared()
    before = pickle.dumps((rule.__dict__, np.random.get_state(), data))
    probe = rule.probe_prepared_target(data, kind="conditional_asset", step_real=np.array([100.0, -50.0]))
    base, new = probe["baseline"], probe["perturbed"]
    assert np.all(new["target_consumption_alpha_2"] != base["target_consumption_alpha_2"])
    np.testing.assert_array_equal(new["target_consumption_gamma_1"], base["target_consumption_gamma_1"])
    changed = replace(data.opening_real, liquid_financial_assets=data.opening_real.liquid_financial_assets + [100, -50])
    log_target, *_ = oracle(
        cfg, data.income_base_real, data.pi_log_ratio_rebased, changed, gamma=base["target_consumption_gamma_1"]
    )
    np.testing.assert_allclose(new["target_consumption_raw_log_target"], log_target, rtol=1e-10, atol=1e-12)
    assert before == pickle.dumps((rule.__dict__, np.random.get_state(), data))


@pytest.mark.parametrize(
    "field,bound_name",
    [
        ("liquid_financial_assets", "liquid_financial_assets_ratio_bounds"),
        ("illiquid_financial_assets", "illiquid_financial_assets_ratio_bounds"),
        ("main_residence_wealth", "housing_assets_ratio_bounds"),
    ],
)
@pytest.mark.parametrize("offset", [-1e-4, 0.0, 1e-4])
def test_asset_cap_entry_exit_and_exact_boundary(field, bound_name, offset):
    cfg = configuration()
    rule = CreditAugmentedConsumption(**cfg)
    data = prepared()
    annual = data.annual_income_base_real
    values = [getattr(data.opening_snapshot.nominal, f) for f in OpeningStockValues.__dataclass_fields__]
    index = list(OpeningStockValues.__dataclass_fields__).index(field)
    values[index] = (cfg["continuous_wealth_calibration"][bound_name][1] + offset) * annual * data.current_deflator
    data = prepared(values=values)
    for signed in [-1, 1]:
        step = signed * annual * 1e-5
        result = rule.probe_prepared_target(data, kind="conditional_asset", stock_field=field, step_real=step)
        changed = replace(data.opening_real, **{field: getattr(data.opening_real, field) + step})
        log_target, *_ = oracle(
            cfg,
            data.income_base_real,
            data.pi_log_ratio_rebased,
            changed,
            gamma=result["baseline"]["target_consumption_gamma_1"],
        )
        np.testing.assert_allclose(
            result["perturbed"]["target_consumption_raw_log_target"], log_target, rtol=1e-10, atol=1e-12
        )


def test_historical_formula_is_distinct_and_has_nonzero_level_response():
    rule = CreditAugmentedConsumption(**configuration())
    data = prepared()
    step = np.full(2, 10.0)
    probe = rule.probe_prepared_target(data, kind="historical_formula", step_real=step)
    np.testing.assert_allclose(
        probe["delta_log_raw_target"], np.log1p(step / data.income_base_real), rtol=1e-10, atol=1e-12
    )
    assert np.all(probe["desired_budget_mpc"] > 0)
    conditional = rule.probe_prepared_target(data, kind="conditional_income", step_real=step)
    assert not np.allclose(probe["raw_target_mpc"], conditional["raw_target_mpc"])


@pytest.mark.parametrize("changes", [{"period_id": 2}, {"pass_kind": "replacement"}, {"household_layout_key": "other"}])
def test_invalid_prepared_identity_rejected(changes):
    rule = CreditAugmentedConsumption(**configuration())
    data = replace(prepared(), **changes)
    with pytest.raises(ValueError, match="identity"):
        rule.evaluate_prepared_target(data, lagged_real_consumption_budget=data.income_base_real)


@pytest.mark.parametrize("step", [float("nan"), float("inf"), -1e6])
def test_invalid_conditional_steps(step):
    rule = CreditAugmentedConsumption(**configuration())
    data = prepared()
    with pytest.raises(ValueError):
        rule.probe_prepared_target(data, kind="conditional_income", step_real=step)


@pytest.mark.parametrize(
    "keyword", ["real_borrowing_rate", "consumer_debt_rate_delta", "uncertainty_delta", "expected_inflation"]
)
def test_nonfinite_dynamic_operands_rejected(keyword):
    rule = CreditAugmentedConsumption(**configuration())
    data = prepared()
    with pytest.raises(ValueError, match=keyword):
        rule.evaluate_prepared_target(data, lagged_real_consumption_budget=data.income_base_real, **{keyword: np.nan})


def test_inconsistent_prepared_annual_base_rejected():
    data = prepared()
    invalid = replace(data, annual_income_base_real=data.annual_income_base_real * 2)
    with pytest.raises(ValueError, match="annualize"):
        CreditAugmentedConsumption(**configuration()).evaluate_prepared_target(
            invalid, lagged_real_consumption_budget=data.income_base_real
        )


def test_same_size_new_layout_cannot_reuse_residuals_until_reset():
    rule = CreditAugmentedConsumption(**configuration())
    data = prepared()
    compute(rule, data)
    old = rule._household_epsilon.copy()
    key = "new-run:reordered"
    changed = replace(
        data,
        household_layout_key=key,
        opening_snapshot=replace(data.opening_snapshot, household_layout_key=key),
        pi_payload=replace(data.pi_payload, household_layout_key=key),
    )
    with pytest.raises(ValueError, match="residual layout"):
        compute(rule, changed)
    np.testing.assert_array_equal(rule._household_epsilon, old)
    rule.set_run_seed(15)
    compute(rule, changed)
    assert rule._prepared_residual_layout == key


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("offset", [-1e-4, 0.0, 1e-4])
def test_negative_nla_and_nla_bound_kinks(side, offset):
    cfg = configuration()
    data = prepared()
    annual = data.annual_income_base_real
    ratio = cfg["continuous_wealth_calibration"]["net_liquid_assets_ratio_bounds"][side] + offset
    nominal = data.opening_snapshot.nominal
    lfa = 6 * annual * data.current_deflator
    mortgage = lfa - nominal.consumer_ledger_debt - ratio * annual * data.current_deflator
    values = [getattr(nominal, f) for f in OpeningStockValues.__dataclass_fields__]
    values[0] = lfa
    values[3] = mortgage
    data = prepared(values=values)
    rule = CreditAugmentedConsumption(**cfg)
    for sign in (-1, 1):
        step = sign * 1e-5 * annual
        result = rule.probe_prepared_target(
            data, kind="conditional_asset", stock_field="mortgage_ledger_debt", step_real=step
        )
        changed = replace(data.opening_real, mortgage_ledger_debt=data.opening_real.mortgage_ledger_debt + step)
        expected, *_ = oracle(
            cfg,
            data.income_base_real,
            data.pi_log_ratio_rebased,
            changed,
            gamma=result["baseline"]["target_consumption_gamma_1"],
        )
        np.testing.assert_allclose(
            result["perturbed"]["target_consumption_raw_log_target"], expected, rtol=1e-10, atol=1e-12
        )


def test_conditional_pi_diagnostics_still_reconstruct_supplied_proxy():
    data = prepared()
    out = CreditAugmentedConsumption(**configuration()).probe_prepared_target(
        data, kind="conditional_income", step_real=10.0
    )["perturbed"]
    np.testing.assert_allclose(
        out["target_consumption_pi_original"] + out["target_consumption_pi_denominator_adjustment"],
        out["target_consumption_permanent_income_log_ratio"],
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        out["target_consumption_log_permanent_income_proxy_real"],
        data.log_permanent_income_proxy_real,
        rtol=1e-10,
        atol=1e-12,
    )
