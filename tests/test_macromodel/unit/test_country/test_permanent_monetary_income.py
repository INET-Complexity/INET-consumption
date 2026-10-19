"""Acceptance checks for monetary coverage and the unchanged learning state."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr

from macromodel.agents.households.func.consumption import CreditAugmentedConsumption
from macromodel.util.permanent_monetary_income import monetary_income_moments


def moments(n=100.0, p=100.0, floor=1.0, mu=0.0, variance=0.04):
    return monetary_income_moments(
        **{
            key: np.array([value])
            for key, value in {
                "non_property": n,
                "property_income": p,
                "income_floor": floor,
                "log_mean": mu,
                "log_variance": variance,
            }.items()
        }
    )


def test_zero_variance_uses_matching_monetary_coverage():
    result = moments(mu=np.log(1.1), variance=0)
    assert result["current"][0] == pytest.approx(200)
    assert result["mean"][0] == pytest.approx(np.log(210 / 200))
    assert result["variance"][0] == 0


def test_pure_earnings_recovers_original_moments_when_floor_irrelevant():
    result = moments(p=0, floor=0.0001, mu=0.1, variance=0.04)
    assert result["mean"][0] == pytest.approx(0.1, abs=1e-10)
    assert result["variance"][0] == pytest.approx(0.04, abs=1e-10)


@pytest.mark.parametrize(
    "n,p,floor,mu,variance",
    [
        (100, 100, 1, 0.1, 0.2),
        (100, -80, 10, -0.1, 0.16),
        (100, 0, 100, 0, 0.04),
        (1, 100, 1, 2, 2),
        (100, 0, 1000, 0, 0.3),
    ],
)
def test_both_moments_agree_with_independent_adaptive_integral(n, p, floor, mu, variance):
    result = moments(n, p, floor, mu, variance)
    sd = np.sqrt(variance)
    baseline = np.log(max(n + p, floor))
    cutoff = (np.log((floor - p) / n) - mu) / sd if floor > p else -12
    a = max(-12, min(12, cutoff))
    offset = np.log(floor) - baseline
    probability = ndtr(cutoff) if floor > p else 0

    def g(z):
        return np.log(max(n * np.exp(mu + sd * z) + p, floor)) - baseline

    def density(z):
        return np.exp(-z * z / 2) / np.sqrt(2 * np.pi)

    expected = probability * offset + quad(lambda z: g(z) * density(z), a, 12, epsabs=1e-11)[0]
    expected_var = (
        probability * (offset - expected) ** 2
        + quad(lambda z: (g(z) - expected) ** 2 * density(z), a, 12, epsabs=1e-11)[0]
    )
    assert result["mean"][0] == pytest.approx(expected, abs=2e-8)
    assert result["variance"][0] == pytest.approx(expected_var, abs=2e-8)


def test_currency_scale_invariance_and_zero_nonproperty():
    first = moments(n=100, p=-50, floor=10)
    scaled = moments(n=1e11, p=-5e10, floor=1e10)
    for key in ("mean", "variance", "projected_floor_probability"):
        np.testing.assert_allclose(first[key], scaled[key], atol=1e-10)
    for n in (0, -10):
        result = moments(n=n, p=50, mu=2)
        assert result["mean"][0] == 0
        assert result["variance"][0] == 0
        assert result["component_floor_used"][0] == (n < 0)


@pytest.mark.parametrize("keyword,value", [("variance", -1), ("mu", np.nan), ("floor", 0), ("p", np.inf)])
def test_invalid_moments_fail_before_state_changes(keyword, value):
    with pytest.raises(ValueError):
        moments(**{keyword: value})


def test_transformed_uncertainty_keeps_same_lag_during_replanning(test_households):
    hh = test_households
    n = hh.ts.current("n_households")
    hh.states["income_belief_priors"] = {
        "income_belief_mu": np.full(n, 0.05),
        "income_belief_p": np.full(n, 0.005),
        "income_belief_rho": np.full(n, 0.9519),
        "sigma2_v": np.full(n, 0.001),
        "sigma2_xi": np.full(n, 0.01),
    }
    hh.functions["consumption"] = CreditAugmentedConsumption(
        uses_income_belief_learning=True,
        income_belief_learning_horizon={"delta": 0.95, "S": 40},
    )
    components = {"non_property": np.full(n, 100.0), "property_income": np.full(n, 100.0), "income_floor": np.ones(n)}
    first = hh.current_income_belief_learning_inputs(
        monetary_income_components=components, common_permanent_income_log_ratio=0.03, common_forecast_variance=0.01
    )
    state = deepcopy(hh.states["income_belief_runtime_state"])
    np.testing.assert_allclose(
        first["permanent_income_log_ratio"], first["permanent_income_log_ratio_individual"] + 0.03
    )
    np.testing.assert_array_equal(first["uncertainty_delta"], np.zeros(n))
    hh.ts.income.append(hh.ts.current("income").copy())
    components["property_income"] *= 2
    revised = hh.current_income_belief_learning_inputs(
        monetary_income_components=components, common_forecast_variance=0.01
    )
    expected = revised["income_uncertainty"] - first["income_uncertainty"]
    np.testing.assert_allclose(revised["uncertainty_delta"], expected)
    again = hh.current_income_belief_learning_inputs(
        monetary_income_components=components, common_forecast_variance=0.01
    )
    np.testing.assert_allclose(again["uncertainty_delta"], expected)
    for key, value in state.items():
        np.testing.assert_array_equal(hh.states["income_belief_runtime_state"][key], value)


def test_incompatible_household_shapes_are_not_broadcast():
    with pytest.raises(ValueError, match="aligned finite"):
        monetary_income_moments(
            non_property=np.ones(2),
            property_income=np.ones(1),
            income_floor=np.ones(2),
            log_mean=np.zeros(2),
            log_variance=np.ones(2),
        )


def test_nonconvergent_integration_is_not_silently_used(monkeypatch):
    import macromodel.util.permanent_monetary_income as module

    monkeypatch.setattr(module, "_quadrature", lambda order: (np.array([1.0 / order]), np.array([2.0])))
    with pytest.raises(ValueError, match="did not converge"):
        moments(p=0, floor=0.0001)


def test_monetary_uncertainty_history_cannot_switch_back_to_earnings(test_households):
    hh = test_households
    n = hh.ts.current("n_households")
    hh.states["income_belief_priors"] = {
        "income_belief_mu": np.zeros(n),
        "income_belief_p": np.full(n, 0.005),
        "income_belief_rho": np.full(n, 0.9519),
        "sigma2_v": np.full(n, 0.001),
        "sigma2_xi": np.full(n, 0.01),
    }
    hh.functions["consumption"] = CreditAugmentedConsumption(
        uses_income_belief_learning=True, income_belief_learning_horizon={"delta": 0.95, "S": 40}
    )
    components = {"non_property": np.full(n, 100.0), "property_income": np.full(n, 100.0), "income_floor": np.ones(n)}
    hh.current_income_belief_learning_inputs(monetary_income_components=components)
    before = deepcopy(hh.states["income_uncertainty_runtime_state"])
    with pytest.raises(ValueError, match="fresh income histories"):
        hh.current_income_belief_learning_inputs()
    np.testing.assert_array_equal(before["current"], hh.states["income_uncertainty_runtime_state"]["current"])


def test_large_signed_property_values_preserve_currency_invariance():
    reference = moments(n=1, p=-1, floor=1)
    large = moments(n=1e308, p=-1e308, floor=1e308)
    assert reference["variance"][0] > 0
    for key in ("mean", "variance", "projected_floor_probability"):
        np.testing.assert_allclose(large[key], reference[key], atol=1e-10, rtol=1e-6)


@pytest.mark.parametrize("mu", [0.0, 1e-16, -1e-16, 0.1, -0.1])
def test_nearly_cancelling_property_receipts_match_high_precision_reference(mu):
    from decimal import Decimal, localcontext

    n, p, floor = 1e16, -9999999999999998.0, 1.0
    with localcontext() as context:
        context.prec = 80
        dn, dp, ds, dm = map(Decimal.from_float, (n, p, floor, mu))
        current = max(dn + dp, ds)
        projected = max(dn * dm.exp() + dp, ds)
        expected = float((projected / current).ln())
    result = moments(n=n, p=p, floor=floor, mu=mu, variance=0.0)
    assert result["current"][0] == 2.0
    assert result["mean"][0] == pytest.approx(expected, abs=1e-12)
    assert result["variance"][0] == 0.0


def test_near_cancellation_with_uncertainty_preserves_floor_and_both_moments():
    # At this scale, N*exp(1e-16*Z)+P equals 2+Z to double precision.
    # This independent integral avoids cancellation in both income and cutoff.
    result = moments(n=1e16, p=-9999999999999998.0, floor=1.0, mu=0.0, variance=1e-32)
    floor_probability = ndtr(-1.0)
    offset = -np.log(2.0)

    def g(z):
        return np.log(max(2.0 + z, 1.0) / 2.0)

    def density(z):
        return np.exp(-z * z / 2.0) / np.sqrt(2.0 * np.pi)

    mean = floor_probability * offset + quad(lambda z: g(z) * density(z), -1.0, 12.0, epsabs=1e-12)[0]
    variance = (
        floor_probability * (offset - mean) ** 2
        + quad(lambda z: (g(z) - mean) ** 2 * density(z), -1.0, 12.0, epsabs=1e-12)[0]
    )
    assert result["projected_floor_probability"][0] == pytest.approx(floor_probability, abs=1e-12)
    assert result["mean"][0] == pytest.approx(mean, abs=1e-10)
    assert result["variance"][0] == pytest.approx(variance, abs=1e-10)
