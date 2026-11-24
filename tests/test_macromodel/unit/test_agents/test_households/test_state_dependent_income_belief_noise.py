"""State-dependent income-belief noise variances and the implied beta3 helper."""

from types import SimpleNamespace

import numpy as np
import pytest

from macro_data.readers.income_belief_priors import load_income_belief_priors_table
from macromodel.agents.households.func.consumption import CreditAugmentedConsumption
from macromodel.agents.households.income_belief_learning import (
    compute_zeta,
    implied_uncertainty_propensity,
    steady_state_posterior_variance,
)
from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.country.country import Country

TABLE = load_income_belief_priors_table(None)
EMPLOYED = TABLE.loc[("employed", "35-49")]
UNEMPLOYED = TABLE.loc[("unemployed", "35-49")]
RHO = float(EMPLOYED["rho"])


def _stub_country(activity_status, weight_income, switch_on=True):
    """Two households: 0 = one member, 1 = two members; all aged 40."""
    return SimpleNamespace(
        households=SimpleNamespace(
            functions={"consumption": SimpleNamespace(uses_state_dependent_income_belief_noise=switch_on)},
            states={
                "income_belief_noise_inputs": {
                    "priors_table": TABLE,
                    "member_weight_income": np.asarray(weight_income, dtype=float),
                }
            },
            ts=SimpleNamespace(current=lambda name: 2),
        ),
        individuals=SimpleNamespace(
            states={
                "Age": np.array([40, 40, 40]),
                "Activity Status": np.asarray(activity_status),
                "Corresponding Household ID": np.array([0, 1, 1]),
            }
        ),
    )


def test__noise_variances_follow_current_status_with_initial_weights():
    E, U = ActivityStatus.EMPLOYED.value, ActivityStatus.UNEMPLOYED.value
    weights = [100.0, 300.0, 100.0]

    before = Country._current_income_belief_noise_variances(_stub_country([E, E, E], weights))
    np.testing.assert_allclose(before["sigma2_v"], EMPLOYED["sigma2_v"])
    np.testing.assert_allclose(before["sigma2_xi"], EMPLOYED["sigma2_xi"])

    # Household 0 loses its only earner; household 1 loses the lower earner (weight 1/4).
    after = Country._current_income_belief_noise_variances(_stub_country([U, E, U], weights))
    np.testing.assert_allclose(
        after["sigma2_v"], [UNEMPLOYED["sigma2_v"], 0.75 * EMPLOYED["sigma2_v"] + 0.25 * UNEMPLOYED["sigma2_v"]]
    )
    np.testing.assert_allclose(
        after["sigma2_xi"], [UNEMPLOYED["sigma2_xi"], 0.75 * EMPLOYED["sigma2_xi"] + 0.25 * UNEMPLOYED["sigma2_xi"]]
    )


def test__switch_off_returns_no_override_and_defaults_off():
    E = ActivityStatus.EMPLOYED.value
    stub = _stub_country([E, E, E], [1.0, 1.0, 1.0], switch_on=False)
    assert Country._current_income_belief_noise_variances(stub) is None
    assert CreditAugmentedConsumption().uses_state_dependent_income_belief_noise is False


def _employed_priors(n_households):
    return {
        "income_belief_mu": np.full(n_households, 0.01),
        "income_belief_p": np.full(n_households, 0.004),
        "income_belief_rho": np.full(n_households, RHO),
        "sigma2_v": np.full(n_households, EMPLOYED["sigma2_v"]),
        "sigma2_xi": np.full(n_households, EMPLOYED["sigma2_xi"]),
    }


def _noise(n_households, row):
    return {"sigma2_v": np.full(n_households, row["sigma2_v"]), "sigma2_xi": np.full(n_households, row["sigma2_xi"])}


def test__switch_carries_posterior_state_and_only_changes_noise(test_households):
    n = test_households.ts.current("n_households")
    test_households.states["income_belief_priors"] = _employed_priors(n)
    income = np.full(n, 100.0)
    test_households.update_income_belief_learning_state(current_income=income, lagged_income=income)
    state = test_households.states["income_belief_runtime_state"]
    mu_prev, p_prev = state["posterior_mean"].copy(), state["posterior_variance"].copy()

    outputs = test_households.update_income_belief_learning_state(
        current_income=income, lagged_income=income, noise_variances=_noise(n, UNEMPLOYED)
    )

    np.testing.assert_allclose(outputs.prior_mean, mu_prev)
    np.testing.assert_allclose(outputs.prior_variance, p_prev)
    predicted = RHO**2 * p_prev + UNEMPLOYED["sigma2_v"]
    np.testing.assert_allclose(outputs.predicted_variance, predicted)
    np.testing.assert_allclose(
        outputs.posterior_variance, predicted * UNEMPLOYED["sigma2_xi"] / (predicted + UNEMPLOYED["sigma2_xi"])
    )
    # The static priors are not overwritten by the per-quarter override.
    np.testing.assert_allclose(test_households.states["income_belief_priors"]["sigma2_v"], EMPLOYED["sigma2_v"])


def test__posterior_variance_rises_in_unemployment_and_falls_back(test_households):
    n = test_households.ts.current("n_households")
    test_households.states["income_belief_priors"] = _employed_priors(n)
    income = np.full(n, 100.0)

    def run(quarters, noise):
        for _ in range(quarters):
            test_households.update_income_belief_learning_state(
                current_income=income, lagged_income=income, noise_variances=noise
            )
        return test_households.states["income_belief_runtime_state"]["posterior_variance"].copy()

    p_employed = run(200, _noise(n, EMPLOYED))
    p_steady_employed = steady_state_posterior_variance(RHO, EMPLOYED["sigma2_v"], EMPLOYED["sigma2_xi"])
    np.testing.assert_allclose(p_employed, p_steady_employed, rtol=1e-8)

    p_unemployed = run(4, _noise(n, UNEMPLOYED))
    assert np.all(p_unemployed > p_employed)

    p_reemployed = run(200, _noise(n, EMPLOYED))
    assert np.all(p_reemployed < p_unemployed)
    np.testing.assert_allclose(p_reemployed, p_steady_employed, rtol=1e-8)


def test__no_override_matches_static_priors(test_households):
    n = test_households.ts.current("n_households")
    test_households.states["income_belief_priors"] = _employed_priors(n)
    rng = np.random.default_rng(0)
    current, lagged = rng.uniform(50, 150, n), rng.uniform(50, 150, n)
    without = test_households.update_income_belief_learning_state(current_income=current, lagged_income=lagged)
    test_households.states.pop("income_belief_runtime_state")
    with_static = test_households.update_income_belief_learning_state(
        current_income=current, lagged_income=lagged, noise_variances=_noise(n, EMPLOYED)
    )
    np.testing.assert_array_equal(without.posterior_mean, with_static.posterior_mean)
    np.testing.assert_array_equal(without.posterior_variance, with_static.posterior_variance)


def test__steady_state_posterior_variance_is_riccati_fixed_point():
    p = steady_state_posterior_variance(RHO, EMPLOYED["sigma2_v"], EMPLOYED["sigma2_xi"])
    predicted = RHO**2 * p + EMPLOYED["sigma2_v"]
    assert p == pytest.approx(predicted * EMPLOYED["sigma2_xi"] / (predicted + EMPLOYED["sigma2_xi"]), rel=1e-12)


def test__implied_uncertainty_propensity_matches_hand_calculation():
    zeta = compute_zeta(RHO, 0.95, 40)
    beta3 = implied_uncertainty_propensity(TABLE, "35-49", zeta, beta_ref=-0.0043)

    p_e = steady_state_posterior_variance(RHO, EMPLOYED["sigma2_v"], EMPLOYED["sigma2_xi"])
    p = p_e
    for _ in range(4):
        predicted = RHO**2 * p + UNEMPLOYED["sigma2_v"]
        p = predicted * UNEMPLOYED["sigma2_xi"] / (predicted + UNEMPLOYED["sigma2_xi"])
    assert beta3 == pytest.approx(-0.0043 / (0.01 * zeta**2 * (p - p_e)))
    assert beta3 == pytest.approx(-28.6, abs=0.1)
    # Linear in the reference coefficient.
    assert implied_uncertainty_propensity(TABLE, "35-49", zeta, beta_ref=-0.0086) == pytest.approx(2 * beta3)
