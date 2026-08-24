"""Continuous calibration v2: decoupled logistics, idiosyncratic term, denominator.

These lock down the three properties the port depends on and that a plausible
future edit could silently break.
"""

import numpy as np
import pytest

from macromodel.agents.households.func.consumption import CreditAugmentedConsumption

V2 = {
    "index_construction": "raw_ratio",
    "weight_net_liquid_assets": 0.6719,
    "weight_illiquid_financial_assets": 0.2486,
    "weight_housing_assets": 0.0795,
    "alpha_2_steepness": 2.012,
    "alpha_2_midpoint": 1.000,
    "gamma_1_steepness": 148.413,
    "gamma_1_midpoint": 0.0532,
    "alpha_2_low": 0.25,
    "alpha_2_high": 0.70,
    "gamma_1_low": 0.05,
    "gamma_1_high": 0.25,
    "net_liquid_assets_ratio_bounds": (-3.4132, 1.4576),
    "illiquid_financial_assets_ratio_bounds": (0.0, 1.6898),
    "housing_assets_ratio_bounds": (0.0, 11.5486),
    "b_raw_min": -2.2928,
    "b_raw_max": 2.3172,
}


def _rule(**kwargs):
    params = dict(uses_continuous_wealth_calibration=True, continuous_wealth_calibration=V2)
    params.update(kwargs)
    return CreditAugmentedConsumption(**params)


def _ratios(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(-3.4132, 1.4576, n),
        rng.uniform(0.0, 1.6898, n),
        rng.uniform(0.0, 11.5486, n),
    )


class TestDecoupledLogistics:
    def test_alpha_2_and_gamma_1_use_their_own_slopes(self):
        """A shared-slope implementation cannot produce both of these at once."""
        nla, ifa, ha = _ratios()
        alpha_2, gamma_1 = _rule()._compute_continuous_wealth_calibration(nla, ifa, ha)
        # k_alpha = 2.0 is gentle: alpha_2 varies smoothly and widely.
        assert alpha_2.std() > 0.02
        # k_gamma = 148 on a [0,1] domain is a step: gamma_1 is at one end or the
        # other for essentially every household.
        interior = np.mean((gamma_1 > 0.055) & (gamma_1 < 0.245))
        assert interior < 0.05

    def test_alpha_2_rises_and_gamma_1_falls_in_b(self):
        """Anti-correlated by construction -- the sign the stale doc had backwards."""
        nla, ifa, ha = _ratios()
        alpha_2, gamma_1 = _rule()._compute_continuous_wealth_calibration(nla, ifa, ha)
        b = 0.6719 * nla + 0.2486 * ifa + 0.0795 * ha
        assert np.corrcoef(b, alpha_2)[0, 1] > 0.9
        assert np.corrcoef(b, gamma_1)[0, 1] < 0.0

    def test_coefficients_stay_inside_their_configured_ranges(self):
        nla, ifa, ha = _ratios()
        alpha_2, gamma_1 = _rule()._compute_continuous_wealth_calibration(nla, ifa, ha)
        # Tolerance is for float64 round-off at the bound only (the logistic
        # saturates exactly there), not for genuine slack.
        tol = 1e-12
        assert alpha_2.min() >= 0.25 - tol and alpha_2.max() <= 0.70 + tol
        assert gamma_1.min() >= 0.05 - tol and gamma_1.max() <= 0.25 + tol

    def test_raw_and_normalised_index_construction_differ(self):
        """Guards the port's central correction: these are NOT interchangeable."""
        nla, ifa, ha = _ratios()
        raw = _rule()._compute_continuous_wealth_calibration(nla, ifa, ha)[0]
        norm_cfg = dict(V2, index_construction="normalised_ratio")
        norm = _rule(continuous_wealth_calibration=norm_cfg)._compute_continuous_wealth_calibration(nla, ifa, ha)[0]
        assert not np.allclose(raw, norm)

    def test_v1_config_reproduces_v1_mapping(self):
        """Legacy shared steepness/b0 must survive untouched."""
        nla, ifa, ha = _ratios()
        alpha_2, gamma_1 = CreditAugmentedConsumption(
            uses_continuous_wealth_calibration=True
        )._compute_continuous_wealth_calibration(nla, ifa, ha)
        nla_n = (np.clip(nla, -4.73, 2.15) + 4.73) / (2.15 + 4.73)
        ifa_n = np.clip(ifa, 0.0, 3.54) / 3.54
        ha_n = np.clip(ha, 0.0, 17.08) / 17.08
        b = np.clip((1.0 * nla_n + 0.502 * ifa_n + 0.287 * ha_n) / 1.789, 0.0, 1.0)
        logistic = 1.0 / (1.0 + np.exp(-34.3 * (b - 0.428)))
        np.testing.assert_allclose(alpha_2, 0.2497 + (0.6997 - 0.2497) * logistic)
        np.testing.assert_allclose(gamma_1, 0.1997 - (0.1997 - 0.0503) * logistic)


class TestIdiosyncraticTerm:
    def test_repeated_calls_return_the_same_draw(self):
        """The MPC probe evaluates the target twice; a redraw destroys the derivative."""
        rule = _rule(idiosyncratic_sd=0.3308)
        np.testing.assert_array_equal(rule._epsilon(500), rule._epsilon(500))

    def test_population_growth_preserves_existing_households(self):
        rule = _rule(idiosyncratic_sd=0.3308)
        first = rule._epsilon(500).copy()
        np.testing.assert_array_equal(rule._epsilon(700)[:500], first)

    def test_does_not_disturb_the_global_numpy_stream(self):
        """Enabling eps must not shift any other seeded draw in the model."""
        rule = _rule(idiosyncratic_sd=0.3308)
        np.random.seed(99)
        expected = np.random.rand(3)
        rule._epsilon(10_000)
        np.random.seed(99)
        np.testing.assert_array_equal(np.random.rand(3), expected)

    def test_zero_sd_is_exactly_zero(self):
        assert not _rule()._epsilon(100).any()

    def test_sd_matches_the_calibrated_value(self):
        eps = _rule(idiosyncratic_sd=0.3308)._epsilon(200_000)
        assert eps.std() == pytest.approx(0.3308, abs=0.005)
        assert eps.mean() == pytest.approx(0.0, abs=0.005)

    def test_negative_sd_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            _rule(idiosyncratic_sd=-0.1)


class TestGeometricAverageIncome:
    def test_geometric_mean_over_the_window(self):
        rule = _rule(income_denominator="geometric_average", income_denominator_window=4)
        history = np.array([[1.0], [10.0], [100.0], [1000.0]])
        # Geometric mean of 1, 10, 100, 1000 is 10^1.5.
        got = rule._geometric_average_income(history, deflator=1.0, historic_deflator=np.ones(4))
        np.testing.assert_allclose(got, [10.0**1.5], rtol=1e-12)

    def test_window_truncates_older_periods(self):
        rule = _rule(income_denominator="geometric_average", income_denominator_window=2)
        history = np.array([[1e9], [4.0], [9.0]])
        np.testing.assert_allclose(rule._geometric_average_income(history, 1.0, np.ones(history.shape[0])), [6.0])

    def test_short_history_uses_what_is_available(self):
        """Warm-up: no jump when the window first fills."""
        rule = _rule(income_denominator="geometric_average", income_denominator_window=20)
        history = np.array([[4.0], [9.0]])
        np.testing.assert_allclose(rule._geometric_average_income(history, 1.0, np.ones(history.shape[0])), [6.0])

    def test_non_positive_observations_are_skipped_not_propagated(self):
        rule = _rule(income_denominator="geometric_average", income_denominator_window=4)
        history = np.array([[4.0], [0.0], [-5.0], [9.0]])
        np.testing.assert_allclose(rule._geometric_average_income(history, 1.0, np.ones(history.shape[0])), [6.0])

    def test_smooths_more_than_a_single_period_denominator(self):
        """The point of the change: this is the issue #90 blow-up mechanism."""
        rule = _rule(income_denominator="geometric_average", income_denominator_window=8)
        rng = np.random.default_rng(3)
        history = np.exp(rng.normal(np.log(30_000.0), 0.8, size=(8, 2000)))
        geometric = rule._geometric_average_income(history, 1.0, np.ones(history.shape[0]))
        assert geometric.std() < history[-1].std() / 2

    def test_deflator_is_applied(self):
        rule = _rule(income_denominator="geometric_average", income_denominator_window=2)
        history = np.array([[4.0], [9.0]])
        np.testing.assert_allclose(rule._geometric_average_income(history, 2.0, np.full(2, 2.0)), [3.0])

    def test_rejects_one_dimensional_history(self):
        rule = _rule(income_denominator="geometric_average")
        with pytest.raises(ValueError, match="2-D"):
            rule._geometric_average_income(np.array([1.0, 2.0]), 1.0, np.ones(1))


class TestConfigValidation:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"income_denominator": "mean"}, "income_denominator"),
            ({"income_denominator_window": 0}, "window"),
            ({"income_denominator_min_periods": 0}, "min_periods"),
            ({"idiosyncratic_persistence": "ar1"}, "persistence"),
            ({"continuous_wealth_calibration": dict(V2, index_construction="pca")}, "index_construction"),
        ],
    )
    def test_bad_configuration_fails_fast(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            _rule(**kwargs)

    def test_geometric_denominator_requires_history(self):
        """No silent fallback: fitted coefficients need the denominator they were fitted with."""
        rule = _rule(income_denominator="geometric_average")
        with pytest.raises(ValueError, match="requires historic_income"):
            rule.compute_target_consumption(
                expected_inflation=0.0,
                current_cpi=1.0,
                initial_cpi=1.0,
                historic_consumption_sum=np.ones((2, 3)),
                saving_rates=np.zeros(3),
                income=np.full(3, 100.0),
                household_benefits=np.zeros(3),
                consumption_weights=np.array([1.0]),
                consumption_weights_by_income=np.array([[1.0]]),
                exogenous_total_consumption=np.zeros(3),
                current_time=1,
                take_consumption_weights_by_income_quantile=False,
                tau_vat=0.0,
                lagged_real_consumption_budget=np.full(3, 50.0),
            )


class TestDeflatorAlignment:
    """Each historic income must be deflated at its own price level.

    Deflating the whole window by the current price level double-counts inflation
    for every lagged observation, biasing the denominator down by cumulative
    inflation -- a bias that grows with the price level and so inflates every
    wealth ratio more the longer a run goes on.
    """

    def test_each_period_uses_its_own_deflator(self):
        rule = _rule(income_denominator="geometric_average", income_denominator_window=2)
        # Nominal 100 then 200, price level 1.0 then 2.0 -> real 100 in both periods.
        history = np.array([[100.0], [200.0]])
        got = rule._geometric_average_income(history, 2.0, np.array([1.0, 2.0]))
        np.testing.assert_allclose(got, [100.0])

    def test_current_deflator_for_all_periods_would_understate(self):
        """Locks in the direction of the bug that was fixed."""
        rule = _rule(income_denominator="geometric_average", income_denominator_window=2)
        history = np.array([[100.0], [200.0]])
        correct = rule._geometric_average_income(history, 2.0, np.array([1.0, 2.0]))
        buggy = history[-2:] / 2.0  # what the old single-deflator path computed
        assert np.exp(np.log(buggy).mean()) < correct[0]

    def test_inflation_neutrality(self):
        """A constant real income gives the same denominator at any inflation rate."""
        rule = _rule(income_denominator="geometric_average", income_denominator_window=8)
        cpi = np.cumprod(np.full(8, 1.05))
        history = (30_000.0 * cpi)[:, None]
        got = rule._geometric_average_income(history, float(cpi[-1]), cpi)
        np.testing.assert_allclose(got, [30_000.0], rtol=1e-12)

    def test_multi_period_window_requires_the_history(self):
        rule = _rule(income_denominator="geometric_average", income_denominator_window=4)
        with pytest.raises(ValueError, match="requires historic_deflator"):
            rule._geometric_average_income(np.ones((4, 2)), 1.0, None)

    def test_misaligned_history_is_rejected(self):
        rule = _rule(income_denominator="geometric_average", income_denominator_window=4)
        with pytest.raises(ValueError, match="align period-for-period"):
            rule._geometric_average_income(np.ones((4, 2)), 1.0, np.ones(3))
