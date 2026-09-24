"""Continuous calibration v3: the gamma_4(B) map.

v1 and v2 held gamma_4 homogeneous and strictly positive, so housing wealth
could only ever raise consumption. v3 fits it as a third logistic over a range
that spans zero. These lock down the properties that distinguish v3 from v2 and
that a plausible future edit could silently break -- above all that an
unconfigured gamma_4 range still reproduces v2 exactly.
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

V3 = {
    "index_construction": "raw_ratio",
    "weight_net_liquid_assets": 0.70,
    "weight_illiquid_financial_assets": 0.10,
    "weight_housing_assets": 0.20,
    "alpha_2_steepness": 43.9445,
    "alpha_2_midpoint": 0.3951,
    "gamma_1_steepness": 43.9445,
    "gamma_1_midpoint": 0.1466,
    "gamma_4_steepness": 43.9445,
    "gamma_4_midpoint": 0.2771,
    "alpha_2_low": 0.25,
    "alpha_2_high": 0.70,
    "gamma_1_low": 0.007,
    "gamma_1_high": 0.25,
    "gamma_4_low": -0.021,
    "gamma_4_high": 0.021,
    "net_liquid_assets_ratio_bounds": (-6.7304, 4.1814),
    "illiquid_financial_assets_ratio_bounds": (0.0, 8.5902),
    "housing_assets_ratio_bounds": (0.0, 23.5064),
    "b_raw_min": -4.2232,
    "b_raw_max": 7.6982,
}


def _rule(calibration=None, **kwargs):
    params = dict(
        uses_continuous_wealth_calibration=True,
        continuous_wealth_calibration=V3 if calibration is None else calibration,
    )
    params.update(kwargs)
    return CreditAugmentedConsumption(**params)


def _ratios(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(-6.7304, 4.1814, n),
        rng.uniform(0.0, 8.5902, n),
        rng.uniform(0.0, 23.5064, n),
    )


class TestGamma4Map:
    def test_gamma_4_falls_in_b(self):
        """Decreasing in B, like gamma_1: less accessible -> larger response.

        Asserted as monotonicity, not linear correlation: at k = 43.9 the map
        saturates at both ends, so Pearson r understates it (~-0.72) even though
        the function is exactly non-increasing.
        """
        nla, ifa, ha = _ratios()
        _, _, gamma_4 = _rule()._compute_continuous_wealth_calibration(nla, ifa, ha)
        b = 0.70 * nla + 0.10 * ifa + 0.20 * ha
        ordered = gamma_4[np.argsort(b)]
        assert np.all(np.diff(ordered) <= 1e-15)

    def test_gamma_4_spans_zero(self):
        """The whole point of v3: housing wealth can DRAG consumption down."""
        nla, ifa, ha = _ratios()
        _, _, gamma_4 = _rule()._compute_continuous_wealth_calibration(nla, ifa, ha)
        assert gamma_4.min() < 0.0 < gamma_4.max()

    def test_gamma_4_stays_inside_its_configured_range(self):
        nla, ifa, ha = _ratios()
        _, _, gamma_4 = _rule()._compute_continuous_wealth_calibration(nla, ifa, ha)
        # Tolerance is for float64 round-off where the logistic saturates, not
        # for genuine slack.
        tol = 1e-12
        assert gamma_4.min() >= -0.021 - tol
        assert gamma_4.max() <= 0.021 + tol

    def test_gamma_4_matches_the_closed_form(self):
        nla, ifa, ha = _ratios()
        _, _, gamma_4 = _rule()._compute_continuous_wealth_calibration(nla, ifa, ha)
        b_raw = (
            0.70 * np.clip(nla, -6.7304, 4.1814) + 0.10 * np.clip(ifa, 0.0, 8.5902) + 0.20 * np.clip(ha, 0.0, 23.5064)
        )
        b = np.clip((b_raw - (-4.2232)) / (7.6982 - (-4.2232)), 0.0, 1.0)
        logistic = 1.0 / (1.0 + np.exp(-43.9445 * (b - 0.2771)))
        np.testing.assert_allclose(gamma_4, 0.021 - (0.021 - (-0.021)) * logistic)


class TestBackwardCompatibility:
    def test_unconfigured_gamma_4_falls_back_to_the_scalar(self):
        """A v2 config must keep gamma_4 homogeneous at housing_wealth_propensity."""
        nla, ifa, ha = _ratios()
        rule = _rule(calibration=V2, housing_wealth_propensity=0.009)
        _, _, gamma_4 = rule._compute_continuous_wealth_calibration(nla, ifa, ha)
        np.testing.assert_allclose(gamma_4, 0.009)

    def test_v2_config_target_is_unchanged_by_the_v3_code_path(self):
        """The gamma_4 map is opt-in: v2 constants must reproduce v2 numbers."""
        nla, ifa, ha = _ratios(n=500)
        rule = _rule(calibration=V2, housing_wealth_propensity=0.009)
        alpha_2, gamma_1, gamma_4 = rule._compute_continuous_wealth_calibration(nla, ifa, ha)
        # wealth_drag as _evaluate_target now forms it, against the pre-change
        # scalar formulation.
        drag_new = gamma_1 * nla + 0.02 * ifa + gamma_4 * ha
        drag_old = gamma_1 * nla + 0.02 * ifa + 0.009 * ha
        np.testing.assert_allclose(drag_new, drag_old)


class TestConfigurationValidation:
    @pytest.mark.parametrize(
        "calibration, match",
        [
            (dict(V3, gamma_4_low=0.021, gamma_4_high=-0.021), "gamma_4 calibration range"),
            (dict(V3, gamma_4_low=0.0, gamma_4_high=0.0), "gamma_4 calibration range"),
        ],
    )
    def test_inverted_or_degenerate_range_is_rejected(self, calibration, match):
        with pytest.raises(ValueError, match=match):
            _rule(calibration=calibration)

    def test_half_configured_map_is_rejected(self):
        """Supplying one endpoint only is a config error, not a partial map."""
        calibration = {k: v for k, v in V3.items() if k != "gamma_4_high"}
        with pytest.raises(ValueError, match="must be supplied together"):
            _rule(calibration=calibration)

    @pytest.mark.parametrize("dropped", ["gamma_4_steepness", "gamma_4_midpoint"])
    def test_shape_without_a_range_is_rejected(self, dropped):
        """Shape keys with no range would be stored, never read, and silently ignored."""
        calibration = {k: v for k, v in V3.items() if k not in ("gamma_4_low", "gamma_4_high")}
        calibration = {k: v for k, v in calibration.items() if k != dropped}
        with pytest.raises(ValueError, match="would be silently ignored"):
            _rule(calibration=calibration)

    def test_a_v2_config_with_no_gamma_4_keys_at_all_is_still_accepted(self):
        """The guard must not break the supported v1/v2 fallback path."""
        rule = _rule(calibration=V2, housing_wealth_propensity=0.009)
        assert rule.continuous_wealth_calibration_gamma_4_range is None


class TestSteepnessSignValidation:
    """A non-positive steepness inverts or flattens the map as surely as an inverted range."""

    @pytest.mark.parametrize("key", ["alpha_2_steepness", "gamma_1_steepness", "gamma_4_steepness"])
    @pytest.mark.parametrize("bad", [-43.9445, 0.0])
    def test_non_positive_steepness_is_rejected(self, key, bad):
        with pytest.raises(ValueError, match="steepness must be positive"):
            _rule(calibration=dict(V3, **{key: bad}))

    def test_the_shipped_v3_constants_pass_validation(self):
        rule = _rule()
        assert rule.continuous_wealth_calibration_gamma_4_steepness > 0.0
