"""B+F: quintile intercepts and scaled/iid residual at the initial_state back-out (opt-in)."""

import numpy as np
import pytest

from macromodel.agents.households.func.consumption import CreditAugmentedConsumption
from tests.test_macromodel.unit.test_agents.test_households.func.test_consumption_calibration_v2 import V2

N = 600
SIGMA = 0.41


def _rule(**kwargs):
    params = dict(
        uses_continuous_wealth_calibration=True,
        continuous_wealth_calibration=V2,
        idiosyncratic_persistence="initial_state",
        idiosyncratic_sd=0.3308,
    )
    params.update(kwargs)
    return CreditAugmentedConsumption(**params)


def _bf(**kwargs):
    return _rule(initial_state_intercepts="income_quintile", initial_state_residual_sd=SIGMA, **kwargs)


def _data(seed=3, fallback=0):
    rng = np.random.default_rng(seed)
    income = np.exp(rng.normal(4.6, 0.6, N))
    quintile = np.argsort(np.argsort(income)) * 5 // N
    quintile = quintile.astype(float)
    quintile[rng.choice(N, fallback, replace=False)] = -1.0
    # Opening consumption with a quintile-specific level shift and a household residual.
    lagged = income * np.exp(rng.normal(-0.2, 0.5, N) + 0.1 * quintile)
    wealth = np.exp(rng.normal(4.0, 1.0, N))
    return income, lagged, wealth, quintile


def _kwargs(income, lagged, wealth, time=1):
    return dict(
        expected_inflation=0.0,
        current_cpi=1.0,
        initial_cpi=1.0,
        historic_consumption_sum=lagged[None, :],
        saving_rates=np.zeros(income.size),
        income=income,
        household_benefits=np.zeros(income.size),
        consumption_weights=np.array([1.0]),
        consumption_weights_by_income=np.array([[1.0]]),
        exogenous_total_consumption=np.zeros(income.size),
        current_time=time,
        take_consumption_weights_by_income_quantile=False,
        tau_vat=0.0,
        liquid_wealth=wealth,
        lagged_real_consumption_budget=lagged,
    )


def _step(rule, data, time=1, with_quintile=True):
    income, lagged, wealth, quintile = data
    extra = {"initial_income_quintile": quintile} if with_quintile else {}
    return rule.compute_target_consumption(**_kwargs(income, lagged, wealth, time), **extra)


def _terms(rule):
    """x_i, b_i, C_prev_i, eps_i of the last evaluation."""
    c = rule.last_target_consumption_components
    eps = c["target_consumption_epsilon"]
    return (
        c["target_consumption_log_long_run"] - eps,
        c["target_consumption_real_income"],
        c["target_consumption_real_lagged_consumption"],
        eps,
    )


def _groups(quintile):
    return np.where(quintile < 0, 5, quintile).astype(int)


def _group_totals(rule, quintile):
    x, b, c_prev, eps = _terms(rule)
    g = _groups(quintile)
    model = np.array([np.sum(np.exp(x + eps)[g == k] * b[g == k]) for k in range(6)])
    opening = np.array([np.sum(c_prev[g == k]) for k in range(6)])
    return model, opening


class TestDefaultsOff:
    def test_defaults_reproduce_plain_backout_bitwise(self):
        data = _data()
        plain, off = _rule(), _rule()
        _step(plain, data, with_quintile=False)
        _step(off, data, with_quintile=False)
        assert np.array_equal(plain._household_epsilon, off._household_epsilon)
        # Reference: the documented plain back-out, computed independently of the new code paths.
        x, b, c_prev, _ = _terms(plain)
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = np.log(c_prev) - np.log(b) - x
        np.testing.assert_allclose(plain._household_epsilon, expected, rtol=0, atol=1e-12)
        assert plain.initial_state_diagnostics == {}

    def test_defaults_ignore_a_supplied_quintile(self):
        data = _data()
        with_q, without = _rule(), _rule()
        _step(with_q, data, with_quintile=True)
        _step(without, data, with_quintile=False)
        assert np.array_equal(with_q._household_epsilon, without._household_epsilon)

    def test_intercepts_alone_keep_the_plain_backout_bitwise(self):
        data = _data(fallback=5)
        plain, split = _rule(), _rule(initial_state_intercepts="income_quintile")
        _step(plain, data, with_quintile=False)
        _step(split, data)
        assert np.array_equal(plain._household_epsilon, split._household_epsilon)
        model, opening = _group_totals(split, data[3])
        np.testing.assert_allclose(model, opening, rtol=1e-10)


class TestFixedResidual:
    @pytest.mark.parametrize("fallback", [0, 7])
    def test_group_totals_equal_opening_totals_exactly(self, fallback):
        data = _data(fallback=fallback)
        rule = _bf()
        _step(rule, data)
        model, opening = _group_totals(rule, data[3])
        np.testing.assert_allclose(model, opening, rtol=1e-12)
        # Aggregate anchor follows.
        assert model.sum() == pytest.approx(opening.sum(), rel=1e-12)

    def test_pooled_residual_sd_equals_target(self):
        data = _data(fallback=7)
        rule = _bf()
        _step(rule, data)
        x, b, c_prev, eps = _terms(rule)
        g = _groups(data[3])
        demeaned = np.concatenate([eps[g == k] - eps[g == k].mean() for k in range(6)])
        assert np.sqrt(np.mean(demeaned**2)) == pytest.approx(SIGMA, rel=1e-10)

    def test_within_group_rank_order_is_preserved(self):
        data = _data(fallback=7)
        plain, rule = _rule(), _bf()
        _step(plain, data)
        _step(rule, data)
        g = _groups(data[3])
        for k in range(6):
            m = g == k
            assert np.array_equal(np.argsort(plain._household_epsilon[m]), np.argsort(rule._household_epsilon[m]))

    def test_kappa_and_epsilon_are_frozen_after_t1(self):
        data = _data()
        rule = _bf()
        _step(rule, data, time=1)
        eps, kappa = rule._household_epsilon.copy(), rule.initial_state_diagnostics["kappa"].copy()
        # Later period: different lagged consumption must not move the frozen split.
        income, lagged, wealth, quintile = data
        later = (income * 1.3, lagged * 0.7, wealth, quintile)
        _step(rule, later, time=2)
        assert np.array_equal(rule._household_epsilon, eps)
        assert np.array_equal(rule.initial_state_diagnostics["kappa"], kappa)

    def test_fallback_households_form_their_own_group(self):
        data = _data(fallback=9)
        rule = _bf()
        _step(rule, data)
        assert rule.initial_state_diagnostics["count"].tolist() == [*np.bincount(_groups(data[3]), minlength=6)]
        assert rule.initial_state_diagnostics["count"][5] == 9
        # Quintile 0 is untouched by the fallback group: its kappa matches a run without them in the data.
        model, opening = _group_totals(rule, data[3])
        np.testing.assert_allclose(model[5], opening[5], rtol=1e-12)

    def test_invalid_households_get_zero_eps_and_are_excluded_from_sums(self):
        """Non-positive C_prev or b_i (log non-finite) is excluded from every sum and keeps eps = 0."""
        rng = np.random.default_rng(1)
        n = 100
        quintile = (np.arange(n) * 5 // n).astype(float)
        c_prev, b, x = np.exp(rng.normal(0, 0.5, n)), np.exp(rng.normal(0, 0.5, n)), rng.normal(-0.2, 0.3, n)
        c_prev[[3, 10]] = [0.0, -1.0]
        b[40] = 0.0
        components = {
            "target_consumption_real_lagged_consumption": c_prev,
            "target_consumption_real_income": b,
            "target_consumption_log_long_run": x,
        }
        with np.errstate(divide="ignore", invalid="ignore"):
            plain = np.log(c_prev) - np.log(b) - x
        plain = np.where(np.isfinite(plain), plain, 0.0)
        eps = _bf()._initial_state_split_epsilon(components, plain, quintile, 1)
        bad = [3, 10, 40]
        assert np.all(eps[bad] == 0.0)
        g, ok = _groups(quintile), np.ones(n, dtype=bool)
        ok[bad] = False
        for k in range(5):
            m = (g == k) & ok
            assert np.sum(np.exp(x + eps)[m] * b[m]) == pytest.approx(np.sum(c_prev[m]), rel=1e-12)

    def test_zero_dispersion_cannot_be_rescaled(self):
        quintile = _data()[3]
        rule = _bf()
        # eps identical for every household: no within-group dispersion to rescale.
        with pytest.raises(ValueError, match="within-group dispersion"):
            rule._initial_state_split_epsilon(
                {
                    "target_consumption_real_lagged_consumption": np.ones(N),
                    "target_consumption_real_income": np.ones(N),
                    "target_consumption_log_long_run": np.zeros(N),
                },
                np.zeros(N),
                quintile,
                1,
            )

    def test_missing_or_malformed_quintile_is_rejected(self):
        data = _data()
        with pytest.raises(ValueError, match="requires initial_income_quintile"):
            _step(_bf(), data, with_quintile=False)
        bad = (*data[:3], data[3] + 7.0)
        with pytest.raises(ValueError, match="one of -1, 0..4"):
            _step(_bf(), bad)

    def test_set_run_seed_clears_the_split_state(self):
        rule = _bf()
        _step(rule, _data())
        rule.set_run_seed(5)
        assert rule._household_epsilon is None and rule._initial_state_kappa is None
        assert rule.initial_state_diagnostics == {}


class TestIidResidual:
    def _iid(self, **kwargs):
        return _bf(initial_state_residual_persistence="iid", **kwargs)

    def test_same_draw_within_a_period_new_draw_across_periods(self):
        data = _data()
        rule = self._iid()
        _step(rule, data, time=1)
        first = rule._household_epsilon.copy()
        _step(rule, data, time=1)  # replacement pass and MPC probe in the same period
        assert np.array_equal(rule._household_epsilon, first)
        _step(rule, data, time=2)
        assert not np.array_equal(rule._household_epsilon, first)
        second = rule._household_epsilon.copy()
        _step(rule, data, time=2)
        assert np.array_equal(rule._household_epsilon, second)

    def test_reproducible_across_instances_with_the_same_seed(self):
        data = _data()
        runs = []
        for _ in range(2):
            rule = self._iid()
            rule.set_run_seed(11)
            path = []
            for t in (1, 1, 2, 3):
                _step(rule, data, time=t)
                path.append(rule._household_epsilon.copy())
            runs.append(path)
        for a, b in zip(*runs):
            assert np.array_equal(a, b)
        other = self._iid()
        other.set_run_seed(12)
        _step(other, data, time=1)
        assert not np.array_equal(other._household_epsilon, runs[0][0])

    def test_global_numpy_stream_is_untouched(self):
        data = _data()
        rule = self._iid()
        np.random.seed(99)
        expected = np.random.rand(3)
        np.random.seed(99)
        for t in (1, 2, 3):
            _step(rule, data, time=t)
        np.testing.assert_array_equal(np.random.rand(3), expected)

    def test_kappa_fixed_from_t1_and_draw_moments(self):
        data = _data()
        rule = self._iid()
        _step(rule, data, time=1)
        kappa = rule.initial_state_diagnostics["kappa"].copy()
        g = _groups(data[3])
        draws = []
        for t in range(2, 42):
            # Even with a drifting state the intercepts stay at their t1 value.
            _step(rule, (data[0] * (1 + 0.01 * t), data[1] * 0.9, data[2], data[3]), time=t)
            draws.append(rule._household_epsilon - kappa[g] + 0.5 * SIGMA**2)
        assert np.array_equal(rule.initial_state_diagnostics["kappa"], kappa)
        draws = np.concatenate(draws)
        assert draws.mean() == pytest.approx(0.0, abs=0.01)
        assert draws.std() == pytest.approx(SIGMA, rel=0.02)

    def test_first_period_uses_the_same_draw_as_later_passes(self):
        data = _data()
        rule = self._iid()
        _step(rule, data, time=1)
        eps_t1 = rule._household_epsilon.copy()
        x, b, c_prev, eps = _terms(rule)
        np.testing.assert_array_equal(eps, eps_t1)


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs, message",
        [
            (dict(initial_state_intercepts="decile"), "initial_state_intercepts"),
            (dict(initial_state_residual_persistence="ar1"), "initial_state_residual_persistence"),
            (dict(initial_state_residual_sd=0.0), "finite positive"),
            (dict(initial_state_residual_sd=-0.4), "finite positive"),
            (dict(initial_state_residual_sd=float("nan")), "finite positive"),
            (dict(initial_state_residual_sd=True), "finite positive"),
            (dict(initial_state_residual_sd=0.41), "requires initial_state_intercepts"),
            (dict(initial_state_intercepts="income_quintile", initial_state_residual_persistence="iid"), "requires"),
            (dict(initial_state_residual_persistence="iid"), "requires"),
        ],
    )
    def test_invalid_combinations_are_rejected(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            _rule(**kwargs)

    @pytest.mark.parametrize("persistence", ["fixed_effect", "iid"])
    def test_options_require_initial_state_persistence(self, persistence):
        with pytest.raises(ValueError, match="require idiosyncratic_persistence='initial_state'"):
            CreditAugmentedConsumption(
                idiosyncratic_persistence=persistence, initial_state_intercepts="income_quintile"
            )

    def test_defaults_are_off(self):
        rule = _rule()
        assert rule.initial_state_intercepts == "none"
        assert rule.initial_state_residual_sd is None
        assert rule.initial_state_residual_persistence == "fixed"


class TestOpeningQuintilePlumbing:
    """The allocator's quintile reaches the rule unchanged and is persisted once."""

    def _opening(self, income=None, dna=True):
        from macromodel.util.initial_consumption import initialise_hfcs_consumption
        from tests.test_macromodel.unit.test_initial_hfcs_consumption import country

        c = country()
        if income is not None:
            c.households.ts.override_current("expected_behavioural_income_raw", np.array(income))
        kwargs = dict(dna_quintile_c_y=[1.0] * 5, household_size=np.array([1.0, 2.0])) if dna else {}
        result = initialise_hfcs_consumption(c, np.array([0.6, 0.9]), **kwargs)
        return c.households, result

    def test_persists_allocator_quintile(self):
        h, result = self._opening()
        np.testing.assert_array_equal(h.ts.initial("initial_hfcs_income_quintile"), [0.0, 2.0])
        np.testing.assert_array_equal(result.quintile, [0, 2])

    def test_fallback_households_are_minus_one(self):
        h, _ = self._opening(income=[0.0, 200.0])
        np.testing.assert_array_equal(h.ts.initial("initial_hfcs_income_quintile"), [-1.0, 0.0])

    def test_not_persisted_without_dna_allocation(self):
        h, result = self._opening(dna=False)
        assert "initial_hfcs_income_quintile" not in h.ts.dicts and result.quintile is None

    def test_households_hook_fails_loudly_without_the_allocation(self):
        from types import SimpleNamespace

        from macromodel.agents.households.households import Households
        from macromodel.timeseries import TimeSeries

        with pytest.raises(ValueError, match="dna_quintile_c_y"):
            Households._initial_income_quintile(SimpleNamespace(ts=TimeSeries(x=np.zeros(1))))
        h, _ = self._opening()
        np.testing.assert_array_equal(Households._initial_income_quintile(h), [0.0, 2.0])
