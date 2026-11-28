"""income_belief_signal="behavioural": the permanent-income term runs on full behavioural income y."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macromodel.agents.households.func.consumption import CreditAugmentedConsumption
from macromodel.agents.households.income_belief_learning import compute_income_belief_learning_outputs
from macromodel.country import Country
from macromodel.timeseries import TimeSeries
from macromodel.util.behavioural_income import reconcile_behavioural_income
from macromodel.util.permanent_monetary_income import monetary_income_moments

GUARD = np.ones(3)
# Three households: well above the guard, with financial income, and below the guard.
WAGES = np.array([100.0, 50.0, 0.5])
FINANCIAL = np.array([0.0, 20.0, 0.0])
LOAN_INTEREST = np.array([2.0, 0.0, 0.0])
TAXES = np.array([1.0, 0.0, 0.0])
ZERO = np.zeros(3)


def _reconcile(wages, financial=FINANCIAL):
    return reconcile_behavioural_income(
        non_property=wages,
        rental=ZERO,
        financial=financial,
        deposit_receipts=ZERO,
        borrower_interest=LOAN_INTEREST,
        imputed_rent=ZERO,
        subsistence=GUARD,
        housing_taxes=TAXES,
    )


def _country(signal="behavioural"):
    """Stub country holding t0 rows; ``add_period`` appends realised and expected rows."""
    consumption = SimpleNamespace(income_belief_signal=signal)
    country = Country.__new__(Country)
    country._staged_household_income_shock = None
    country.start_period = pd.Period("2014Q1", freq="Q")
    rows = {
        "income_employee": WAGES,
        "income_social_transfers": ZERO,
        "income_financial_assets": FINANCIAL,
        "income_rental": ZERO,
        "housing_taxes_property": TAXES,
        "housing_taxes_residence": ZERO,
        "interest_paid_on_loans": LOAN_INTEREST,
        "interest_paid_on_deposits": ZERO,
        "expected_income_employee": WAGES,
        "expected_income_social_transfers": ZERO,
        "expected_income_financial_assets": FINANCIAL,
        **{
            name: ZERO
            for name in (
                "income_bridge_current",
                "income_bridge_property",
                "income_bridge_floor_used",
                "income_bridge_component_floor_used",
            )
        },
    }
    country.households = SimpleNamespace(
        ts=TimeSeries(**{name: values.copy() for name, values in rows.items()}),
        functions={"wealth": object(), "consumption": consumption},
        compute_interest_paid_on_deposits=lambda **kwargs: np.zeros(3),
        current_consumption_imputed_rent=lambda: np.zeros(3),
    )
    ts = country.households.ts
    country.households.compute_non_property_income = lambda: (
        ts.current("income_employee") + ts.current("income_social_transfers")
    )
    country.banks = SimpleNamespace(
        ts=TimeSeries(interest_rate_on_household_deposits=np.zeros(1), overdraft_rate_on_household_deposits=np.zeros(1))
    )
    country.credit_market = SimpleNamespace(preview_opening_household_interest=lambda: LOAN_INTEREST.copy())
    country.economy = SimpleNamespace(
        ts=TimeSeries(
            subsistence_consumption=GUARD.copy(),
            income_guard=GUARD.copy(),
            cpi=np.ones(1),
            cpi_fixed_basket=np.ones(1),
            unemployment_rate=np.array([0.1]),
        ),
        consumer_price_level_series_name=lambda: "cpi",
    )
    country.individuals = SimpleNamespace(ts=TimeSeries(n_individuals=[10]))
    country.central_bank = SimpleNamespace(ts=TimeSeries(policy_rate=[0.01]))
    country._record_behavioural_income(expected=True, initial=True)
    country._record_behavioural_income(expected=False, initial=True)
    return country


def _add_period(country, wages=WAGES, price=1.0, shock=None):
    ts = country.households.ts
    for name, values in {
        "income_employee": wages,
        "income_social_transfers": ZERO,
        "income_financial_assets": FINANCIAL,
        "income_rental": ZERO,
        "housing_taxes_property": TAXES,
        "housing_taxes_residence": ZERO,
        "interest_paid_on_loans": LOAN_INTEREST,
        "interest_paid_on_deposits": ZERO,
        "expected_income_employee": wages,
        "expected_income_social_transfers": ZERO,
        "expected_income_financial_assets": FINANCIAL,
    }.items():
        ts.dicts[name].append(values.copy())
    country.economy.ts.dicts["cpi"].append(np.array([price]))
    country.economy.ts.cpi_fixed_basket.append(np.array([price]))
    country.economy.ts.unemployment_rate.append(np.array([0.1]))
    country.individuals.ts.n_individuals.append([10])
    country.central_bank.ts.policy_rate.append([0.01])
    country.economy.ts.income_guard.append(GUARD.copy())
    country._staged_household_income_shock = shock
    country._record_behavioural_income(expected=True)
    country._record_behavioural_income(expected=False)
    country._staged_household_income_shock = None


def _push_signal_rows(country, rows, bound=None, prices=None):
    """Replace the signal history with explicit rows (one price per row)."""
    ts = country.households.ts
    ts["behavioural_income_signal"] = rows[0].copy()
    ts["behavioural_income_signal_guard_bound"] = (np.zeros(3) if bound is None else bound[0]).copy()
    country.economy.ts["cpi"] = np.array([1.0 if prices is None else prices[0]])
    for k, row in enumerate(rows[1:], start=1):
        ts.dicts["behavioural_income_signal"].append(row.copy())
        ts.dicts["behavioural_income_signal_guard_bound"].append((np.zeros(3) if bound is None else bound[k]).copy())
        country.economy.ts.dicts["cpi"].append(np.array([1.0 if prices is None else prices[k]]))


# ---- flag -------------------------------------------------------------------------------------------------------


def test_flag_defaults_to_non_property_and_rejects_unknown_value():
    assert CreditAugmentedConsumption().income_belief_signal == "non_property"
    with pytest.raises(ValueError, match="income_belief_signal"):
        CreditAugmentedConsumption(income_belief_signal="cash")


@pytest.mark.parametrize(
    "aligned, learning", [(False, False), (True, False), (False, True)], ids=["neither", "no-learning", "no-aligned"]
)
def test_behavioural_signal_requires_aligned_income_and_learning(aligned, learning):
    with pytest.raises(ValueError, match="requires uses_aligned_behavioural_income"):
        CreditAugmentedConsumption(
            income_belief_signal="behavioural",
            uses_aligned_behavioural_income=aligned,
            uses_income_belief_learning=learning,
        )
    rule = CreditAugmentedConsumption(
        income_belief_signal="behavioural", uses_aligned_behavioural_income=True, uses_income_belief_learning=True
    )
    assert rule.income_belief_signal == "behavioural"


# ---- signal concept ---------------------------------------------------------------------------------------------


def test_signal_is_realised_behavioural_income_with_guard_flag():
    country = _country()
    _add_period(country)
    ts = country.households.ts
    expected = _reconcile(WAGES)["income"]
    np.testing.assert_allclose(ts.current("behavioural_income_signal"), expected)
    np.testing.assert_allclose(ts.current("behavioural_income_signal"), ts.current("behavioural_income"))
    np.testing.assert_array_equal(ts.current("behavioural_income_signal_guard_bound"), [0.0, 0.0, 1.0])
    # y = sqrt(N (N + P)) - J - housing taxes + H, with P = financial income: not N.
    assert expected[1] == pytest.approx(np.sqrt(50.0 * 70.0))
    assert expected[0] == pytest.approx(100.0 - 2.0 - 1.0)
    assert len(ts.historic("behavioural_income_signal")) == 2


def test_flag_off_creates_no_signal_series():
    country = _country(signal="non_property")
    _add_period(country)
    assert "behavioural_income_signal" not in country.households.ts.dicts
    assert "behavioural_income_signal_guard_bound" not in country.households.ts.dicts


def test_staged_shock_enters_planning_income_but_not_the_signal_or_regressor():
    shock = np.array([40.0, 40.0, 40.0])
    shocked, clean = _country(), _country()
    _add_period(shocked, shock=shock)
    _add_period(clean)
    # y is non-linear in N: the signal is recomputed with pre-shock N, not y minus the shock.
    np.testing.assert_array_equal(
        shocked.households.ts.current("behavioural_income_signal"),
        clean.households.ts.current("behavioural_income_signal"),
    )
    np.testing.assert_allclose(shocked.households.ts.current("behavioural_income"), _reconcile(WAGES + shock)["income"])
    assert not np.allclose(
        shocked.households.ts.current("behavioural_income"), clean.households.ts.current("behavioural_income")
    )
    np.testing.assert_array_equal(
        shocked._permanent_income_simulation_sources().real_pc_income,
        clean._permanent_income_simulation_sources().real_pc_income,
    )


def test_common_regressor_uses_behavioural_income_per_capita_in_real_terms_when_flag_on():
    country = _country()
    _add_period(country, price=2.0)
    y0, y1 = (_reconcile(WAGES)["income"].sum(),) * 2
    sources = country._permanent_income_simulation_sources()
    np.testing.assert_allclose(sources.real_pc_income, [100.0, 100.0 * (y1 / 2.0) / y0])
    # Flag off: the regressor stays on wages + transfers.
    country.households.functions["consumption"].income_belief_signal = "non_property"
    wages = WAGES.sum()
    np.testing.assert_allclose(
        country._permanent_income_simulation_sources().real_pc_income, [100.0, 100.0 * (wages / 2.0) / wages]
    )


# ---- Kalman signal, lag gate, guard -----------------------------------------------------------------------------


def test_lag_gate_first_update_at_t5_for_behavioural_t4_for_non_property():
    rows = [WAGES * (1.0 + 0.01 * k) for k in range(7)]
    for length, behavioural_ready, non_property_ready in [(5, False, True), (6, True, True), (4, False, False)]:
        country = _country()
        _push_signal_rows(country, rows[:length])
        _, lagged, _ = country._income_belief_signal_inputs()
        assert (lagged is not None) == behavioural_ready, length
        country.households.functions["consumption"].income_belief_signal = "non_property"
        for name in ("income_employee", "income_social_transfers"):
            country.households.ts.dicts[name] = [WAGES * (1.0 + 0.01 * k) for k in range(length)]
        _, lagged, _ = country._income_belief_signal_inputs()
        assert (lagged is not None) == non_property_ready, length


def test_signal_is_real_and_lagged_by_four_quarters_with_the_matching_price():
    country = _country()
    rows = [WAGES * (1.0 + 0.1 * k) for k in range(6)]
    prices = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    _push_signal_rows(country, rows, prices=prices)
    current, lagged, missing = country._income_belief_signal_inputs()
    np.testing.assert_allclose(current, rows[5] / 1.5)
    np.testing.assert_allclose(lagged, rows[1] / 1.1)
    assert not missing.any()


def test_households_at_the_guard_now_or_a_year_ago_are_missing_and_get_no_update():
    n = 3
    country = _country()
    rows = [WAGES * 1.0 for _ in range(6)]
    bound = [np.zeros(n) for _ in range(6)]
    bound[5] = np.array([1.0, 0.0, 0.0])  # at the guard now
    bound[1] = np.array([0.0, 1.0, 0.0])  # at the guard a year ago (lag row 1)
    _push_signal_rows(country, rows, bound=bound)
    current, lagged, missing = country._income_belief_signal_inputs()
    np.testing.assert_array_equal(missing, [True, True, False])
    priors = {
        "income_belief_mu": np.full(n, 0.1),
        "income_belief_p": np.full(n, 0.5),
        "income_belief_rho": np.full(n, 0.8),
        "sigma2_xi": np.full(n, 1.0),
        "sigma2_v": np.full(n, 0.2),
    }
    out = compute_income_belief_learning_outputs(
        current_income=current * np.array([1.0, 1.0, 1.1]), lagged_income=lagged, priors=priors, missing=missing
    )
    np.testing.assert_allclose(out.kalman_gain[:2], 0.0)
    np.testing.assert_allclose(out.posterior_mean[:2], out.predicted_mean[:2])
    assert out.kalman_gain[2] > 0
    # The missing households are also excluded from the common component.
    assert out.common_income_growth_signal == pytest.approx(np.log(1.1))
    np.testing.assert_array_equal(out.floor_used, [True, True, False])


def test_common_growth_in_y_does_not_move_beliefs():
    """c_t carries the aggregate; the demeaned signal must leave the idiosyncratic posterior unchanged."""
    n = 3
    priors = {
        "income_belief_mu": np.array([0.02, -0.01, 0.0]),
        "income_belief_p": np.full(n, 0.5),
        "income_belief_rho": np.full(n, 0.8),
        "sigma2_xi": np.full(n, 1.0),
        "sigma2_v": np.full(n, 0.2),
    }
    posteriors = []
    for common_growth in (1.0, 1.05):
        country = _country()
        rows = [WAGES * 0 + np.array([100.0, 60.0, 30.0]) for _ in range(6)]
        rows[5] = np.array([103.0, 58.0, 30.0]) * common_growth
        _push_signal_rows(country, rows)
        current, lagged, missing = country._income_belief_signal_inputs()
        posteriors.append(
            compute_income_belief_learning_outputs(
                current_income=current, lagged_income=lagged, priors=priors, missing=missing
            ).posterior_mean
        )
    np.testing.assert_allclose(posteriors[0], posteriors[1], atol=1e-12)


def test_flag_off_signal_is_the_unchanged_wage_and_transfer_series():
    country = _country(signal="non_property")
    ts = country.households.ts
    n = 3
    ts.dicts["income_employee"] = [WAGES * (1.0 + 0.1 * k) for k in range(6)]
    ts.dicts["income_social_transfers"] = [np.full(n, float(k)) for k in range(6)]
    country.economy.ts["cpi"] = np.array([1.0])
    for price in (1.1, 1.2, 1.3, 1.4, 1.5):
        country.economy.ts.dicts["cpi"].append(np.array([price]))
    current, lagged, missing = country._income_belief_signal_inputs()
    np.testing.assert_array_equal(current, (WAGES * 1.5 + 5.0) / 1.5)
    np.testing.assert_array_equal(lagged, (WAGES * 1.1 + 1.0) / 1.1)
    assert missing is None


# ---- bridge -----------------------------------------------------------------------------------------------------


def test_bridge_on_raw_y_projects_all_of_y_and_t0_seed_matches_it():
    country = _country()
    raw_y = country.households.ts.current("expected_behavioural_income_raw")
    income = country.households.ts.current("expected_behavioural_income")
    bridge = monetary_income_moments(
        non_property=raw_y,
        property_income=np.zeros(3),
        income_floor=GUARD,
        log_mean=np.full(3, np.log(1.1)),
        log_variance=np.zeros(3),
    )
    # M is y itself, and the multiplier scales y (not wages) so that ln(Mp/M) is the learned log ratio.
    np.testing.assert_allclose(bridge["current"], income)
    np.testing.assert_allclose(bridge["mean"][:2], np.log(1.1))
    # The t0 seed carries the same fields.
    ts = country.households.ts
    np.testing.assert_allclose(ts.current("income_bridge_current"), bridge["current"])
    np.testing.assert_array_equal(ts.current("income_bridge_property"), ZERO)
    np.testing.assert_array_equal(ts.current("income_bridge_floor_used"), bridge["floor_used"])
    np.testing.assert_array_equal(ts.current("income_bridge_component_floor_used"), bridge["component_floor_used"])


def test_t0_bridge_seed_is_unchanged_on_the_default_basis():
    country = _country(signal="non_property")
    ts = country.households.ts
    np.testing.assert_allclose(ts.current("income_bridge_current"), np.maximum(WAGES, GUARD))
    np.testing.assert_array_equal(ts.current("income_bridge_floor_used"), [0.0, 0.0, 1.0])
