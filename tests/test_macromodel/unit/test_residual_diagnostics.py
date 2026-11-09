"""Native observer non-interference, capture and lifecycle checks on synthetic data."""

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "run_model"))
from src.experiments.paired_checkpoint import PairedCheckpoint, _RandomState  # noqa: E402
from src.experiments.residual_diagnostics import ResidualDiagnosticCapture  # noqa: E402

from macromodel.agents.households.func.financial_feasibility import PostGrantFeasiblePlan  # noqa: E402
from tests.test_macromodel.unit import test_two_pass_diagnostics as fixtures  # noqa: E402


@pytest.fixture
def observed_model(datawrapper, monkeypatch):
    return fixtures.integrated_model.__wrapped__(datawrapper, monkeypatch)


def plan(n):
    return PostGrantFeasiblePlan(
        credit_granted=np.full(n, 2.0),
        credit_rationing_gap=np.zeros(n),
        planned_liquidation_total=np.full(n, 3.0),
        residual_shortfall_after_granted_credit=np.full(n, 5.0),
        residual_shortfall_after_lfa=np.full(n, 10.0),
        reserved_liquidation_total=np.full(n, 3.0),
    )


def sequence(model):
    country = model.countries["FRA"]
    hh = country.households
    country._set_household_target_demand(replace_current=False)
    hh.compute_target_credit(None)
    hh.ts.override_current("expected_income_employee", hh.ts.current("expected_income_employee") * 1.2)
    country._set_household_target_demand(replace_current=True)
    # Direct seams omit housing/investment clearing; supply explicit zero cash uses.
    count = hh.ts.current("n_households")
    hh.ts.override_current("price_paid_for_property", np.zeros(count))
    hh.ts.override_current("received_mortgages", np.zeros(count))
    hh.ts.override_current("target_investment", np.zeros_like(hh.ts.current("target_investment")))
    hh.post_grant_feasible_plan = plan(hh.ts.current("n_households"))
    hh.refresh_post_labour_feasibility(np.full(hh.ts.current("n_households"), 4.0))
    hh.apply_consumption_floor_to_post_grant_plan(
        consumption_before_floor=hh.consumption_cash_factor() * hh.ts.current("target_consumption").sum(1),
        subsistence_floor=np.full(hh.ts.current("n_households"), 3.0),
    )


@pytest.mark.parametrize("policy", ["initial_state", "fixed_effect"])
def test_native_target_funding_and_floor_are_transparent(observed_model, policy):
    rule = observed_model.countries["FRA"].households.functions["consumption"]
    rule.idiosyncratic_persistence = policy
    checkpoint = PairedCheckpoint(observed_model)
    control, observed = checkpoint.clone(), checkpoint.clone()
    with checkpoint.activate_rng():
        sequence(control)
    before = pickle.dumps(_RandomState.capture())
    with checkpoint.activate_rng():
        with ResidualDiagnosticCapture(observed) as capture:
            assert observed.countries["FRA"].households.functions["consumption"]._household_epsilon is None
            with pytest.raises(ValueError, match="before installing hooks"):
                PairedCheckpoint(observed)
            sequence(observed)
    assert pickle.dumps(_RandomState.capture()) == before
    a, b = observed.countries["FRA"].households, control.countries["FRA"].households
    for key in a.ts.dicts:
        np.testing.assert_array_equal(np.asarray(a.ts.historic(key)), np.asarray(b.ts.historic(key)))
    assert not observed.prehooks and not hasattr(observed, "_residual_diagnostic_capture")
    for owner, names in [
        (a, ["compute_target_credit", "refresh_post_labour_feasibility", "current_financing_income"]),
        (a.financial_feasibility, ["refresh_cash_needs", "settle_consumption_floor"]),
        (a.functions["consumption"], ["compute_target_consumption"]),
    ]:
        assert all(name not in vars(owner) for name in names)
    checkpoint2 = PairedCheckpoint(observed)
    clone = checkpoint2.clone()
    assert clone.countries["FRA"].households is not a
    targets = [r for r in capture.records if r["event"] == "target"]
    assert len(targets) == 2 and targets[0]["epsilon_before"] is None
    np.testing.assert_array_equal(targets[0]["epsilon_after"], targets[1]["epsilon_before"])
    assert targets[0]["inputs"]["consumption_inputs"]["pass_kind"] == "first"
    assert targets[1]["inputs"]["consumption_inputs"]["pass_kind"] == "replacement"
    assert [r["event"] for r in capture.records if r["event"] != "scheduled_service"] == [
        "startup",
        "target",
        "credit_request",
        "target",
        "cash_refresh",
        "funding_components",
        "floor",
    ]
    cash = next(r for r in capture.records if r["event"] == "cash_refresh")
    components = next(r for r in capture.records if r["event"] == "funding_components")
    getters = components["native_getter_results"]
    for name in (
        "current_financing_income",
        "current_consumption_cash_rent",
        "target_investment_cash",
        "consumption_cash_factor",
    ):
        assert len(getters[name]) == 1
    raw = components["state_before"]
    args = components["inputs"]
    np.testing.assert_allclose(
        cash["inputs"]["cash_income"],
        getters["current_financing_income"][0]
        + raw["received_mortgages"]
        + raw["property_sale_proceeds"]
        - raw["mortgage_sale_repayment"],
    )
    np.testing.assert_allclose(
        cash["inputs"]["cash_uses"],
        getters["consumption_cash_factor"][0] * raw["target_consumption"].sum(1)
        + getters["current_consumption_cash_rent"][0]
        + getters["target_investment_cash"][0]
        + args["scheduled_debt_service"]
        + raw["price_paid_for_property"],
    )
    floor = next(r for r in capture.records if r["event"] == "floor")
    np.testing.assert_array_equal(
        floor["inputs"]["plan"]["residual_shortfall_after_granted_credit"],
        cash["output"]["residual_shortfall_after_granted_credit"],
    )
    with pytest.raises(ValueError):
        targets[0]["epsilon_after"].setflags(write=True)
    with pytest.raises(TypeError):
        targets[0]["inputs"]["current_cpi"] = 0
    saved = targets[0]["epsilon_after"].copy()
    a.functions["consumption"]._household_epsilon[:] = 999
    np.testing.assert_array_equal(targets[0]["epsilon_after"], saved)


def test_floor_captures_native_cut_and_derived_topup(observed_model):
    country = observed_model.countries["FRA"]
    feasibility = country.households.financial_feasibility
    native = plan(3)
    # Frozen dataclass replacement is the native carrier operation.
    from dataclasses import replace

    native = replace(native, residual_shortfall_after_granted_credit=np.array([0.0, 3.0, 30.0]))
    with ResidualDiagnosticCapture(observed_model) as capture:
        result = feasibility.settle_consumption_floor(
            native, consumption_before_floor=np.array([2.0, 20.0, 20.0]), subsistence_floor=np.full(3, 5.0)
        )
    record = capture.records[-1]
    np.testing.assert_array_equal(record["output"]["consumption_cut_amount"], [0, 3, 15])
    np.testing.assert_array_equal(record["floor_top_up_derived"], [3, 0, 0])
    np.testing.assert_array_equal(record["output"]["consumption_after_floor"], result.consumption_after_floor)


def test_native_failure_restores_overrides_and_hook(observed_model):
    hh = observed_model.countries["FRA"].households
    original = hh.financial_feasibility.refresh_cash_needs

    def fail(*args, **kwargs):
        raise RuntimeError("native failure")

    hh.financial_feasibility.refresh_cash_needs = fail
    with pytest.raises(RuntimeError, match="native failure"):
        with ResidualDiagnosticCapture(observed_model):
            hh.financial_feasibility.refresh_cash_needs(
                plan(2), cash_uses=np.ones(2), cash_income=np.ones(2), available_lfa=np.ones(2)
            )
    assert hh.financial_feasibility.refresh_cash_needs is fail and not observed_model.prehooks
    hh.financial_feasibility.refresh_cash_needs = original


def test_nested_reentry_and_partial_install_cleanup(observed_model):
    capture = ResidualDiagnosticCapture(observed_model)
    with capture:
        with pytest.raises(RuntimeError, match="single-use"):
            capture.__enter__()
        with pytest.raises(RuntimeError, match="already installed"):
            with ResidualDiagnosticCapture(observed_model):
                pass
    observed_model.countries["BAD"] = SimpleNamespace(households=SimpleNamespace(functions={}))
    with pytest.raises(KeyError):
        with ResidualDiagnosticCapture(observed_model):
            pass
    hh = observed_model.countries["FRA"].households
    assert "compute_target_consumption" not in vars(hh.functions["consumption"])
    assert not observed_model.prehooks and not hasattr(observed_model, "_residual_diagnostic_capture")


def test_nondefault_instance_target_and_error_evidence_restore(observed_model):
    rule = observed_model.countries["FRA"].households.functions["consumption"]
    original = rule.compute_target_consumption
    # Preserve signature for native argument binding.
    from functools import wraps

    @wraps(original)
    def fail(*args, **kwargs):
        raise RuntimeError("target failed")

    rule.compute_target_consumption = fail
    with pytest.raises(RuntimeError, match="target failed"):
        with ResidualDiagnosticCapture(observed_model) as capture:
            observed_model.countries["FRA"]._set_household_target_demand(replace_current=False)
    assert rule.compute_target_consumption is fail
    assert capture.records[-1]["event"] == "target_error"
    assert rule._household_epsilon is None


def test_scheduled_service_and_payment_snapshots(observed_model, monkeypatch):
    country = observed_model.countries["FRA"]
    hh = country.households
    calls = []

    def settle():
        calls.append(1)
        hh.ts.dicts["post_relief_funding_gap"] = [np.array([2.0, 3.0])]

    monkeypatch.setattr(country, "settle_authoritative_household_payments", settle)
    with ResidualDiagnosticCapture(observed_model) as capture:
        scheduled = country.credit_market.compute_opening_scheduled_consumption_payments_by_household()
        country.settle_authoritative_household_payments()
    assert calls == [1]
    service = next(r for r in capture.records if r["event"] == "scheduled_service")
    np.testing.assert_array_equal(service["values"], scheduled)
    closing = next(r for r in capture.records if r["event"] == "payments_settled")
    np.testing.assert_array_equal(closing["household"]["post_relief_funding_gap"]["value"], [2, 3])
    assert closing["household"]["post_relief_funding_gap"]["history_length"] == 1
    assert country.settle_authoritative_household_payments is settle


def test_membership_change_and_other_thread_rejected(observed_model):
    from concurrent.futures import ThreadPoolExecutor

    country = observed_model.countries["FRA"]
    with ResidualDiagnosticCapture(observed_model):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(country.credit_market.compute_opening_scheduled_consumption_payments_by_household)
            with pytest.raises(RuntimeError, match="process/thread"):
                future.result()
        country.households.states["corr_individuals"].reverse()
        with pytest.raises(ValueError, match="membership/order"):
            country.credit_market.compute_opening_scheduled_consumption_payments_by_household()
