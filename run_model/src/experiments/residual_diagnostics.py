"""Scoped, read-only residual startup and native funding observations."""

import os
import threading
from dataclasses import fields, is_dataclass
from functools import wraps
from inspect import signature

import numpy as np

from .two_pass_diagnostics import _freeze, _prepared_record

_MISSING = object()
_START = (
    "income",
    "expected_income",
    "expected_behavioural_income",
    "consumption",
    "consumption_cash_expenditure",
    "consumption_including_housing",
    "cacf_real_consumption_budget",
    "liquid_financial_assets",
    "illiquid_financial_assets",
    "wealth_main_residence",
    "mortgage_debt",
    "consumption_loan_debt",
    "rent",
    "rent_imputed",
)


def _record(value):
    if hasattr(value, "household_layout_key") and hasattr(value, "history_window"):
        return _prepared_record(value)
    if hasattr(value, "opening_snapshot") and hasattr(value, "income_base_real"):
        return _prepared_record(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _record(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {key: _record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_record(item) for item in value)
    return value


class ResidualDiagnosticCapture:
    """Observe one bound model serially; detach before checkpoint/reset.

    Native functions are called once. No evaluator, RNG, accounting producer or
    history updater is invoked solely to obtain a diagnostic. A temporary no-op
    prehook makes PairedCheckpoint reject capture until this scope exits.
    """

    def __init__(self, model):
        self.model = model
        self._records = []
        self._restores = []
        self._entered = False
        self._owner = (os.getpid(), threading.get_ident())
        self._funding = {}
        self._memberships = {}
        self._hook = lambda *_args, **_kwargs: None

    @property
    def records(self):
        return tuple(self._records)

    @staticmethod
    def _membership(hh):
        return tuple(tuple(int(member) for member in row) for row in hh.states["corr_individuals"])

    def _check(self, code, country, rule):
        if (os.getpid(), threading.get_ident()) != self._owner:
            raise RuntimeError("Residual capture requires its original process/thread.")
        if self.model.countries[code] is not country or country.households.functions["consumption"] is not rule:
            raise RuntimeError("Do not reset or replace owners inside a residual capture scope.")
        if self._membership(country.households) != self._memberships[code]:
            raise ValueError("Household membership/order changed inside capture.")

    def _append(self, code, country, event, **payload):
        self._records.append(
            _freeze(
                {
                    "country_code": str(code),
                    "period_id": len(country.households.ts.historic("income")),
                    "year": self.model.timestep.year,
                    "month": self.model.timestep.month,
                    "event": event,
                    **_record(payload),
                }
            )
        )

    def _patch(self, owner, name, factory):
        previous = vars(owner).get(name, _MISSING)
        wrapper = factory(getattr(owner, name))
        self._restores.append((owner, name, previous))
        setattr(owner, name, wrapper)

    def _target(self, code, country, rule):
        def factory(original):
            sig = signature(original)

            @wraps(original)
            def observed(*args, **kwargs):
                self._check(code, country, rule)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                before = _freeze(_record(dict(bound.arguments)))
                epsilon_before = _freeze(getattr(rule, "_household_epsilon", None))
                rng_before = _freeze(rule._idiosyncratic_rng.bit_generator.state)
                try:
                    result = original(*args, **kwargs)
                except BaseException as error:
                    self._append(code, country, "target_error", inputs=before, error=type(error).__name__)
                    raise
                comp = getattr(rule, "last_target_consumption_components", None)
                epsilon = getattr(rule, "_household_epsilon", None)
                deterministic = None
                if comp is not None and "target_consumption_raw_log_target" in comp:
                    deterministic = comp["target_consumption_raw_log_target"] - comp["target_consumption_epsilon"]
                self._append(
                    code,
                    country,
                    "target",
                    inputs=before,
                    epsilon_before=epsilon_before,
                    epsilon_after=epsilon,
                    residual_rng_before=rng_before,
                    residual_rng_after=rule._idiosyncratic_rng.bit_generator.state,
                    components=comp,
                    goods_demand=result,
                    deterministic_log_target_derived=deterministic,
                )
                return result

            return observed

        return factory

    def _getter(self, code, key):
        def factory(original):
            @wraps(original)
            def observed(*args, **kwargs):
                result = original(*args, **kwargs)
                if code in self._funding:
                    self._funding[code].setdefault(key, []).append(_freeze(result))
                return result

            return observed

        return factory

    def _refresh(self, code, country, rule):
        def factory(original):
            sig = signature(original)

            @wraps(original)
            def observed(*args, **kwargs):
                self._check(code, country, rule)
                if code in self._funding:
                    raise RuntimeError("Nested funding refresh is unsupported.")
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                hh = country.households
                state = _freeze(
                    {
                        key: hh.ts.current(key)
                        for key in (
                            "target_consumption",
                            "price_paid_for_property",
                            "received_mortgages",
                            "property_sale_proceeds",
                            "liquid_financial_assets",
                        )
                    }
                )
                self._funding[code] = {}
                try:
                    result = original(*args, **kwargs)
                    self._append(
                        code,
                        country,
                        "funding_components",
                        inputs=dict(bound.arguments),
                        state_before=state,
                        native_getter_results=self._funding[code],
                    )
                    return result
                finally:
                    del self._funding[code]

            return observed

        return factory

    def _plan(self, code, country, rule, event):
        def factory(original):
            sig = signature(original)

            @wraps(original)
            def observed(*args, **kwargs):
                self._check(code, country, rule)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                before = _freeze(_record(dict(bound.arguments)))
                result = original(*args, **kwargs)
                extra = {}
                if event == "floor":
                    floor = np.asarray(bound.arguments["subsistence_floor"], dtype=float)
                    floor = np.where(np.isfinite(floor), np.maximum(floor, 0), 0)
                    extra["maximum_cut_derived"] = np.maximum(result.consumption_before_floor - floor, 0)
                    extra["floor_top_up_derived"] = np.maximum(
                        floor - (result.consumption_before_floor - result.consumption_cut_amount), 0
                    )
                self._append(code, country, event, inputs=before, output=result, **extra)
                return result

            return observed

        return factory

    def _credit(self, code, country, rule):
        def factory(original):
            @wraps(original)
            def observed(*args, **kwargs):
                self._check(code, country, rule)
                hh = country.households
                plan = _freeze(_record(hh.pre_grant_feasible_plan))
                result = original(*args, **kwargs)
                self._append(
                    code,
                    country,
                    "credit_request",
                    pre_grant_plan=plan,
                    actual_live_credit_requested=hh.ts.current("live_credit_requested"),
                    target_consumption_loans=hh.ts.current("target_consumption_loans"),
                )
                return result

            return observed

        return factory

    def _service(self, code, country, rule, name):
        def factory(original):
            @wraps(original)
            def observed(*args, **kwargs):
                self._check(code, country, rule)
                result = original(*args, **kwargs)
                self._append(code, country, "scheduled_service", source_method=name, values=result)
                return result

            return observed

        return factory

    def _payments(self, code, country, rule):
        def factory(original):
            @wraps(original)
            def observed(*args, **kwargs):
                self._check(code, country, rule)
                result = original(*args, **kwargs)
                hh = country.households
                names = (
                    "pre_relief_funding_gap",
                    "post_relief_funding_gap",
                    "actual_mortgage_payment",
                    "unpaid_mortgage_payment",
                    "scheduled_consumer_payment",
                    "actual_consumer_payment",
                    "unpaid_consumer_payment",
                    "stage5_subsistence_support",
                    "debt_installments",
                    "consumer_interest_paid",
                    "consumer_principal_paid",
                    "interest_paid",
                )
                self._append(
                    code,
                    country,
                    "payments_settled",
                    household={
                        key: {"value": hh.ts.current(key), "history_length": len(hh.ts.historic(key))}
                        for key in names
                        if key in hh.ts.dicts
                    },
                    unavailable=tuple(key for key in names if key not in hh.ts.dicts),
                )
                return result

            return observed

        return factory

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Residual capture is single-use.")
        if hasattr(self.model, "_residual_diagnostic_capture"):
            raise RuntimeError("Residual capture is already installed.")
        if (os.getpid(), threading.get_ident()) != self._owner:
            raise RuntimeError("Residual capture requires its original process/thread.")
        if not isinstance(self.model.prehooks, list):
            raise TypeError("Model prehooks must be a list.")
        self._entered = True
        self.model._residual_diagnostic_capture = self
        self.model.prehooks.append(self._hook)
        try:
            for code, country in self.model.countries.items():
                hh = country.households
                rule = hh.functions["consumption"]
                self._memberships[code] = self._membership(hh)
                self._append(
                    code,
                    country,
                    "startup",
                    household_ids=tuple(range(len(self._memberships[code]))),
                    household_memberships=self._memberships[code],
                    household={
                        key: {"value": hh.ts.current(key), "history_length": len(hh.ts.historic(key))}
                        for key in _START
                        if key in hh.ts.dicts
                    },
                    epsilon=getattr(rule, "_household_epsilon", None),
                    residual_rng=rule._idiosyncratic_rng.bit_generator.state,
                )
                self._patch(rule, "compute_target_consumption", self._target(code, country, rule))
                self._patch(hh, "refresh_post_labour_feasibility", self._refresh(code, country, rule))
                for name in (
                    "current_financing_income",
                    "current_consumption_cash_rent",
                    "target_investment_cash",
                    "consumption_cash_factor",
                ):
                    self._patch(hh, name, self._getter(code, name))
                self._patch(hh, "compute_target_credit", self._credit(code, country, rule))
                feasibility = hh.financial_feasibility
                self._patch(feasibility, "refresh_cash_needs", self._plan(code, country, rule, "cash_refresh"))
                self._patch(feasibility, "settle_consumption_floor", self._plan(code, country, rule, "floor"))
                for name in (
                    "compute_opening_scheduled_consumption_payments_by_household",
                    "compute_opening_scheduled_mortgage_payments_by_household",
                ):
                    self._patch(country.credit_market, name, self._service(code, country, rule, name))
                self._patch(country, "settle_authoritative_household_payments", self._payments(code, country, rule))
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self):
        for owner, name, previous in reversed(self._restores):
            if previous is _MISSING:
                vars(owner).pop(name, None)
            else:
                setattr(owner, name, previous)
        self._restores.clear()
        self.model.prehooks[:] = [hook for hook in self.model.prehooks if hook is not self._hook]
        if getattr(self.model, "_residual_diagnostic_capture", None) is self:
            del self.model._residual_diagnostic_capture

    def __exit__(self, *_exc):
        self._restore()
