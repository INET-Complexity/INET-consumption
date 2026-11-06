"""Read-only snapshots of live household target calls in cloned validation arms."""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from types import MappingProxyType

import numpy as np

_REQUIRED = (
    "target_consumption_income_base",
    "target_consumption_annual_income_base",
    "target_consumption_pi_original",
    "target_consumption_monetary_income_nominal",
    "target_consumption_permanent_income_log_ratio",
    "target_consumption_epsilon",
    "target_consumption_alpha_2",
    "target_consumption_gamma_1",
    "target_consumption_raw_log_target",
    "target_consumption_raw_target_real",
    "target_consumption_unguarded_delta_log_consumption",
    "target_consumption_delta_log_consumption",
    "target_consumption_partial_adjustment_gap",
    "target_consumption_interest_rate_cashflow",
    "target_consumption_uncertainty",
    "target_consumption_growth_clipped",
    "target_consumption_calibrated_total",
    "target_consumption_cash_rent",
    "target_consumption_imputed_rent",
    "target_consumption_real_budget",
)
_STATE = (
    "expected_income",
    "expected_behavioural_income",
    "expected_behavioural_income_raw",
    "expected_behavioural_income_floor_used",
    "expected_behavioural_income_component_floor_used",
    "income_employee",
    "income_social_transfers",
    "cacf_real_consumption_budget",
    "liquid_financial_assets",
    "illiquid_financial_assets",
    "wealth_main_residence",
    "consumption_loan_debt",
    "mortgage_debt",
    "rent",
    "rent_imputed",
    "credit_requested",
    "received_consumption_loans",
    "stage5_subsistence_support",
)
_MISSING = object()


def _freeze(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Diagnostic arrays must not contain mutable Python objects.")
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool, np.generic)):
        return value
    raise TypeError(f"Unsupported diagnostic value: {type(value).__name__}")


def _prepared_record(prepared):
    """Only the actual prepared operands/window; no producer/evaluator invocation."""
    names = (
        "period_id",
        "pass_kind",
        "household_layout_key",
        "income_base_real",
        "annual_income_base_real",
        "pi_log_ratio_rebased",
        "pi_log_denominator_adjustment",
        "log_permanent_income_proxy_real",
        "floor_hit_mask",
        "floor_hit_counts",
        "window_row_count",
        "current_deflator",
    )
    record = {name: getattr(prepared, name) for name in names}
    record["opening_real"] = vars(prepared.opening_real)
    record["opening_nominal"] = vars(prepared.opening_snapshot.nominal)
    record["pi_payload"] = {
        "monetary_income_nominal": prepared.pi_payload.monetary_income_nominal,
        "pi_log_ratio_original": prepared.pi_payload.pi_log_ratio_original,
    }
    history = prepared.histories
    n = prepared.window_row_count
    record["history_window"] = {
        "behavioural_income_nominal": history.behavioural_income_nominal_history[-n:],
        "subsistence_nominal": history.subsistence_nominal_history[-n:],
        "declared_consumer_price_history": history.consumer_price_history,
        "current_consumer_price": history.current_consumer_price,
        "initial_consumer_price": history.initial_consumer_price,
        "price_history_layout": history.price_history_layout,
        "months_per_period": history.months_per_period,
        "population_scale": history.income_units.population_scale,
        "currency": history.income_units.currency,
    }
    return record


@dataclass(frozen=True)
class TargetPassSnapshot:
    country_code: str
    period_id: int
    year: int
    month: int
    pass_kind: str
    rule_class: str
    inputs: Mapping
    household_before: Mapping
    components: Mapping
    unavailable_components: tuple[str, ...]
    goods_demand: np.ndarray


class TwoPassDiagnosticCapture:
    """Observe successful calls exactly once, restoring instance methods on exit.

    Enter only after cloning, and leave before checkpointing again. No learning,
    RNG, equation replay, time-series append, response calculation or serialization
    occurs here. Snapshots keep both passes even though native persisted diagnostics
    replace the first pass. Zero-growth bypasses explicitly omit stale rule caches.
    This observer supports serial execution of the bound arm only.
    """

    def __init__(self, model):
        self._model = model
        self._records = []
        self._restores = []
        self._entered = False

    @property
    def records(self):
        return tuple(self._records)

    def _wrap(self, code, country, original):
        call_signature = signature(original)

        @wraps(original)
        def observed(*args, **kwargs):
            arguments = call_signature.bind(*args, **kwargs)
            arguments.apply_defaults()
            values = dict(arguments.arguments)
            prepared = values.pop("consumption_inputs", None)
            values["consumption_inputs"] = None if prepared is None else _prepared_record(prepared)
            inputs = _freeze(values)
            hh = country.households
            state = _freeze(
                {
                    key: {
                        "current": hh.ts.current(key),
                        "previous": hh.ts.prev(key) if len(hh.ts.historic(key)) > 1 else None,
                    }
                    for key in _STATE
                    if key in hh.ts.dicts
                }
            )
            period = len(hh.ts.historic("income"))
            date = (self._model.timestep.year, self._model.timestep.month)
            rule = hh.functions["consumption"]
            result = original(*args, **kwargs)
            components = (
                None
                if values.get("assume_zero_growth", False)
                else getattr(rule, "last_target_consumption_components", None)
            )
            copied = _freeze({} if components is None else components)
            self._records.append(
                TargetPassSnapshot(
                    str(code),
                    period,
                    *date,
                    "replacement" if values.get("replace_current_diagnostics", False) else "first",
                    f"{type(rule).__module__}.{type(rule).__qualname__}",
                    inputs,
                    state,
                    copied,
                    tuple(key for key in _REQUIRED if key not in copied),
                    _freeze(result),
                )
            )
            return result

        return observed

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Diagnostic capture cannot be entered more than once.")
        self._entered = True
        try:
            for code, country in self._model.countries.items():
                hh = country.households
                previous = vars(hh).get("compute_target_consumption", _MISSING)
                wrapper = self._wrap(code, country, hh.compute_target_consumption)
                self._restores.append((hh, previous))
                hh.compute_target_consumption = wrapper
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self):
        for hh, previous in reversed(self._restores):
            if previous is _MISSING:
                del hh.compute_target_consumption
            else:
                hh.compute_target_consumption = previous
        self._restores.clear()

    def __exit__(self, *_exc):
        self._restore()
