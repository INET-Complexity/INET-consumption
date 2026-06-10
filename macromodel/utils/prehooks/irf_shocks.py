"""Pre-hooks for paired impulse-response shock experiments."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import blake2b
from typing import Literal

import numpy as np

from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.simulation import Simulation

ShockKind = Literal[
    "government_consumption",
    "income_tax",
    "policy_rate",
    "unemployment_rate",
]


@dataclass(frozen=True)
class ShockSpec:
    """Description of an exogenous shock arm for paired IRF experiments."""

    name: str
    kind: ShockKind
    period: int
    magnitude: float
    duration: int = 1
    mode: Literal["additive", "multiplicative"] = "additive"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ShockSpec.name must be non-empty.")
        if self.period < 0:
            raise ValueError("ShockSpec.period must be non-negative.")
        if self.duration < 1:
            raise ValueError("ShockSpec.duration must be at least 1.")
        if self.mode not in {"additive", "multiplicative"}:
            raise ValueError("ShockSpec.mode must be 'additive' or 'multiplicative'.")


class _PolicyRateShockProxy:
    """Policy-rate function wrapper that injects a time-varying additive shock."""

    def __init__(self, wrapped) -> None:
        self.wrapped = wrapped
        self.shock = 0.0

    def compute_rate(self, *args, **kwargs):
        kwargs["shock"] = float(kwargs.get("shock", 0.0)) + float(self.shock)
        return self.wrapped.compute_rate(*args, **kwargs)


def _period_to_year_month(initial_year: int, time_unit: int, period: int) -> tuple[int, int]:
    if period < 0:
        raise ValueError("period must be non-negative.")
    month_index = 1 + int(period) * int(time_unit)
    year = int(initial_year) + (month_index - 1) // 12
    month = ((month_index - 1) % 12) + 1
    return year, month


def _year_month_to_period(initial_year: int, time_unit: int, year: int, month: int) -> int:
    month_index = (int(year) - int(initial_year)) * 12 + int(month)
    if month_index < 1:
        return -1
    return (month_index - 1) // int(time_unit)


def _active_period(spec: ShockSpec, initial_year: int, time_unit: int, year: int, month: int) -> bool:
    period = _year_month_to_period(initial_year, time_unit, year, month)
    return spec.period <= period < spec.period + spec.duration


def _shock_values(values: np.ndarray, *, spec: ShockSpec) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if spec.mode == "multiplicative":
        return values * (1.0 + float(spec.magnitude))
    return values + float(spec.magnitude)


def _stable_shock_seed(base_seed: int | None, spec: ShockSpec, period: int) -> int:
    seed_material = f"{base_seed or 0}:{spec.name}:{spec.kind}:{spec.period}:{spec.duration}:{period}".encode()
    digest = blake2b(seed_material, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32)


def _firm_employee_counts(corresponding_firm: np.ndarray, n_firms: int) -> np.ndarray:
    valid_firms = np.asarray(corresponding_firm, dtype=int)
    valid_firms = valid_firms[valid_firms >= 0]
    return np.bincount(valid_firms, minlength=int(n_firms))


def _remove_from_firm_employment(firm_employments: list, *, firm_id: int, individual_id: int) -> None:
    employment = firm_employments[int(firm_id)]
    try:
        employment.remove(int(individual_id))
        return
    except AttributeError:
        pass
    except ValueError:
        return
    firm_employments[int(firm_id)] = [worker for worker in employment if int(worker) != int(individual_id)]


def create_government_consumption_shock_hook(
    *,
    country_code: str,
    initial_year: int,
    time_unit: int,
    spec: ShockSpec,
) -> Callable[[Simulation, int, int], None]:
    """Create a prehook that shocks exogenous real government consumption."""

    if spec.kind != "government_consumption":
        raise ValueError("Government consumption hook requires kind='government_consumption'.")
    original_values: np.ndarray | None = None

    def government_consumption_shock_hook(simulation: Simulation, year: int, month: int) -> None:
        nonlocal original_values
        if country_code not in simulation.countries:
            raise ValueError(f"Government consumption shock cannot find country '{country_code}'.")
        country = simulation.countries[country_code]
        column = "Real Government Consumption (Value)"
        if column not in country.exogenous.national_accounts_during.columns:
            raise ValueError(f"Cannot shock government consumption: missing exogenous column {column!r}.")
        if original_values is None:
            original_values = country.exogenous.national_accounts_during[column].to_numpy(dtype=float).copy()
        country.exogenous.national_accounts_during.loc[:, column] = original_values
        if not _active_period(spec, initial_year, time_unit, year, month):
            return
        start = spec.period
        stop = min(spec.period + spec.duration, len(original_values))
        shocked = original_values.copy()
        shocked[start:stop] = _shock_values(shocked[start:stop], spec=spec)
        country.exogenous.national_accounts_during.loc[:, column] = shocked
        logging.info("Applied government consumption shock %s at %s-%s", spec.name, year, month)

    return government_consumption_shock_hook


def create_tax_rate_shock_hook(
    *,
    country_code: str,
    initial_year: int,
    time_unit: int,
    spec: ShockSpec,
    tax_state: str = "Income Tax",
) -> Callable[[Simulation, int, int], None]:
    """Create a prehook that temporarily shocks a central-government tax state."""

    if spec.kind != "income_tax":
        raise ValueError("Tax-rate hook requires kind='income_tax'.")
    baseline: float | np.ndarray | None = None

    def tax_rate_shock_hook(simulation: Simulation, year: int, month: int) -> None:
        nonlocal baseline
        if country_code not in simulation.countries:
            raise ValueError(f"Tax-rate shock cannot find country '{country_code}'.")
        states = simulation.countries[country_code].central_government.states
        if tax_state not in states:
            raise ValueError(f"Cannot shock tax state {tax_state!r}; available states: {sorted(states)}.")
        if baseline is None:
            current = states[tax_state]
            baseline = float(current) if np.isscalar(current) else np.asarray(current, dtype=float).copy()
        states[tax_state] = baseline.copy() if isinstance(baseline, np.ndarray) else float(baseline)
        if not _active_period(spec, initial_year, time_unit, year, month):
            return
        shocked = _shock_values(np.asarray(baseline, dtype=float), spec=spec)
        states[tax_state] = shocked if isinstance(baseline, np.ndarray) else float(shocked)
        logging.info("Applied tax-rate shock %s to %s at %s-%s", spec.name, tax_state, year, month)

    return tax_rate_shock_hook


def create_policy_rate_shock_hook(
    *,
    country_code: str,
    initial_year: int,
    time_unit: int,
    spec: ShockSpec,
) -> Callable[[Simulation, int, int], None]:
    """Create a prehook that adds an additive shock to the policy-rate rule."""

    if spec.kind != "policy_rate":
        raise ValueError("Policy-rate hook requires kind='policy_rate'.")
    if spec.mode != "additive":
        raise ValueError("Policy-rate shocks are additive rate-point shocks.")

    def policy_rate_shock_hook(simulation: Simulation, year: int, month: int) -> None:
        if country_code not in simulation.countries:
            raise ValueError(f"Policy-rate shock cannot find country '{country_code}'.")
        policy_functions = simulation.countries[country_code].central_bank.functions
        current = policy_functions["policy_rate"]
        if not isinstance(current, _PolicyRateShockProxy):
            current = _PolicyRateShockProxy(current)
            policy_functions["policy_rate"] = current
        current.shock = float(spec.magnitude) if _active_period(spec, initial_year, time_unit, year, month) else 0.0
        if current.shock:
            logging.info("Applied policy-rate shock %s at %s-%s: %s", spec.name, year, month, current.shock)

    return policy_rate_shock_hook


def create_unemployment_rate_shock_hook(
    *,
    country_code: str,
    initial_year: int,
    time_unit: int,
    spec: ShockSpec,
) -> Callable[[Simulation, int, int], None]:
    """Create a prehook that separates sampled employed workers.

    For ``mode='additive'``, ``magnitude`` is an unemployment-rate point shock:
    ``0.01`` attempts to separate about 1 percent of the labour force. For
    ``mode='multiplicative'``, ``magnitude`` is the share of currently employed
    workers to separate. The normal labour market can rehire separated workers,
    so the realised unemployment-rate IRF remains endogenous.
    """

    if spec.kind != "unemployment_rate":
        raise ValueError("Unemployment-rate hook requires kind='unemployment_rate'.")
    if spec.magnitude <= 0.0:
        raise ValueError("Unemployment-rate shock magnitude must be strictly positive.")
    applied_periods: set[int] = set()

    def unemployment_rate_shock_hook(simulation: Simulation, year: int, month: int) -> None:
        if country_code not in simulation.countries:
            raise ValueError(f"Unemployment-rate shock cannot find country '{country_code}'.")
        period = _year_month_to_period(initial_year, time_unit, year, month)
        if period in applied_periods or not _active_period(spec, initial_year, time_unit, year, month):
            return
        country = simulation.countries[country_code]
        activity = country.individuals.states["Activity Status"]
        corresponding_firm = country.individuals.states["Corresponding Firm ID"]
        firm_employments = country.firms.states["Employments"]
        n_firms = int(country.firms.ts.current("n_firms")[0])
        firm_counts = _firm_employee_counts(corresponding_firm, n_firms)
        employed = activity == ActivityStatus.EMPLOYED
        unemployed = activity == ActivityStatus.UNEMPLOYED
        labour_force = int(np.sum(employed | unemployed))
        employed_ids = np.flatnonzero(employed & (corresponding_firm >= 0) & (firm_counts[corresponding_firm] > 1))
        if labour_force == 0 or employed_ids.size == 0:
            applied_periods.add(period)
            return

        if spec.mode == "multiplicative":
            n_to_separate = int(np.ceil(float(spec.magnitude) * float(np.sum(employed))))
        else:
            n_to_separate = int(np.ceil(float(spec.magnitude) * float(labour_force)))
        n_to_separate = max(0, min(n_to_separate, int(employed_ids.size)))
        if n_to_separate == 0:
            applied_periods.add(period)
            return

        rng = np.random.default_rng(_stable_shock_seed(simulation.random_seed, spec, period))
        separated_ids = rng.choice(employed_ids, size=n_to_separate, replace=False)
        for individual_id in separated_ids:
            firm_id = int(corresponding_firm[int(individual_id)])
            activity[int(individual_id)] = ActivityStatus.UNEMPLOYED
            corresponding_firm[int(individual_id)] = -1
            _remove_from_firm_employment(firm_employments, firm_id=firm_id, individual_id=int(individual_id))

        country.individuals.states["Started New Job"][separated_ids] = False
        country.individuals.states["Offered Wage of Accepted Job"][separated_ids] = 0.0
        applied_periods.add(period)
        logging.info(
            "Applied unemployment separation shock %s at %s-%s: separated=%s",
            spec.name,
            year,
            month,
            n_to_separate,
        )

    return unemployment_rate_shock_hook


def create_irf_shock_hook(
    *,
    country_code: str,
    initial_year: int,
    time_unit: int,
    spec: ShockSpec,
) -> Callable[[Simulation, int, int], None]:
    """Build a prehook for an IRF shock specification."""

    if spec.kind == "government_consumption":
        return create_government_consumption_shock_hook(
            country_code=country_code,
            initial_year=initial_year,
            time_unit=time_unit,
            spec=spec,
        )
    if spec.kind == "income_tax":
        return create_tax_rate_shock_hook(
            country_code=country_code,
            initial_year=initial_year,
            time_unit=time_unit,
            spec=spec,
        )
    if spec.kind == "policy_rate":
        return create_policy_rate_shock_hook(
            country_code=country_code,
            initial_year=initial_year,
            time_unit=time_unit,
            spec=spec,
        )
    if spec.kind == "unemployment_rate":
        return create_unemployment_rate_shock_hook(
            country_code=country_code,
            initial_year=initial_year,
            time_unit=time_unit,
            spec=spec,
        )
    raise ValueError(f"Unsupported IRF shock kind: {spec.kind!r}.")
