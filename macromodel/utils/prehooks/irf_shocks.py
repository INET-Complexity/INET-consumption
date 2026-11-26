"""Pre-hooks for paired impulse-response shock experiments."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import blake2b
from typing import Literal

import numpy as np

from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.simulation import Simulation
from macromodel.utils.prehooks.household_income_shock import create_household_income_shock_hook

ShockKind = Literal[
    "government_consumption",
    "income_tax",
    "policy_rate",
    "unemployment_rate",
    "common_income_uncertainty",
    "individual_income_uncertainty",
    "total_income_uncertainty",
    "household_transfer",
]
_SAFE_SHOCK_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class ShockSpec:
    """Description of an exogenous shock arm for paired IRF experiments.

    For income-uncertainty shocks, ``magnitude`` is the expected proportional
    change in the relevant variance component, and ``random_log_sd`` is the
    standard deviation of its lognormal multiplier. Those kinds require
    ``mode="multiplicative"``. Common shocks draw one multiplier shared by all
    households; individual shocks draw one multiplier per household. Total
    shocks scale the whole uncertainty index theta (individual plus common
    component) by one factor; ``random_log_sd=None`` makes it deterministic.

    For ``household_transfer``, ``magnitude`` is the equal nominal payment per
    household as a fraction of median positive household income, paid once by
    the central government through the MPC experiment's staged-income channel.
    """

    name: str
    kind: ShockKind
    period: int
    magnitude: float
    duration: int = 1
    mode: Literal["additive", "multiplicative"] = "additive"
    random_log_sd: float | None = None
    prevent_rehiring: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.prevent_rehiring, bool):
            raise ValueError("prevent_rehiring must be a boolean.")
        if self.prevent_rehiring and self.kind != "unemployment_rate":
            raise ValueError("prevent_rehiring is only supported for unemployment_rate shocks.")
        try:
            magnitude = np.asarray(self.magnitude, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("ShockSpec.magnitude must be a finite scalar.") from exc
        if magnitude.ndim != 0 or not np.isfinite(magnitude.item()):
            raise ValueError("ShockSpec.magnitude must be a finite scalar.")
        if not self.name:
            raise ValueError("ShockSpec.name must be non-empty.")
        if not _SAFE_SHOCK_NAME.fullmatch(self.name):
            raise ValueError(
                "ShockSpec.name may only contain letters, digits, underscores, hyphens, and dots, "
                "and must start with a letter or digit."
            )
        if self.period < 0:
            raise ValueError("ShockSpec.period must be non-negative.")
        if self.duration < 1:
            raise ValueError("ShockSpec.duration must be at least 1.")
        if self.mode not in {"additive", "multiplicative"}:
            raise ValueError("ShockSpec.mode must be 'additive' or 'multiplicative'.")


class _GovernmentConsumptionShockProxy:
    """Government-consumption setter wrapper that shocks desired consumption targets."""

    def __init__(self, wrapped, spec: ShockSpec) -> None:
        self.wrapped = wrapped
        self.spec = spec
        self.active = False

    def compute_target_consumption(
        self,
        *,
        previous_desired_government_consumption,
        model,
        historic_total_consumption,
        initial_good_prices,
        current_good_prices,
        expected_growth,
        expected_inflation,
        current_time,
        exogenous_total_consumption,
        forecasting_window,
        assume_zero_noise=False,
        historic_real_gdp_growth=None,
        historic_productivity_growth=None,
        posted_price_index_by_industry=None,
    ):
        target = self.wrapped.compute_target_consumption(
            previous_desired_government_consumption=previous_desired_government_consumption,
            model=model,
            historic_total_consumption=historic_total_consumption,
            initial_good_prices=initial_good_prices,
            current_good_prices=current_good_prices,
            expected_growth=expected_growth,
            expected_inflation=expected_inflation,
            current_time=current_time,
            exogenous_total_consumption=exogenous_total_consumption,
            forecasting_window=forecasting_window,
            assume_zero_noise=assume_zero_noise,
            historic_real_gdp_growth=historic_real_gdp_growth,
            historic_productivity_growth=historic_productivity_growth,
            posted_price_index_by_industry=posted_price_index_by_industry,
        )
        if not self.active:
            return target
        target = np.asarray(target, dtype=float)
        if self.spec.mode == "multiplicative":
            return np.maximum(0.0, target * (1.0 + float(self.spec.magnitude)))

        magnitude = float(self.spec.magnitude)
        if magnitude == 0.0:
            return target
        delta = abs(magnitude) * _nonnegative_weights(target)
        return np.maximum(0.0, target + np.sign(magnitude) * delta)


class _PolicyRateShockProxy:
    """Policy-rate function wrapper that injects a period-rate additive shock."""

    def __init__(self, wrapped) -> None:
        self.wrapped = wrapped
        self.shock = 0.0

    def compute_rate(self, *args, **kwargs):
        base_rate = self.wrapped.compute_rate(*args, **kwargs)
        shocked_rate = base_rate + float(self.shock)
        lower_bound = getattr(self.wrapped, "lower_bound", None)
        if lower_bound is not None:
            # Policy floors are annual; both the stored rate and this shock
            # are per-period. Preserve the additive rate-path intervention.
            time_unit = kwargs.get("time_unit", args[6] if len(args) > 6 else 1)
            shocked_rate = max(shocked_rate, lower_bound / _periods_per_year(time_unit))
        return shocked_rate


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


def _periods_per_year(time_unit: int) -> int:
    if time_unit <= 0 or 12 % time_unit != 0:
        raise ValueError("time_unit must be a positive divisor of 12.")
    return 12 // int(time_unit)


def _shock_values(values: np.ndarray, *, spec: ShockSpec) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if spec.mode == "multiplicative":
        return values * (1.0 + float(spec.magnitude))
    return values + float(spec.magnitude)


def _nonnegative_weights(values: np.ndarray) -> np.ndarray:
    weights = np.asarray(values, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    weights_sum = weights.sum()
    if weights_sum <= 0.0:
        return np.full(weights.shape, 1.0 / len(weights))
    return weights / weights_sum


def _stable_shock_seed(base_seed: int | None, spec: ShockSpec, period: int) -> int:
    seed_material = f"{base_seed or 0}:{spec.name}:{spec.kind}:{spec.period}:{spec.duration}:{period}".encode()
    digest = blake2b(seed_material, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32)


def _firm_employee_counts(corresponding_firm: np.ndarray, n_firms: int) -> np.ndarray:
    valid_firms = np.asarray(corresponding_firm, dtype=int)
    valid_firms = valid_firms[valid_firms >= 0]
    return np.bincount(valid_firms, minlength=int(n_firms))


def _as_scalar_int(value) -> int:
    values = np.asarray(value)
    return int(values.reshape(-1)[0]) if values.ndim else int(values)


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
    """Create a prehook that shocks model government desired consumption."""

    if spec.kind != "government_consumption":
        raise ValueError("Government consumption hook requires kind='government_consumption'.")

    def government_consumption_shock_hook(simulation: Simulation, year: int, month: int) -> None:
        if country_code not in simulation.countries:
            raise ValueError(f"Government consumption shock cannot find country '{country_code}'.")
        country = simulation.countries[country_code]
        consumption_functions = country.government_entities.functions
        current = consumption_functions["consumption"]
        if not isinstance(current, _GovernmentConsumptionShockProxy):
            current = _GovernmentConsumptionShockProxy(current, spec)
            consumption_functions["consumption"] = current
        current.active = _active_period(spec, initial_year, time_unit, year, month)
        if current.active:
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
        if not np.all(np.isfinite(shocked)) or np.any((shocked < 0.0) | (shocked > 1.0)):
            raise ValueError(f"Income-tax shock {spec.name!r} would set tax rates outside [0, 1].")
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
    """Create a prehook that adds an annualized rate-point shock to policy rates."""

    if spec.kind != "policy_rate":
        raise ValueError("Policy-rate hook requires kind='policy_rate'.")
    if spec.mode != "additive":
        raise ValueError("Policy-rate shocks are additive annual rate-point shocks.")
    period_shock = float(spec.magnitude) / float(_periods_per_year(time_unit))

    def policy_rate_shock_hook(simulation: Simulation, year: int, month: int) -> None:
        if country_code not in simulation.countries:
            raise ValueError(f"Policy-rate shock cannot find country '{country_code}'.")
        policy_functions = simulation.countries[country_code].central_bank.functions
        current = policy_functions["policy_rate"]
        if not isinstance(current, _PolicyRateShockProxy):
            current = _PolicyRateShockProxy(current)
            policy_functions["policy_rate"] = current
        current.shock = period_shock if _active_period(spec, initial_year, time_unit, year, month) else 0.0
        if current.shock:
            logging.info(
                "Applied policy-rate shock %s at %s-%s: annual=%s period=%s",
                spec.name,
                year,
                month,
                spec.magnitude,
                current.shock,
            )

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
    so the realised unemployment-rate IRF remains endogenous. With
    ``prevent_rehiring=True``, directly separated workers remain ineligible for
    hiring until the end of the common shock window. Other workers can be hired.
    Their unemployment status and benefit eligibility are unchanged.
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
        country = simulation.countries[country_code]
        if spec.prevent_rehiring:
            states = country.individuals.states
            deadlines = states.setdefault(
                "IRF Rehiring Blocked Until", np.zeros(len(states["Activity Status"]), dtype=int)
            )
            states["IRF Rehiring Blocked"] = deadlines > period
        if period in applied_periods or not _active_period(spec, initial_year, time_unit, year, month):
            return
        country = simulation.countries[country_code]
        activity = country.individuals.states["Activity Status"]
        corresponding_firm = country.individuals.states["Corresponding Firm ID"]
        firm_employments = country.firms.states["Employments"]
        n_firms = _as_scalar_int(country.firms.ts.current("n_firms"))
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
        if spec.prevent_rehiring:
            deadlines[separated_ids] = np.maximum(deadlines[separated_ids], spec.period + spec.duration)
            states["IRF Rehiring Blocked"] = deadlines > period
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
    if spec.kind == "household_transfer":
        return create_household_transfer_shock_hook(
            country_code=country_code,
            initial_year=initial_year,
            time_unit=time_unit,
            spec=spec,
        )
    if spec.kind in {"common_income_uncertainty", "individual_income_uncertainty", "total_income_uncertainty"}:
        return create_income_uncertainty_shock_hook(
            country_code=country_code,
            initial_year=initial_year,
            time_unit=time_unit,
            spec=spec,
        )
    raise ValueError(f"Unsupported IRF shock kind: {spec.kind!r}.")


def create_income_uncertainty_shock_hook(
    *,
    country_code: str,
    initial_year: int,
    time_unit: int,
    spec: ShockSpec,
) -> Callable[[Simulation, int, int], None]:
    """Create a reproducible common or household-specific variance shock.

    The lognormal multipliers keep variance components positive. Their
    expected value is ``1 + magnitude``; ``random_log_sd`` sets dispersion
    around that mean. Draws are keyed by simulation seed, shock specification
    and economic period, so repeated planning calls in one period see the same
    values and separate IRF arms do not consume the model's shared random
    stream.
    """
    if spec.kind not in {"common_income_uncertainty", "individual_income_uncertainty", "total_income_uncertainty"}:
        raise ValueError("Income-uncertainty hook requires a common, individual or total uncertainty kind.")
    if not np.isfinite(spec.magnitude) or spec.magnitude <= -1.0:
        raise ValueError("Income-uncertainty shock magnitude must be finite and greater than -1.")
    deterministic = spec.kind == "total_income_uncertainty" and spec.random_log_sd is None
    if not deterministic and (
        spec.random_log_sd is None or not np.isfinite(spec.random_log_sd) or spec.random_log_sd <= 0.0
    ):
        raise ValueError("Income-uncertainty shocks require a finite, positive random_log_sd.")
    if spec.mode != "multiplicative":
        raise ValueError("Income-uncertainty shocks require mode='multiplicative'.")
    if spec.period < 1:
        raise ValueError("Income-uncertainty shocks must start after the first initialization period.")

    def income_uncertainty_shock_hook(simulation: Simulation, year: int, month: int) -> None:
        if country_code not in simulation.countries:
            raise ValueError(f"Income-uncertainty shock cannot find country '{country_code}'.")
        model_period = _year_month_to_period(initial_year, time_unit, year, month)
        # Simulation periods are zero-based; the household mapper keys periods
        # by income-history length, which starts at one for the first run step.
        household_period = model_period + 1
        country = simulation.countries[country_code]
        n_households = _as_scalar_int(country.households.ts.current("n_households"))
        common_factor = 1.0
        individual_factors = np.ones(n_households, dtype=float)

        if _active_period(spec, initial_year, time_unit, year, month):
            if deterministic:
                factor = 1.0 + float(spec.magnitude)
            else:
                rng = np.random.default_rng(_stable_shock_seed(simulation.random_seed, spec, model_period))
                log_sd = float(spec.random_log_sd)
                log_mean = np.log1p(float(spec.magnitude)) - 0.5 * log_sd**2
            if spec.kind == "total_income_uncertainty":
                # theta = individual * f_i + common * f_c: equal factors scale theta itself.
                if not deterministic:
                    factor = float(rng.lognormal(mean=log_mean, sigma=log_sd))
                common_factor = factor
                individual_factors = np.full(n_households, factor, dtype=float)
            elif spec.kind == "common_income_uncertainty":
                common_factor = float(rng.lognormal(mean=log_mean, sigma=log_sd))
            else:
                individual_factors = rng.lognormal(
                    mean=log_mean,
                    sigma=log_sd,
                    size=n_households,
                )

        # Store one period's draw for the household learning-input mapper.
        # Reassigning each period also clears the pulse after its duration.
        country.households.states["income_uncertainty_shock_factors"] = {
            "period": household_period,
            "common": common_factor,
            "individual": individual_factors,
        }

    return income_uncertainty_shock_hook


def create_household_transfer_shock_hook(
    *,
    country_code: str,
    initial_year: int,
    time_unit: int,
    spec: ShockSpec,
    exempt_from_support_test: bool = True,
) -> Callable[[Simulation, int, int], None]:
    """Create a one-off government transfer to every household, as in the MPC experiment.

    The equal nominal payment is ``magnitude`` times median positive household
    income in the shock period. It is staged through
    ``Country.stage_household_income_shock`` (same channel and support-test
    treatment as the MPC experiment), and the central government books its
    total in ``total_one_off_transfers``, which enters the deficit.
    """
    if spec.kind != "household_transfer":
        raise ValueError("Household-transfer hook requires kind='household_transfer'.")
    if spec.duration != 1:
        raise ValueError("Household-transfer shocks are one-off: duration must be 1.")
    if spec.mode != "multiplicative" or spec.magnitude <= 0.0:
        raise ValueError(
            "Household-transfer shocks need mode='multiplicative' and a positive fraction of median income."
        )
    target_year, target_month = _period_to_year_month(initial_year, time_unit, spec.period)
    return create_household_income_shock_hook(
        country_code=country_code,
        target_year=target_year,
        target_month=target_month,
        shock_fraction_of_median_income=float(spec.magnitude),
        exempt_from_support_test=exempt_from_support_test,
    )
