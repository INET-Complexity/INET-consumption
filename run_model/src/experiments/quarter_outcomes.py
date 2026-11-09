"""Read-only, in-memory end-of-quarter records for cloned validation arms."""

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass

import numpy as np

from .paired_response import _ids
from .two_pass_diagnostics import _freeze

_OUTCOMES = ("income", "consumption", "consumption_cash_expenditure", "consumption_including_housing")
_HOUSEHOLD = (
    "liquid_financial_assets",
    "illiquid_financial_assets",
    "wealth_main_residence",
    "mortgage_debt",
    "consumption_loan_debt",
    "debt",
    "rent",
    "rent_imputed",
    "target_consumption",
    "cacf_real_consumption_budget",
    "received_consumption_loans",
    "received_mortgages",
    "live_credit_requested",
    "funded_from_liquid_assets",
    "stage5_subsistence_support",
    "income_social_transfers",
    "income_public_pension",
    "income_other_social_transfers",
    "income_unemployment_benefits",
    "income_employee",
    "cash_saving_before_financing",
    "stage5_cash_ledger_residual",
    "debt_installments",
    "interest_paid",
    "realised_household_expenditure",
    "portfolio_settlement_committed_lfa_flow",
    "portfolio_settlement_committed_ifa_flow",
)
_PLAN = (
    "credit_requested",
    "credit_granted",
    "funded_from_liquid_assets",
    "planned_liquidation_total",
    "reserved_liquidation_total",
    "liquidation_sale_cost",
    "settled_liquidation_total",
    "consumption_after_floor",
)
_BANK = (
    "deposits_from_households",
    "deposits_from_firms",
    "deposits",
    "equity",
    "liability",
    "total_outstanding_loans",
    "interest_rate_on_household_deposits",
)
_GOVERNMENT = (
    "debt",
    "deficit",
    "revenue",
    "total_unemployment_benefits",
    "total_public_pension_benefits",
    "total_other_social_transfers",
    "total_necessity_support",
)
_ECONOMY = ("gdp_output", "unemployment_rate", "total_growth")


@dataclass(frozen=True)
class QuarterOutcome:
    revision: str
    arm: str
    seed: int
    country_code: str
    iteration: int
    completed_row: int
    year: int
    month: int
    household_ids: tuple
    household_memberships: tuple
    price_series: str
    current_price: float
    initial_price: float
    deflator: float
    nominal_outcomes: Mapping
    household_state: Mapping
    funding_plan: Mapping
    banks: Mapping
    government: Mapping
    economy: Mapping
    external_funding: Mapping
    unavailable: Mapping
    source_rows: Mapping


class QuarterOutcomeCollector:
    """Append detached snapshots after native settlement, never alter model accounts.

    Enter after cloning and installing intervention hooks, so payer verification
    runs before this posthook. IDs/order come from the common registered cohort;
    index labels alone do not replace the frozen membership check. start_iteration
    identifies the next iteration, with initial row 0. No runner, storage, response
    computation, period resampling, fabricated financed spending or filtering.
    Optional fields retain raw nonfinite values with explicit availability reasons.
    Native consumption is nominal VAT-exclusive goods; cash/including-housing are
    native purchaser-price outcomes excluding investment. Supply them separately
    to measurement; never relabel an aggregate scalar as a household vector.
    """

    def __init__(self, model, *, revision, arm, seed, start_iteration, household_ids, interventions=None):
        if not all(isinstance(x, str) and x.strip() for x in (revision, arm)):
            raise ValueError("Declare nonempty revision and arm labels.")
        if any(
            isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in (seed, start_iteration)
        ):
            raise ValueError("Seed and start_iteration must be integers.")
        if start_iteration < 0 or not isinstance(household_ids, Mapping) or not household_ids:
            raise ValueError("Declare a nonnegative iteration and nonempty country/cohort mapping.")
        self._model, self._revision, self._arm, self._seed = model, revision, arm, int(seed)
        self._next = int(start_iteration)
        self._ids = {code: _ids(ids) for code, ids in household_ids.items()}
        self._interventions = {} if interventions is None else dict(interventions)
        if set(self._interventions) - set(self._ids):
            raise ValueError("Interventions must belong to collected countries.")
        self._records, self._memberships, self._owners = [], {}, {}
        self._entered = False
        self._hook = self._capture

    @property
    def records(self):
        return tuple(self._records)

    @staticmethod
    def _membership(country):
        return tuple(tuple(int(i) for i in row) for row in country.households.states["corr_individuals"])

    @staticmethod
    def _read(ts, keys, prefix, unavailable, rows):
        result = {}
        for key in keys:
            label = f"{prefix}.{key}"
            if key not in ts.dicts or not ts.historic(key):
                unavailable[label] = "missing"
                continue
            value = np.asarray(ts.current(key), dtype=float)
            result[key] = value
            rows[label] = len(ts.historic(key))
            if not np.all(np.isfinite(value)):
                unavailable[label] = "nonfinite"
        return result

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Outcome collector cannot be entered more than once.")
        for code, ids in self._ids.items():
            if code not in self._model.countries:
                raise ValueError("Collected country is missing.")
            country = self._model.countries[code]
            membership = self._membership(country)
            if len(membership) != len(ids) or len(country.households.ts.historic("income")) != self._next + 1:
                raise ValueError("Initial household cohort/income rows do not match the declared checkpoint.")
            self._memberships[code], self._owners[code] = membership, country
        self._model.posthooks.append(self._hook)
        self._entered = True
        return self

    def _capture(self, model, t, year, month):
        if model is not self._model or t != self._next or (year, month) != (model.timestep.year, model.timestep.month):
            raise ValueError("Outcome hook requires its original arm and next consecutive iteration.")
        snapshots = []
        for code, ids in self._ids.items():
            country = model.countries[code]
            if country is not self._owners[code] or self._membership(country) != self._memberships[code]:
                raise ValueError("Household cohort/order changed during outcome collection.")
            hh, economy = country.households, country.economy
            if len(hh.ts.historic("income")) != t + 2:
                raise ValueError("Collect only after one native income settlement per quarter.")
            missing, rows = {}, {}
            outcomes = self._read(hh.ts, _OUTCOMES, "outcome", missing, rows)
            for key, value in outcomes.items():
                if value.shape != (len(ids),) or not np.all(np.isfinite(value)) or rows[f"outcome.{key}"] != t + 2:
                    raise ValueError(f"Native outcome {key} is nonfinite, malformed or stale.")
            price = float(economy.current_consumer_price_level())
            initial = float(economy.initial_consumer_price_level())
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                d = price / initial if initial != 0 else np.nan
            if not np.isfinite(d) or not np.isfinite(price) or not np.isfinite(initial) or min(price, initial, d) <= 0:
                raise ValueError("Selected consumer-price deflator must be finite and positive.")
            price_series = economy.consumer_price_level_series_name()
            rows[f"economy.{price_series}"] = len(economy.ts.historic(price_series))
            if rows[f"economy.{price_series}"] != t + 2:
                raise ValueError("Selected consumer-price history is stale or misaligned.")
            household = self._read(hh.ts, _HOUSEHOLD, "household", missing, rows)
            household["corresponding_bank_id"] = np.asarray(hh.states["Corresponding Bank ID"])
            plan = getattr(hh, "post_grant_feasible_plan", None)
            funding = {}
            for key in _PLAN:
                value = None if plan is None else getattr(plan, key, None)
                funding[key] = value
                if value is None:
                    missing[f"funding_plan.{key}"] = "missing"
                elif not np.all(np.isfinite(np.asarray(value, dtype=float))):
                    missing[f"funding_plan.{key}"] = "nonfinite"
            banks = self._read(country.banks.ts, _BANK, "banks", missing, rows)
            government = self._read(country.central_government.ts, _GOVERNMENT, "government", missing, rows)
            macro = self._read(economy.ts, _ECONOMY, "economy", missing, rows)
            external = {"status": "not_applicable"}
            intervention = self._interventions.get(code)
            booking = None if intervention is None else intervention.booking
            if booking is not None:
                if not is_dataclass(booking):
                    raise ValueError("External funding booking must be a native experiment record.")
                booked_date = (booking.year, booking.month)
                if booked_date == (year, month):
                    external = {field.name: getattr(booking, field.name) for field in fields(booking)}
                    external.update(status="settled", booking_type=type(booking).__name__)
                elif booked_date < (year, month):
                    external = {"status": "no_new_booking"}
                else:
                    raise ValueError("Funding booking cannot be dated after the completed quarter.")
            elif intervention is not None:
                if (year, month) >= intervention.date:
                    raise ValueError("Install the outcome hook after intervention settlement verification.")
                external = {"status": "not_due"}
            snapshots.append(
                QuarterOutcome(
                    self._revision,
                    self._arm,
                    self._seed,
                    str(code),
                    int(t),
                    int(t + 1),
                    int(year),
                    int(month),
                    ids,
                    self._memberships[code],
                    price_series,
                    price,
                    initial,
                    d,
                    *(
                        _freeze(value)
                        for value in (outcomes, household, funding, banks, government, macro, external, missing, rows)
                    ),
                )
            )
        self._records.extend(snapshots)
        self._next += 1

    def __exit__(self, *_exc):
        self._model.posthooks[:] = [hook for hook in self._model.posthooks if hook is not self._hook]
