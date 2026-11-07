"""Opening LFA endowments for paired validation, separate from current income."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from macromodel.agents.banks.banks import Banks
from macromodel.timeseries import TimeSeries
from macromodel.util.get_histogram import get_histogram

from .income_transfer import OneTimeIncomeTransfer, _immutable


@dataclass(frozen=True)
class EndowmentSnapshot:
    """Sidecar evidence: native current rows become the intervened opening state."""

    income_period: int
    household_before: tuple
    household_after: tuple
    banks_before: tuple
    banks_after: tuple

    def __post_init__(self):
        for field in ("household_before", "household_after", "banks_before", "banks_after"):
            object.__setattr__(self, field, tuple((key, _immutable(value)) for key, value in getattr(self, field)))


@dataclass(frozen=True)
class EndowmentBooking:
    year: int
    month: int
    household_endowments_nominal: np.ndarray
    bank_funding_nominal: np.ndarray
    payer_outflow_nominal: float

    def __post_init__(self):
        for field in ("household_endowments_nominal", "bank_funding_nominal"):
            object.__setattr__(self, field, _immutable(getattr(self, field)))


class OneTimeLFAEndowment(OneTimeIncomeTransfer):
    """Reuse date/ownership guards; override flow staging and settlement completely.

    Install on a fresh checkpoint clone after unshocked initialization. Native
    current rows are replaced, never mutated/extended. Keep opening_snapshot as
    mandatory pre-shock evidence when saving: native HDF5's current stock row now
    represents the post-intervention opening, not the original checkpoint row.
    Income, debts, priors, residuals, old history and initial_states are untouched.
    """

    def __init__(self, country_code, *, year, month, amounts_nominal):
        super().__init__(country_code, year=year, month=month, amounts_nominal=amounts_nominal)
        self.opening_snapshot = None

    @staticmethod
    def _bank_values(country, lfa):
        banks = country.banks
        # Run the native append-based aggregator only on a disposable scratch TS.
        view = SimpleNamespace(ts=TimeSeries(**{k: v[-1] for k, v in banks.ts.dicts.items()}))
        Banks.update_deposits(
            view,
            current_firm_deposits=country.firms.ts.current("deposits"),
            current_household_deposits=lfa,
            firm_corresponding_bank=country.firms.states["Corresponding Bank ID"],
            households_corresponding_bank=country.households.states["Corresponding Bank ID"],
        )
        view.ts.override_current("deposits", Banks.compute_deposits(view))
        view.ts.override_current("liability", Banks.compute_liability(view))
        return {
            k: view.ts.current(k)
            for k in (
                "deposits_from_households",
                "total_deposits_from_households",
                "deposits_from_firms",
                "total_deposits_from_firms",
                "deposits",
                "liability",
            )
        }

    @staticmethod
    def _assert_accounts(ts, expected):
        for key, value in expected.items():
            actual = np.asarray(ts.current(key), dtype=float)
            value = np.asarray(value, dtype=float)
            if actual.shape != value.shape or not np.all(np.isfinite(actual)) or not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid opening account {key}.")
            if not np.allclose(actual, value, rtol=1e-10, atol=1e-7):
                raise ValueError(f"Inconsistent opening account {key}; refusing to repair baseline accounts.")

    def stage(self, model, year, month):
        country = self._country_for(model)
        if self._period is not None or (year, month) < self.date:
            return
        if (year, month) != self.date:
            raise RuntimeError("Endowment date was missed; refusing a late intervention.")
        hh, banks = country.households, country.banks
        period = len(hh.ts.historic("income"))
        if period < 2:
            raise ValueError("Endowment requires completed unshocked initialization before intervention.")
        if country._staged_household_income_shock is not None:
            raise ValueError("LFA endowment cannot coexist with a staged income receipt.")
        rule = hh.functions["consumption"]
        if getattr(rule, "idiosyncratic_persistence", None) == "initial_state" and rule._household_epsilon is None:
            raise ValueError("Initialize consumption residuals before the endowment.")
        adapter = getattr(country, "_consumption_input_adapter", None)
        snapshot = None if adapter is None else adapter.opening_snapshot
        if snapshot is not None and snapshot.period_id >= period:
            raise ValueError("Endowment must precede this period's opening-stock capture.")
        lfa = np.asarray(hh.ts.current("liquid_financial_assets"), dtype=float)
        before_hh = {
            "liquid_financial_assets": lfa,
            "total_liquid_financial_assets": np.array([lfa.sum()]),
            "wealth_financial_assets": lfa + hh.ts.current("illiquid_financial_assets"),
        }
        before_hh["wealth"] = before_hh["wealth_financial_assets"] + hh.ts.current("wealth_real_assets")
        before_hh["net_wealth"] = before_hh["wealth"] - hh.ts.current("debt")
        self._assert_accounts(hh.ts, before_hh)
        before_bank = self._bank_values(country, lfa)
        # Closing liability was evaluated before the closing deposit row. Keep
        # that native timing offset: only the endowment's current liability
        # contribution changes, not the baseline's lagged balancing position.
        before_bank["liability"] = np.asarray(banks.ts.current("liability"), dtype=float)
        self._assert_accounts(banks.ts, before_bank)
        with np.errstate(over="ignore", invalid="ignore"):
            new_lfa = lfa + self.amounts_nominal
            after_hh = {
                key: value + self.amounts_nominal
                for key, value in before_hh.items()
                if key != "total_liquid_financial_assets"
            }
            after_hh["total_liquid_financial_assets"] = np.array([new_lfa.sum()])
            after_bank = self._bank_values(country, new_lfa)
            # The native formula's deposit-position term must stay on the same
            # pre-intervention basis on both sides of this finite difference.
            liability_delta = (
                np.maximum(0, after_bank["deposits_from_households"])
                - np.maximum(0, before_bank["deposits_from_households"])
                + np.maximum(0, after_bank["deposits_from_firms"])
                - np.maximum(0, before_bank["deposits_from_firms"])
            )
            after_bank["liability"] = before_bank["liability"] + liability_delta
        if any(not np.all(np.isfinite(v)) for v in (*after_hh.values(), *after_bank.values())):
            raise ValueError("Endowment would produce nonfinite accounts.")
        # Preserve existing histogram bin counts and scale, if diagnostics exist.
        for ts, before, after, keys in (
            (hh.ts, before_hh, after_hh, ("wealth",)),
            (banks.ts, before_bank, after_bank, ("deposits", "liability")),
        ):
            for key in keys:
                name = key + "_histogram"
                if name in ts.dicts:
                    old = np.asarray(ts.current(name), dtype=float)
                    before[name] = old
                    after[name] = get_histogram(after[key], country.scale, bins=old.shape[1] - 1)
        evidence = EndowmentSnapshot(
            period,
            tuple(before_hh.items()),
            tuple(after_hh.items()),
            tuple(before_bank.items()),
            tuple(after_bank.items()),
        )
        # All arithmetic and schema checks precede writes; the zero arm is exact.
        if np.any(self.amounts_nominal):
            for ts, updates in ((hh.ts, after_hh), (banks.ts, after_bank)):
                for key, value in updates.items():
                    ts.override_current(key, np.asarray(value, dtype=float).copy())
        self.opening_snapshot = evidence
        self._period = period

    def verify_settlement(self, model, _t, year, month):
        country = self._country_for(model)
        if self.booking is not None:
            return
        if self._period is None:
            if (year, month) >= self.date:
                raise RuntimeError("Endowment settlement observed without successful staging.")
            return
        if (year, month) != self.date:
            raise RuntimeError("Endowment was not reconciled in its target quarter.")
        hh, banks = country.households, country.banks
        if len(hh.ts.historic("income")) != self._period + 1 or country._staged_household_income_shock is not None:
            raise RuntimeError("Endowment requires one completed period without a staged income receipt.")
        income, expected = np.asarray(hh.ts.current("income")), np.asarray(hh.compute_income())
        if (
            income.shape != expected.shape
            or not np.all(np.isfinite(expected))
            or not np.allclose(income, expected, rtol=1e-10, atol=1e-7)
        ):
            raise RuntimeError("LFA endowment must not be recorded as current income.")
        # Native closing liability uses the previous bank-deposit row; preserve
        # that policy here and validate current aggregation/balancing separately.
        closing = self._bank_values(country, hh.ts.current("liquid_financial_assets"))
        closing.pop("liability")
        self._assert_accounts(banks.ts, closing)
        allocated = np.bincount(
            np.asarray(self._layout_key[1], dtype=int),
            weights=self.amounts_nominal,
            minlength=banks.ts.current("n_banks"),
        )
        self.booking = EndowmentBooking(year, month, self.amounts_nominal, allocated, float(self.amounts_nominal.sum()))
