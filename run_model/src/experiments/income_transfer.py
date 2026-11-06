"""One-time externally funded receipts for the paired validation harness."""

from dataclasses import dataclass

import numpy as np


def _immutable(values):
    values = np.asarray(values, dtype=float)
    return np.frombuffer(values.tobytes(), dtype=float).reshape(values.shape)


@dataclass(frozen=True)
class TransferBooking:
    """Gross funding, not the endogenous change in income/deposits after support."""

    year: int
    month: int
    income_period: int
    household_receipts_nominal: np.ndarray
    bank_funding_nominal: np.ndarray
    payer_outflow_nominal: float

    def __post_init__(self):
        for name in ("household_receipts_nominal", "bank_funding_nominal"):
            object.__setattr__(self, name, _immutable(getattr(self, name)))


class OneTimeIncomeTransfer:
    """Install on one cloned arm, before its target-date first planning pass.

    Amounts are a frozen, already population-scaled nominal H-vector supplied by
    registration. Zero is allowed for a control. The helper neither chooses shock
    sizes nor edits stocks/fiscal accounts. Native income and wealth settlement
    perform the cash booking. Posthook validation records the external payer once.
    Capture checkpoints before installing these model-bound hooks.
    """

    def __init__(self, country_code, *, year, month, amounts_nominal):
        if not isinstance(country_code, str) or not country_code:
            raise ValueError("country_code must be nonempty.")
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in (year, month)):
            raise ValueError("Transfer date must use integer year/month.")
        if year < 1 or not 1 <= month <= 12:
            raise ValueError("Transfer date must have positive year and month 1..12.")
        amounts = np.asarray(amounts_nominal, dtype=float)
        if amounts.ndim != 1 or amounts.size == 0 or not np.all(np.isfinite(amounts)) or np.any(amounts < 0):
            raise ValueError("Transfer amounts must be a finite nonnegative nonempty H-vector.")
        with np.errstate(over="ignore"):
            total = amounts.sum()
        if not np.isfinite(total):
            raise ValueError("Aggregate transfer must be finite.")
        self.country_code = country_code
        self.date = (int(year), int(month))
        self.amounts_nominal = _immutable(amounts)
        self.booking = None
        self._model = None
        self._period = None

    @staticmethod
    def _layout(country):
        return (
            tuple(tuple(int(i) for i in members) for members in country.households.states["corr_individuals"]),
            tuple(np.asarray(country.households.states["Corresponding Bank ID"], dtype=int)),
        )

    def install(self, model):
        """Attach fresh pre/post hooks; never share a transfer instance across arms."""
        if self._model is not None:
            raise ValueError("Transfer is already installed; create a fresh instance per arm.")
        if self.country_code not in model.countries:
            raise ValueError("Transfer country is missing from the simulation.")
        country = model.countries[self.country_code]
        if np.asarray(country.households.ts.current("income")).shape != self.amounts_nominal.shape:
            raise ValueError("Transfer amounts must match household dimensions.")
        layout = self._layout(country)
        banks = country.banks.ts.current("n_banks")
        ids = np.asarray(layout[1])
        if len(layout[0]) != self.amounts_nominal.size or ids.shape != self.amounts_nominal.shape:
            raise ValueError("Transfer layout must identify each household.")
        if np.any(ids < 0) or np.any(ids >= banks):
            raise ValueError("Transfer bank assignments are invalid.")
        self._layout_key = layout
        self._country = country
        self._model = model
        model.prehooks.append(self.stage)
        model.posthooks.append(self.verify_settlement)

    def _country_for(self, model):
        if model is not self._model or model.countries.get(self.country_code) is not self._country:
            raise ValueError("Transfer hooks are bound to their original arm/country.")
        if self._layout(self._country) != self._layout_key:
            raise ValueError("Transfer household ordering/bank layout changed.")
        return self._country

    def stage(self, model, year, month):
        country = self._country_for(model)
        if self._period is not None:
            return
        if (year, month) < self.date:
            return
        if (year, month) != self.date:
            raise RuntimeError("Transfer date was missed; refusing a late intervention.")
        period = len(country.households.ts.historic("income"))
        # Native staging rejects an already pending intervention before any debit.
        country.stage_household_income_shock(self.amounts_nominal)
        self._period = period

    def verify_settlement(self, model, _t, year, month):
        country = self._country_for(model)
        if self.booking is not None:
            return
        if self._period is None:
            if (year, month) >= self.date:
                raise RuntimeError("Transfer settlement observed without successful staging.")
            return
        if (year, month) != self.date:
            raise RuntimeError("Transfer was not reconciled in its target quarter.")
        hh, banks = country.households, country.banks
        if country._staged_household_income_shock is not None:
            raise RuntimeError("Native income settlement has not cleared the staged receipt.")
        if len(hh.ts.historic("income")) != self._period + 1:
            raise RuntimeError("Transfer requires exactly one completed income row.")
        expected = np.asarray(hh.compute_income(), dtype=float) + self.amounts_nominal
        actual = np.asarray(hh.ts.current("income"), dtype=float)
        if (
            expected.shape != actual.shape
            or not np.all(np.isfinite(expected))
            or not np.allclose(actual, expected, rtol=1e-10, atol=1e-7)
        ):
            raise RuntimeError("Realised income does not reconcile with the one-time receipt.")
        ids = np.asarray(self._layout_key[1], dtype=int)
        bank_households = np.bincount(
            ids, weights=hh.ts.current("liquid_financial_assets"), minlength=banks.ts.current("n_banks")
        )
        for field, expected_balance in (
            ("deposits_from_households", bank_households),
            ("deposits", np.asarray(banks.compute_deposits(), dtype=float)),
        ):
            balance = np.asarray(banks.ts.current(field), dtype=float)
            if (
                balance.shape != expected_balance.shape
                or not np.all(np.isfinite(balance))
                or not np.all(np.isfinite(expected_balance))
                or not np.allclose(balance, expected_balance, rtol=1e-10, atol=1e-7)
            ):
                raise RuntimeError(f"Native bank {field} does not reconcile with settlement.")
        allocated = np.bincount(ids, weights=self.amounts_nominal, minlength=banks.ts.current("n_banks"))
        self.booking = TransferBooking(
            year, month, self._period, self.amounts_nominal, allocated, float(self.amounts_nominal.sum())
        )
