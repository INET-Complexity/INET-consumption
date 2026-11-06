"""Explicit, inactive income/state adapter for the retained consumption rule.

Owners capture before transactions and supply a ready PI result for each pass.
This module never runs a producer, writes a TimeSeries, or evaluates a target.
MonetaryUnits are declarations by the owner, not inferred from numeric magnitudes.
"""

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Literal, Mapping

import numpy as np

if TYPE_CHECKING:
    from macromodel.agents.households.households import Households
    from macromodel.economy.economy import Economy
    from macromodel.markets.credit_market.credit_market import CreditMarket

PassKind = Literal["first", "replacement"]
PriceHistoryLayout = Literal["aligned", "missing_current"]


def _array(value, name, *, ndim, positive=False, nonnegative=False):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array.") from exc
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array.")
    if positive and np.any(array <= 0):
        raise ValueError(f"{name} must be positive.")
    if nonnegative and np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative.")
    # Immutable bytes backing also prevents callers re-enabling WRITEABLE.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _positive_scalar(value, name):
    if np.ndim(value) != 0 or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite scalar.")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite scalar.") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite scalar.")
    return value


def _identity(period_id, household_layout_key, pass_kind="first"):
    if isinstance(period_id, (bool, np.bool_)) or not isinstance(period_id, (int, np.integer)) or period_id < 0:
        raise ValueError("period_id must be a nonnegative integer.")
    if not isinstance(household_layout_key, str) or not household_layout_key.strip():
        raise ValueError("household_layout_key must identify the run and household order.")
    if pass_kind not in ("first", "replacement"):
        raise ValueError("pass_kind must be first or replacement.")


class _ImmutableRecord:
    """Immutable records can be shared safely when a run is deep-copied."""

    def __deepcopy__(self, memo):
        return self


@dataclass(frozen=True)
class MonetaryUnits(_ImmutableRecord):
    """Nominal currency and common population expansion, declared per source."""

    currency: str
    population_scale: float

    def __post_init__(self):
        if not isinstance(self.currency, str) or not self.currency.strip():
            raise ValueError("currency must be a nonempty currency identifier.")
        object.__setattr__(self, "population_scale", _positive_scalar(self.population_scale, "population_scale"))


def _units(*values):
    if not values or any(not isinstance(value, MonetaryUnits) or value != values[0] for value in values):
        raise ValueError("Monetary unit metadata must match across income, floors, PI and stocks.")


@dataclass(frozen=True)
class OpeningStockValues(_ImmutableRecord):
    """Nominal or real H-vectors; ledger debts include interest arrears."""

    liquid_financial_assets: np.ndarray
    illiquid_financial_assets: np.ndarray
    main_residence_wealth: np.ndarray
    mortgage_ledger_debt: np.ndarray
    consumer_ledger_debt: np.ndarray
    mortgage_principal: np.ndarray
    consumer_principal: np.ndarray

    def __post_init__(self):
        shape = None
        for field in fields(self):
            value = _array(
                getattr(self, field.name),
                field.name,
                ndim=1,
                nonnegative=field.name != "liquid_financial_assets",
            )
            if value.size == 0 or (shape is not None and value.shape != shape):
                raise ValueError("opening stocks must have the same nonempty household dimensions.")
            shape = value.shape
            object.__setattr__(self, field.name, value)

    @property
    def net_liquid_assets(self):
        with np.errstate(over="ignore", invalid="ignore"):
            value = self.liquid_financial_assets - self.mortgage_ledger_debt - self.consumer_ledger_debt
        return _array(value, "net_liquid_assets", ndim=1)


@dataclass(frozen=True)
class OpeningSnapshot(_ImmutableRecord):
    period_id: int
    household_layout_key: str
    units: MonetaryUnits
    nominal: OpeningStockValues

    def __post_init__(self):
        _identity(self.period_id, self.household_layout_key)
        _units(self.units)
        if not isinstance(self.nominal, OpeningStockValues):
            raise ValueError("nominal must be OpeningStockValues.")


@dataclass(frozen=True)
class PermanentIncomePayload(_ImmutableRecord):
    """M and q copied together from one ready learning_inputs result."""

    period_id: int
    pass_kind: PassKind
    household_layout_key: str
    units: MonetaryUnits
    monetary_income_nominal: np.ndarray
    pi_log_ratio_original: np.ndarray

    def __post_init__(self):
        _identity(self.period_id, self.household_layout_key, self.pass_kind)
        _units(self.units)
        for name in ("monetary_income_nominal", "pi_log_ratio_original"):
            object.__setattr__(
                self, name, _array(getattr(self, name), name, ndim=1, positive=name == "monetary_income_nominal")
            )
        if (
            self.monetary_income_nominal.size == 0
            or self.monetary_income_nominal.shape != self.pi_log_ratio_original.shape
        ):
            raise ValueError("PI payload must have matching nonempty household dimensions.")

    @classmethod
    def from_learning_inputs(cls, learning_inputs: Mapping, **metadata):
        """Copy without calling the learning producer or reading target diagnostics."""
        try:
            return cls(
                **metadata,
                monetary_income_nominal=learning_inputs["income_bridge_current"],
                pi_log_ratio_original=learning_inputs["permanent_income_log_ratio"],
            )
        except KeyError as exc:
            raise ValueError(f"PI payload is missing {exc.args[0]}.") from exc


@dataclass(frozen=True)
class IncomeHistories(_ImmutableRecord):
    """Unwindowed, declared-aligned nominal inputs; labels are optional row IDs.

    If any labels are supplied, all three paths must carry corresponding labels.
    Optional upstream histories are retained only where the producer supplies them.
    """

    behavioural_income_nominal_history: np.ndarray
    subsistence_nominal_history: np.ndarray
    consumer_price_history: np.ndarray
    current_consumer_price: float
    initial_consumer_price: float
    price_history_layout: PriceHistoryLayout
    income_units: MonetaryUnits
    subsistence_units: MonetaryUnits
    months_per_period: float = 3
    history_window: int = 4
    income_period_labels: tuple | None = None
    subsistence_period_labels: tuple | None = None
    price_period_labels: tuple | None = None
    upstream_raw_history: np.ndarray | None = None
    upstream_floor_used_history: np.ndarray | None = None
    upstream_component_floor_used_history: np.ndarray | None = None

    def __post_init__(self):
        _units(self.income_units, self.subsistence_units)
        for name in ("current_consumer_price", "initial_consumer_price", "months_per_period"):
            object.__setattr__(self, name, _positive_scalar(getattr(self, name), name))
        if (
            isinstance(self.history_window, (bool, np.bool_))
            or not isinstance(self.history_window, (int, np.integer))
            or self.history_window <= 0
        ):
            raise ValueError("history_window must be a positive integer.")
        if self.months_per_period == 3 and self.history_window != 4:
            raise ValueError("quarterly history_window must be 4.")
        for name in ("behavioural_income_nominal_history", "subsistence_nominal_history", "consumer_price_history"):
            object.__setattr__(
                self,
                name,
                _array(
                    getattr(self, name),
                    name,
                    ndim=1 if name == "consumer_price_history" else 2,
                    positive=name != "behavioural_income_nominal_history",
                ),
            )
        shape = self.behavioural_income_nominal_history.shape
        if not all(shape) or self.subsistence_nominal_history.shape != shape:
            raise ValueError("B and subsistence histories must have identical nonempty K x H dimensions.")
        k = shape[0]
        missing = self.price_history_layout == "missing_current"
        if self.price_history_layout not in ("aligned", "missing_current") or self.consumer_price_history.shape != (
            k - int(missing),
        ):
            raise ValueError("consumer_price_history count must match the declared price_history_layout.")
        if not missing and not np.isclose(
            self.consumer_price_history[-1], self.current_consumer_price, rtol=1e-12, atol=0
        ):
            raise ValueError("consumer_price_history final price must match current_consumer_price.")
        label_names = ("income_period_labels", "subsistence_period_labels", "price_period_labels")
        labels = [getattr(self, name) for name in label_names]
        if any(value is not None for value in labels):
            if any(value is None for value in labels):
                raise ValueError("period labels must be supplied for all three histories.")
            labels = [tuple(value) for value in labels]
            if len(labels[0]) != k or labels[0] != labels[1] or labels[2] != labels[0][: k - int(missing)]:
                raise ValueError("income, subsistence and price period labels must match.")
            for name, value in zip(label_names, labels):
                object.__setattr__(self, name, value)
        for name in ("upstream_raw_history", "upstream_floor_used_history", "upstream_component_floor_used_history"):
            value = getattr(self, name)
            if value is not None:
                value = _array(value, name, ndim=2)
                if value.shape != shape or (name != "upstream_raw_history" and not np.all((value == 0) | (value == 1))):
                    raise ValueError(f"{name} must match B dimensions; flags must be zero or one.")
                object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ConsumptionInputs(_ImmutableRecord):
    """Prepared inputs only: no maps, targets, ECM or normalized rate indices."""

    period_id: int
    pass_kind: PassKind
    household_layout_key: str
    histories: IncomeHistories
    pi_payload: PermanentIncomePayload
    opening_snapshot: OpeningSnapshot
    opening_real: OpeningStockValues
    income_base_real: np.ndarray
    annual_income_base_real: np.ndarray
    pi_log_ratio_rebased: np.ndarray
    pi_log_denominator_adjustment: np.ndarray
    log_permanent_income_proxy_real: np.ndarray
    floor_hit_mask: np.ndarray
    floor_hit_counts: np.ndarray
    window_row_count: int
    current_deflator: float

    def __post_init__(self):
        for name in (
            "income_base_real",
            "annual_income_base_real",
            "pi_log_ratio_rebased",
            "pi_log_denominator_adjustment",
            "log_permanent_income_proxy_real",
        ):
            object.__setattr__(
                self,
                name,
                _array(
                    getattr(self, name), name, ndim=1, positive=name in ("income_base_real", "annual_income_base_real")
                ),
            )
        for name, dtype in (("floor_hit_mask", bool), ("floor_hit_counts", np.int64)):
            value = np.asarray(getattr(self, name), dtype=dtype)
            object.__setattr__(self, name, np.frombuffer(value.tobytes(), dtype=dtype).reshape(value.shape))


def prepare_inputs(
    histories: IncomeHistories,
    opening_snapshot: OpeningSnapshot,
    pi_payload: PermanentIncomePayload,
    *,
    period_id: int,
    pass_kind: PassKind,
    household_layout_key: str,
) -> ConsumptionInputs:
    """Pure validated history smoothing and M-based PI rebase in log arithmetic."""
    _identity(period_id, household_layout_key, pass_kind)
    if (
        not isinstance(histories, IncomeHistories)
        or not isinstance(opening_snapshot, OpeningSnapshot)
        or not isinstance(pi_payload, PermanentIncomePayload)
    ):
        raise ValueError("histories, opening_snapshot and same-pass pi_payload are required typed records.")
    for record in (opening_snapshot, pi_payload):
        if (record.period_id, record.household_layout_key) != (period_id, household_layout_key):
            raise ValueError("opening snapshot and PI payload period/layout tags must match preparation.")
    if pi_payload.pass_kind != pass_kind:
        raise ValueError("PI payload pass_kind must match preparation.")
    _units(histories.income_units, histories.subsistence_units, opening_snapshot.units, pi_payload.units)
    h = histories.behavioural_income_nominal_history.shape[1]
    if opening_snapshot.nominal.liquid_financial_assets.shape != (h,) or pi_payload.monetary_income_nominal.shape != (
        h,
    ):
        raise ValueError("history, snapshot and PI household dimensions must match.")
    prices = histories.consumer_price_history
    if histories.price_history_layout == "missing_current":
        prices = np.append(prices, histories.current_consumer_price)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        deflators = prices / histories.initial_consumer_price
        d = histories.current_consumer_price / histories.initial_consumer_price
    _array(deflators, "historical deflators", ndim=1, positive=True)
    _positive_scalar(d, "current_deflator")
    n = min(histories.history_window, prices.size)
    b_nom = histories.behavioural_income_nominal_history[-n:]
    s_nom = histories.subsistence_nominal_history[-n:]
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        log_b = np.mean(
            np.log(np.maximum(b_nom, s_nom))
            - (np.log(prices[-n:]) - np.log(histories.initial_consumer_price))[:, None],
            axis=0,
        )
        b = np.exp(log_b)
        annual_b = (12 / histories.months_per_period) * b
        log_m_real = np.log(pi_payload.monetary_income_nominal) - np.log(d)
        adjustment = log_m_real - log_b
        proxy = log_m_real + pi_payload.pi_log_ratio_original
        rebased = pi_payload.pi_log_ratio_original + adjustment
        real = OpeningStockValues(
            **{field.name: getattr(opening_snapshot.nominal, field.name) / d for field in fields(OpeningStockValues)}
        )
    # Evaluate NLA now so overflowing ledger subtraction fails here, not downstream.
    real.net_liquid_assets
    return ConsumptionInputs(
        period_id,
        pass_kind,
        household_layout_key,
        histories,
        pi_payload,
        opening_snapshot,
        real,
        b,
        annual_b,
        rebased,
        adjustment,
        proxy,
        b_nom < s_nom,
        (b_nom < s_nom).sum(axis=0),
        n,
        d,
    )


@dataclass(frozen=True)
class ConsumptionSources:
    """Read-only production objects with explicitly declared monetary units.

    Sources remain mutable and are read at the requested collection point.
    Only produced records own arrays. The owner guarantees flows are ready.
    """

    households: "Households"
    economy: "Economy"
    credit_market: "CreditMarket"
    income_units: MonetaryUnits
    subsistence_units: MonetaryUnits
    stock_units: MonetaryUnits
    principal_units: MonetaryUnits

    def __post_init__(self):
        _units(self.income_units, self.subsistence_units, self.stock_units, self.principal_units)


class ConsumptionInputAdapter:
    """One explicit opening snapshot per period/run; no active runtime hooks."""

    def __init__(self):
        self._opening_snapshot: OpeningSnapshot | None = None

    @property
    def opening_snapshot(self) -> OpeningSnapshot | None:
        return self._opening_snapshot

    def reset(self) -> None:
        self._opening_snapshot = None

    def capture_opening(
        self,
        sources: ConsumptionSources,
        *,
        period_id: int,
        household_layout_key: str,
        pass_kind: PassKind = "first",
    ) -> OpeningSnapshot:
        """Copy pre-transaction current HH stocks and pre-credit principal exposures."""
        _identity(period_id, household_layout_key, pass_kind)
        old = self._opening_snapshot
        if pass_kind != "first" or (old is not None and period_id <= old.period_id):
            raise ValueError("capture_opening requires a later first pass; duplicate/backward capture is forbidden.")
        if old is not None and household_layout_key != old.household_layout_key:
            raise ValueError("household layout changed; reset before starting a different run/layout.")
        hh = sources.households.ts
        _, consumer, _, mortgage = sources.credit_market.household_contractual_debt_rate_components(
            use_opening_schedule=False
        )
        stocks = OpeningStockValues(
            hh.current("liquid_financial_assets"),
            hh.current("illiquid_financial_assets"),
            hh.current("wealth_main_residence"),
            hh.current("mortgage_debt"),
            hh.current("consumption_loan_debt"),
            mortgage,
            consumer,
        )
        h = hh.current("n_households")
        if stocks.liquid_financial_assets.shape != (h,) or (
            old is not None and stocks.liquid_financial_assets.shape != old.nominal.liquid_financial_assets.shape
        ):
            raise ValueError("opening stocks household dimensions changed; reset required.")
        if old is not None:
            _units(old.units, sources.stock_units)
        snapshot = OpeningSnapshot(period_id, household_layout_key, sources.stock_units, stocks)
        self._opening_snapshot = snapshot
        return snapshot

    def prepare(
        self,
        histories: IncomeHistories,
        pi_payload: PermanentIncomePayload,
        *,
        period_id: int,
        pass_kind: PassKind,
        household_layout_key: str,
    ) -> ConsumptionInputs:
        return prepare_inputs(
            histories,
            self._opening_snapshot,
            pi_payload,
            period_id=period_id,
            pass_kind=pass_kind,
            household_layout_key=household_layout_key,
        )

    def collect_inputs(
        self,
        sources: ConsumptionSources,
        pi_payload: PermanentIncomePayload,
        *,
        period_id: int,
        pass_kind: PassKind,
        household_layout_key: str,
        current_consumer_price: float,
        price_history_layout: PriceHistoryLayout,
        history_window: int = 4,
    ) -> ConsumptionInputs:
        """Read ready B/S/selected CPI paths, without producer calls or stock recapture."""
        _identity(period_id, household_layout_key, pass_kind)
        snapshot = self._opening_snapshot
        if snapshot is None or (snapshot.period_id, snapshot.household_layout_key) != (period_id, household_layout_key):
            raise ValueError("collection requires a matching captured opening snapshot.")
        _units(
            snapshot.units,
            sources.income_units,
            sources.subsistence_units,
            sources.stock_units,
            sources.principal_units,
        )
        hh, economy = sources.households.ts, sources.economy
        if snapshot.nominal.liquid_financial_assets.shape != (hh.current("n_households"),):
            raise ValueError("source household dimensions changed; reset required.")
        price_name = economy.consumer_price_level_series_name()
        optional = {}
        for output, suffix in (
            ("upstream_raw_history", "raw"),
            ("upstream_floor_used_history", "floor_used"),
            ("upstream_component_floor_used_history", "component_floor_used"),
        ):
            key = "expected_behavioural_income_" + suffix
            if key in hh.dicts:
                optional[output] = hh.historic(key)
        try:
            prices = np.asarray(economy.ts.historic(price_name), dtype=float)
            if prices.ndim == 2 and prices.shape[1] == 1:
                prices = prices[:, 0]
            histories = IncomeHistories(
                hh.historic("expected_behavioural_income"),
                economy.ts.historic("subsistence_consumption"),
                prices,
                current_consumer_price,
                economy.initial_consumer_price_level(),
                price_history_layout,
                sources.income_units,
                sources.subsistence_units,
                months_per_period=economy.time_unit,
                history_window=history_window,
                **optional,
            )
        except KeyError as exc:
            raise ValueError(f"consumption source is missing {exc.args[0]}.") from exc
        return self.prepare(
            histories, pi_payload, period_id=period_id, pass_kind=pass_kind, household_layout_key=household_layout_key
        )
