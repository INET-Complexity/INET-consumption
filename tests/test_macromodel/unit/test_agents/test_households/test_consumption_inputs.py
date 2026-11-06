"""Independent arithmetic and production-source checks for the inactive adapter."""

import pickle
from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from macromodel.agents.households.consumption_inputs import (
    ConsumptionInputAdapter,
    ConsumptionSources,
    IncomeHistories,
    MonetaryUnits,
    OpeningSnapshot,
    OpeningStockValues,
    PermanentIncomePayload,
    prepare_inputs,
)
from macromodel.configurations import CountryConfiguration, SimulationConfiguration, load_country_configuration
from macromodel.simulation import Simulation
from macromodel.timeseries import TimeSeries

UNITS = MonetaryUnits("EUR", 1)
TAGS = dict(period_id=2, pass_kind="first", household_layout_key="run-1:ordered-two")


def stocks(scale=1):
    # Signed liquid assets are valid; ledger debt deliberately differs from principal.
    return OpeningStockValues(
        *[
            np.array(pair) * scale
            for pair in (
                (100, -10),
                (200, 300),
                (400, 0),
                (55, 0),
                (22, 12),
                (50, 0),
                (20, 10),
            )
        ]
    )


def snapshot(**kwargs):
    return OpeningSnapshot(2, TAGS["household_layout_key"], UNITS, stocks(), **kwargs)


def payload(**kwargs):
    return PermanentIncomePayload(
        **(TAGS | dict(units=UNITS, monetary_income_nominal=[140, 90], pi_log_ratio_original=[0.13, -0.2]) | kwargs)
    )


def histories(**kwargs):
    # C0=100, dated real income [10,20,40,80] / [20,40,80,160].
    return IncomeHistories(
        **(
            dict(
                behavioural_income_nominal_history=[[10, 20], [40, 80], [40, 80], [160, 320]],
                subsistence_nominal_history=np.ones((4, 2)),
                consumer_price_history=[100, 200, 100, 200],
                current_consumer_price=200,
                initial_consumer_price=100,
                price_history_layout="aligned",
                income_units=UNITS,
                subsistence_units=UNITS,
            )
            | kwargs
        )
    )


def prepare(history=None, opening=None, pi=None, **tags):
    return prepare_inputs(history or histories(), opening or snapshot(), pi or payload(), **(TAGS | tags))


@pytest.mark.parametrize(
    "rows,expected",
    [
        (1, [80, 160]),
        (2, [np.sqrt(40 * 80), np.sqrt(80 * 160)]),
        (4, [800**0.5, 3200**0.5]),
        (5, [800**0.5, 3200**0.5]),
    ],
)
def test_hand_computed_window_and_startup(rows, expected):
    h = histories()
    b, s, c = h.behavioural_income_nominal_history, h.subsistence_nominal_history, h.consumer_price_history
    if rows == 5:
        b, s, c = np.vstack(([1e8, 1e8], b)), np.vstack(([1, 1], s)), np.r_[100, c]
    else:
        b, s, c = b[-rows:], s[-rows:], c[-rows:]
    result = prepare(
        replace(h, behavioural_income_nominal_history=b, subsistence_nominal_history=s, consumer_price_history=c)
    )
    np.testing.assert_allclose(result.income_base_real, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.annual_income_base_real, np.array(expected) * 4, rtol=1e-10, atol=1e-12)
    assert result.window_row_count == min(4, rows)
    np.testing.assert_allclose(result.opening_real.net_liquid_assets, [11.5, -11], rtol=1e-10, atol=1e-12)


def test_nonpositive_income_uses_each_dates_floor_and_retains_upstream_flags():
    raw = np.array([[0, 50], [-2, 0], [80, 30], [0, -5]])
    floor = np.array([[10, 20], [40, 40], [30, 30], [160, 100]])
    h = histories(behavioural_income_nominal_history=raw, subsistence_nominal_history=floor)
    result = prepare(h)
    expected = [(10 * 20 * 80 * 80) ** 0.25, (50 * 20 * 30 * 50) ** 0.25]
    np.testing.assert_allclose(result.income_base_real, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(result.floor_hit_counts, [3, 2])
    flags = (raw < floor).astype(float)
    already = replace(
        h,
        behavioural_income_nominal_history=np.maximum(raw, floor),
        upstream_raw_history=raw,
        upstream_floor_used_history=flags,
        upstream_component_floor_used_history=np.zeros_like(raw),
    )
    repeated = prepare(already)
    np.testing.assert_allclose(repeated.income_base_real, expected, rtol=1e-10, atol=1e-12)
    assert not repeated.floor_hit_mask.any()
    np.testing.assert_array_equal(repeated.histories.upstream_floor_used_history, flags)
    np.testing.assert_array_equal(repeated.histories.upstream_raw_history, raw)


@pytest.mark.parametrize("rows", [1, 4])
def test_declared_missing_current_appends_once(rows):
    h = histories()
    aligned = replace(
        h,
        behavioural_income_nominal_history=h.behavioural_income_nominal_history[-rows:],
        subsistence_nominal_history=h.subsistence_nominal_history[-rows:],
        consumer_price_history=h.consumer_price_history[-rows:],
    )
    missing = replace(
        aligned, price_history_layout="missing_current", consumer_price_history=aligned.consumer_price_history[:-1]
    )
    np.testing.assert_array_equal(prepare(aligned).income_base_real, prepare(missing).income_base_real)


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"consumer_price_history": [100, 100, 100, 199]}, "final price"),
        ({"consumer_price_history": [100, 100, 100]}, "count"),
        ({"price_history_layout": "missing_current"}, "count"),
        ({"price_history_layout": "guess"}, "layout"),
        ({"subsistence_nominal_history": [[1, 1]]}, "dimensions"),
        (
            {
                "behavioural_income_nominal_history": np.empty((0, 2)),
                "subsistence_nominal_history": np.empty((0, 2)),
                "consumer_price_history": [],
            },
            "nonempty",
        ),
        (
            {"behavioural_income_nominal_history": np.empty((4, 0)), "subsistence_nominal_history": np.empty((4, 0))},
            "nonempty",
        ),
        ({"subsistence_nominal_history": np.ones((4, 1))}, "dimensions"),
        ({"initial_consumer_price": 0}, "initial_consumer_price"),
        ({"current_consumer_price": -1}, "current_consumer_price"),
        ({"months_per_period": np.inf}, "months_per_period"),
        ({"history_window": 0}, "history_window"),
        ({"history_window": True}, "history_window"),
        ({"history_window": 2}, "quarterly history_window"),
        ({"upstream_floor_used_history": np.full((4, 2), 2)}, "upstream_floor"),
        ({"upstream_raw_history": np.ones((1, 2))}, "upstream_raw"),
    ],
)
def test_history_errors(updates, match):
    with pytest.raises(ValueError, match=match):
        histories(**updates)


@pytest.mark.parametrize(
    "name", ["behavioural_income_nominal_history", "subsistence_nominal_history", "consumer_price_history"]
)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_history_operands_rejected(name, bad):
    value = np.array(getattr(histories(), name), copy=True)
    value.flat[0] = bad
    with pytest.raises(ValueError, match=name):
        histories(**{name: value})


@pytest.mark.parametrize("name", ["subsistence_nominal_history", "consumer_price_history"])
@pytest.mark.parametrize("bad", [0, -1])
def test_positive_history_quantities(name, bad):
    value = np.array(getattr(histories(), name), copy=True)
    value.flat[0] = bad
    with pytest.raises(ValueError, match=name):
        histories(**{name: value})


def test_price_match_tolerance_and_period_labels():
    h = histories(
        current_consumer_price=200 * (1 + 5e-13),
        income_period_labels=(0, 1, 2, 3),
        subsistence_period_labels=(0, 1, 2, 3),
        price_period_labels=(0, 1, 2, 3),
    )
    prepare(h)
    prepare(
        replace(
            h,
            price_history_layout="missing_current",
            consumer_price_history=[100, 200, 100],
            price_period_labels=(0, 1, 2),
        )
    )
    for updates in (
        {"price_period_labels": (0, 1, 2, 4)},
        {"subsistence_period_labels": (0, 1, 1, 3)},
        {"price_period_labels": None},
    ):
        with pytest.raises(ValueError, match="labels"):
            replace(h, **updates)


@pytest.mark.parametrize("q", [[0.13, -0.2], [10000, -10000]])
def test_m_based_pi_level_preservation_without_exponentiating_q(q):
    pi = payload(pi_log_ratio_original=q)
    result = prepare(pi=pi)
    expected = np.log(np.array([70, 45])) + q
    np.testing.assert_allclose(result.log_permanent_income_proxy_real, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        np.log(result.income_base_real) + result.pi_log_ratio_rebased, expected, rtol=1e-10, atol=1e-12
    )
    changed = prepare(replace(histories(), behavioural_income_nominal_history=np.full((4, 2), 1000)), pi=pi)
    np.testing.assert_array_equal(changed.log_permanent_income_proxy_real, result.log_permanent_income_proxy_real)
    assert not np.allclose(changed.pi_log_ratio_rebased, result.pi_log_ratio_rebased)


@pytest.mark.parametrize(
    "name,bad",
    [
        ("monetary_income_nominal", [0, 10]),
        ("monetary_income_nominal", [-1, 10]),
        ("monetary_income_nominal", [np.nan, 10]),
        ("pi_log_ratio_original", [np.inf, 0]),
        ("pi_log_ratio_original", [0]),
    ],
)
def test_invalid_pi_payload(name, bad):
    with pytest.raises(ValueError):
        payload(**{name: bad})


@pytest.mark.parametrize(
    "updates", [{"period_id": 3}, {"pass_kind": "replacement"}, {"household_layout_key": "other-order"}]
)
def test_stale_pi_tags_are_rejected(updates):
    with pytest.raises(ValueError, match="tags|pass_kind"):
        prepare(pi=payload(**updates))


@pytest.mark.parametrize("period", [-1, True, 1.5])
def test_invalid_period_identity(period):
    with pytest.raises(ValueError, match="period_id"):
        payload(period_id=period)


def test_missing_pi_and_household_dimensions():
    with pytest.raises(ValueError, match="required"):
        prepare_inputs(histories(), snapshot(), None, **TAGS)
    with pytest.raises(ValueError, match="missing"):
        PermanentIncomePayload.from_learning_inputs({}, **TAGS, units=UNITS)
    with pytest.raises(ValueError, match="dimensions"):
        prepare(pi=payload(monetary_income_nominal=[1], pi_log_ratio_original=[0]))


@pytest.mark.parametrize("months", [1, 3, 12])
@pytest.mark.parametrize("scale", [1, 5000])
def test_common_scale_and_frequency_invariance(months, scale):
    units = MonetaryUnits("EUR", scale)
    h = histories(
        behavioural_income_nominal_history=np.full((4, 2), 1200 * months / 12 * scale),
        subsistence_nominal_history=np.full((4, 2), 10 * months / 12 * scale),
        consumer_price_history=[100] * 4,
        current_consumer_price=100,
        income_units=units,
        subsistence_units=units,
        months_per_period=months,
    )
    result = prepare(
        h,
        OpeningSnapshot(2, TAGS["household_layout_key"], units, stocks(scale)),
        payload(units=units, monetary_income_nominal=np.full(2, 1500 * months / 12 * scale)),
    )
    np.testing.assert_allclose(result.annual_income_base_real / scale, [1200, 1200], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        result.opening_real.net_liquid_assets / result.annual_income_base_real,
        [23 / 1200, -22 / 1200],
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.pi_log_ratio_rebased, np.array([0.13, -0.2]) + np.log(1.25), rtol=1e-10, atol=1e-12
    )


def test_explicit_unit_mismatches_rejected():
    other = MonetaryUnits("EUR", 5000)
    with pytest.raises(ValueError, match="unit metadata"):
        histories(subsistence_units=other)
    with pytest.raises(ValueError, match="unit metadata"):
        prepare(pi=payload(units=other))
    with pytest.raises(ValueError, match="unit metadata"):
        prepare(pi=payload(units=MonetaryUnits("USD", 1)))


def test_overflow_underflow_errors_do_not_silently_replace_results():
    with pytest.raises(ValueError):
        prepare(
            histories(current_consumer_price=1e308, initial_consumer_price=1e-308, consumer_price_history=[1e308] * 4)
        )
    with pytest.raises(ValueError, match="annual_income_base_real"):
        prepare(
            histories(
                behavioural_income_nominal_history=np.full((4, 2), 1e308),
                consumer_price_history=[100] * 4,
                current_consumer_price=100,
            )
        )


def synthetic_sources():
    st = stocks()
    hh_ts = TimeSeries(
        n_households=2,
        **{
            "liquid_financial_assets": st.liquid_financial_assets.copy(),
            "illiquid_financial_assets": st.illiquid_financial_assets.copy(),
            "wealth_main_residence": st.main_residence_wealth.copy(),
            "mortgage_debt": st.mortgage_ledger_debt.copy(),
            "consumption_loan_debt": st.consumer_ledger_debt.copy(),
            "wealth_other_properties": np.array([1e6, 1e6]),
            "expected_behavioural_income": np.array([100.0, 100.0]),
        },
    )
    hh_ts.dicts["liquid_financial_assets"].insert(0, np.array([9999.0, 9999.0]))  # t-2
    household = SimpleNamespace(
        ts=hh_ts,
        states={"Tenure Status of the Main Residence": np.array([1, 3])},
        last_target_consumption_components={"permanent_income_log_ratio": np.array([999, 999])},
    )
    econ_ts = TimeSeries(selected_cpi=np.array([100.0]), subsistence_consumption=np.array([10.0, 10.0]))
    economy = SimpleNamespace(
        ts=econ_ts,
        time_unit=3,
        consumer_price_level_series_name=lambda: "selected_cpi",
        initial_consumer_price_level=lambda: 100.0,
    )
    credit = SimpleNamespace(
        household_contractual_debt_rate_components=Mock(
            return_value=(np.zeros(2), st.consumer_principal.copy(), np.zeros(2), st.mortgage_principal.copy())
        )
    )
    return ConsumptionSources(household, economy, credit, UNITS, UNITS, UNITS, UNITS)


def collect(adapter, sources, pi=None, **tags):
    return adapter.collect_inputs(
        sources, pi or payload(), **(TAGS | tags), current_consumer_price=100, price_history_layout="aligned"
    )


def test_explicit_two_pass_lifecycle_mutations_next_period_and_reset():
    sources = synthetic_sources()
    adapter = ConsumptionInputAdapter()
    with pytest.raises(ValueError, match="matching captured"):
        collect(adapter, sources, payload(pass_kind="replacement"), pass_kind="replacement")
    opening = adapter.capture_opening(sources, **TAGS)
    first = collect(adapter, sources)
    np.testing.assert_array_equal(first.opening_real.liquid_financial_assets, [100, -10])
    # Intervening post-housing, asset and loan states must never be recaptured.
    sources.households.ts.liquid_financial_assets.append(np.array([500.0, 600.0]))
    sources.households.ts.override_current("wealth_main_residence", np.array([0.0, 1000.0]))
    sources.households.states["Tenure Status of the Main Residence"][:] = [3, 1]
    sources.households.ts.override_current("expected_behavioural_income", np.array([400.0, 400.0]))
    sources.credit_market.household_contractual_debt_rate_components.return_value[1][:] = 999
    second = collect(
        adapter,
        sources,
        payload(pass_kind="replacement", monetary_income_nominal=[200, 300], pi_log_ratio_original=[0.5, 0.7]),
        pass_kind="replacement",
    )
    assert second.opening_snapshot is opening
    np.testing.assert_array_equal(second.opening_real.main_residence_wealth, [400, 0])
    np.testing.assert_array_equal(second.opening_real.consumer_principal, [20, 10])
    np.testing.assert_array_equal(second.opening_real.consumer_ledger_debt, [22, 12])
    np.testing.assert_allclose(second.income_base_real, [400, 400], rtol=1e-10, atol=1e-12)
    assert not np.allclose(second.pi_log_ratio_rebased, first.pi_log_ratio_rebased)
    assert sources.credit_market.household_contractual_debt_rate_components.call_count == 1
    for updates in (
        {},
        {"period_id": 1},
        {"period_id": 3, "household_layout_key": "reordered"},
        {"pass_kind": "replacement"},
    ):
        with pytest.raises(ValueError):
            adapter.capture_opening(sources, **(TAGS | updates))
        assert adapter.opening_snapshot is opening
    assert sources.credit_market.household_contractual_debt_rate_components.call_count == 1
    with pytest.raises(ValueError):
        collect(adapter, sources, payload(period_id=3), period_id=3)
    next_opening = adapter.capture_opening(sources, **(TAGS | {"period_id": 3}))
    np.testing.assert_array_equal(next_opening.nominal.liquid_financial_assets, [500, 600])
    adapter.reset()
    assert adapter.opening_snapshot is None
    fresh = adapter.capture_opening(sources, period_id=0, household_layout_key="run-2:new-order")
    assert fresh.period_id == 0


def test_changed_household_count_and_missing_sources_fail_without_replacing_snapshot():
    sources = synthetic_sources()
    adapter = ConsumptionInputAdapter()
    original = adapter.capture_opening(sources, **TAGS)
    sources.households.ts.override_current("n_households", 3)
    with pytest.raises(ValueError, match="dimensions"):
        adapter.capture_opening(sources, **(TAGS | {"period_id": 3}))
    assert adapter.opening_snapshot is original
    with pytest.raises(ValueError, match="dimensions"):
        collect(adapter, sources)
    sources.households.ts.override_current("n_households", 2)
    del sources.households.ts.dicts["expected_behavioural_income"]
    with pytest.raises(ValueError, match="missing"):
        collect(adapter, sources)


def test_copies_own_all_arrays_and_cannot_be_made_writable():
    source = np.array([140.0, 90.0])
    pi = payload(monetary_income_nominal=source)
    h = histories()
    result = prepare(h, pi=pi)
    source[:] = -999
    np.testing.assert_array_equal(result.pi_payload.monetary_income_nominal, [140, 90])
    for record in (pi, h, result, result.opening_snapshot.nominal, result.opening_real):
        for field in fields(record):
            array = getattr(record, field.name)
            if isinstance(array, np.ndarray):
                assert not array.flags.writeable
                with pytest.raises(ValueError):
                    array.setflags(write=True)
    assert deepcopy(result) is result
    assert deepcopy(result).income_base_real.flags.writeable is False


def test_production_sources_and_ready_pi_are_isolated_across_both_passes(datawrapper, monkeypatch):
    root = Path(__file__).resolve().parents[5]
    production = load_country_configuration(root / "run_model/config/country_config_FRA.yaml", country_iso3="FRA")
    config = CountryConfiguration()
    config.households = production.households
    config.households.functions.wealth.parameters["frm_coefficients_path"] = str(
        root / "run_model/data/raw_data/portfolio/FR_portfolio_frm_coefficients.json"
    )
    config.banks.parameters.household_consumption_loan_maturity = (
        production.banks.parameters.household_consumption_loan_maturity
    )
    config.banks.parameters.requires_household_underwriting_snapshot = True
    config.consumer_credit, config.credit_market = production.consumer_credit, production.credit_market
    simulation = Simulation.from_datawrapper(
        datawrapper=datawrapper,
        simulation_configuration=SimulationConfiguration(country_configurations={"FRA": config}, seed=15),
    )
    country = simulation.countries["FRA"]
    units = MonetaryUnits("EUR", country.scale)
    sources = ConsumptionSources(country.households, country.economy, country.credit_market, units, units, units, units)
    hh, econ = country.households, country.economy
    monkeypatch.setattr(econ, "current_expected_consumer_period_inflation", lambda: 0.0)
    adapter = ConsumptionInputAdapter()
    period_id = len(hh.ts.historic("income"))
    tags = dict(period_id=period_id, household_layout_key="production:initial-order")
    opening = adapter.capture_opening(sources, **tags)
    original = hh.current_income_belief_learning_inputs
    ready = []

    def producer(**kwargs):
        result = original(**kwargs)
        ready.append(
            PermanentIncomePayload.from_learning_inputs(
                result, **tags, pass_kind="first" if not ready else "replacement", units=units
            )
        )
        return result

    monkeypatch.setattr(hh, "current_income_belief_learning_inputs", producer)
    results = []
    for pass_kind in ("first", "replacement"):
        # Existing producers/target run once; collection is then invoked independently.
        country._set_household_target_demand(replace_current=pass_kind == "replacement")
        before = pickle.dumps(
            (
                hh.ts.dicts,
                econ.ts.dicts,
                hh.states,
                {k: v for k, v in hh.functions["consumption"].__dict__.items() if k != "compute_target_consumption"},
                np.random.get_state(),
                country.credit_market.states,
            )
        )
        with monkeypatch.context() as guard:
            guard.setattr(
                hh, "current_income_belief_learning_inputs", Mock(side_effect=AssertionError("producer rerun"))
            )
            guard.setattr(
                hh.functions["consumption"],
                "compute_target_consumption",
                Mock(side_effect=AssertionError("target invoked")),
            )
            result = adapter.collect_inputs(
                sources,
                ready[-1],
                **tags,
                pass_kind=pass_kind,
                current_consumer_price=econ.current_consumer_price_level(),
                price_history_layout="missing_current",
            )
        after = pickle.dumps(
            (
                hh.ts.dicts,
                econ.ts.dicts,
                hh.states,
                {k: v for k, v in hh.functions["consumption"].__dict__.items() if k != "compute_target_consumption"},
                np.random.get_state(),
                country.credit_market.states,
            )
        )
        assert before == after
        assert result.opening_snapshot is opening
        assert result.histories.upstream_raw_history is not None
        np.testing.assert_array_equal(
            result.histories.upstream_floor_used_history, hh.ts.historic("expected_behavioural_income_floor_used")
        )
        results.append(result)
    assert len(ready) == 2
    np.testing.assert_allclose(
        results[0].income_base_real, results[1].income_base_real, rtol=1e-10, atol=1e-12 * country.scale
    )
    np.testing.assert_array_equal(opening.nominal.main_residence_wealth, hh.ts.current("wealth_main_residence"))


@pytest.mark.parametrize("name", [field.name for field in fields(OpeningStockValues)])
def test_invalid_opening_stock_operands(name):
    with pytest.raises(ValueError, match=name):
        replace(stocks(), **{name: [np.nan, 0]})
    with pytest.raises(ValueError, match="dimensions"):
        replace(stocks(), **{name: [1]})
    if name != "liquid_financial_assets":
        with pytest.raises(ValueError, match="nonnegative"):
            replace(stocks(), **{name: [-1, 0]})


def test_collector_rejects_multicolumn_price_rows_and_preserves_payload_copy():
    sources = synthetic_sources()
    adapter = ConsumptionInputAdapter()
    adapter.capture_opening(sources, **TAGS)
    ready = {"income_bridge_current": np.array([140.0, 90.0]), "permanent_income_log_ratio": np.array([0.13, -0.2])}
    pi = PermanentIncomePayload.from_learning_inputs(ready, **TAGS, units=UNITS)
    ready["income_bridge_current"][:] = 999
    ready["permanent_income_log_ratio"][:] = 999
    result = collect(adapter, sources, pi)
    np.testing.assert_allclose(
        result.log_permanent_income_proxy_real, np.log([140, 90]) + [0.13, -0.2], rtol=1e-10, atol=1e-12
    )
    sources.economy.ts.override_current("selected_cpi", np.array([100.0, 100.0]))
    with pytest.raises(ValueError, match="consumer_price_history"):
        collect(adapter, sources, pi)


def test_adapter_deepcopy_can_advance_and_reset_independently():
    sources = synthetic_sources()
    adapter = ConsumptionInputAdapter()
    opening = adapter.capture_opening(sources, **TAGS)
    cloned = deepcopy(adapter)
    assert cloned is not adapter
    assert cloned.opening_snapshot is opening  # Immutable data may safely be shared.
    cloned.capture_opening(sources, **(TAGS | {"period_id": 3}))
    assert adapter.opening_snapshot is opening
    cloned.reset()
    assert adapter.opening_snapshot is opening
