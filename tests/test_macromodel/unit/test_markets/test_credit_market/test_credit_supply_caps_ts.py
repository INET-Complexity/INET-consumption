import numpy as np

from macromodel.markets.credit_market.credit_market import (
    _append_credit_supply_caps_to_banks_ts,
    _compute_credit_supply_caps_by_type,
)
from macromodel.timeseries import TimeSeries


class _BankParams:
    def __init__(self, capital_adequacy_ratio: float):
        self.capital_adequacy_ratio = capital_adequacy_ratio


class _BanksStub:
    def __init__(self, ts: TimeSeries, parameters: _BankParams):
        self.ts = ts
        self.parameters = parameters


def _make_banks_stub(
    *,
    equity: np.ndarray,
    total_outstanding_loans: np.ndarray,
    firms_fraction: np.ndarray,
    hh_cons_fraction: np.ndarray,
    mortgage_fraction: np.ndarray,
    capital_adequacy_ratio: float = 0.1,
) -> _BanksStub:
    n_banks = int(equity.size)
    ts = TimeSeries(
        n_banks=n_banks,
        equity=equity,
        total_outstanding_loans=total_outstanding_loans,
        new_loans_fraction_firms=firms_fraction,
        new_loans_fraction_hh_cons=hh_cons_fraction,
        new_loans_fraction_mortgages=mortgage_fraction,
        credit_supply_cap_total=np.full(n_banks, np.nan),
        credit_supply_cap_firms=np.full(n_banks, np.nan),
        credit_supply_cap_firms_short_term=np.full(n_banks, np.nan),
        credit_supply_cap_firms_long_term=np.full(n_banks, np.nan),
        credit_supply_cap_households_consumption=np.full(n_banks, np.nan),
        credit_supply_cap_mortgages=np.full(n_banks, np.nan),
        total_credit_supply_cap_total=[np.nan],
        total_credit_supply_cap_firms=[np.nan],
        total_credit_supply_cap_firms_short_term=[np.nan],
        total_credit_supply_cap_firms_long_term=[np.nan],
        total_credit_supply_cap_households_consumption=[np.nan],
        total_credit_supply_cap_mortgages=[np.nan],
    )
    return _BanksStub(ts=ts, parameters=_BankParams(capital_adequacy_ratio=capital_adequacy_ratio))


def test_compute_credit_supply_caps_by_type_matches_hand_calculation_when_temperature_is_zero():
    banks = _make_banks_stub(
        equity=np.array([10.0, 20.0]),
        total_outstanding_loans=np.array([0.0, 0.0]),
        firms_fraction=np.array([0.5, 0.5]),
        hh_cons_fraction=np.array([0.3, 0.3]),
        mortgage_fraction=np.array([0.2, 0.2]),
        capital_adequacy_ratio=0.1,
    )

    caps = _compute_credit_supply_caps_by_type(
        banks=banks,
        current_npl_firm_loans=0.0,
        current_npl_hh_cons_loans=0.0,
        current_npl_mortgages=0.0,
        credit_supply_temperature=0.0,
        total_target_short_term_credit=100.0,
        total_target_long_term_credit=0.0,
    )

    assert np.allclose(caps["total"], np.array([100.0, 200.0]))
    assert np.allclose(caps["firms"], np.array([50.0, 100.0]))
    assert np.allclose(caps["firms_short_term"], np.array([50.0, 100.0]))
    assert np.allclose(caps["firms_long_term"], 0.0)
    assert np.allclose(caps["households_consumption"], np.array([30.0, 60.0]))
    assert np.allclose(caps["mortgages"], np.array([20.0, 40.0]))


def test_compute_credit_supply_caps_by_type_returns_zero_type_caps_when_no_weights():
    banks = _make_banks_stub(
        equity=np.array([10.0, 20.0]),
        total_outstanding_loans=np.array([0.0, 0.0]),
        firms_fraction=np.array([0.0, 0.0]),
        hh_cons_fraction=np.array([0.0, 0.0]),
        mortgage_fraction=np.array([0.0, 0.0]),
        capital_adequacy_ratio=0.1,
    )

    caps = _compute_credit_supply_caps_by_type(
        banks=banks,
        current_npl_firm_loans=0.0,
        current_npl_hh_cons_loans=0.0,
        current_npl_mortgages=0.0,
        credit_supply_temperature=2.0,
        total_target_short_term_credit=0.0,
        total_target_long_term_credit=0.0,
    )

    assert np.allclose(caps["total"], np.array([100.0, 200.0]))
    assert np.allclose(caps["firms"], 0.0)
    assert np.allclose(caps["households_consumption"], 0.0)
    assert np.allclose(caps["mortgages"], 0.0)


def test_append_credit_supply_caps_updates_banks_ts_vectors_and_totals():
    banks = _make_banks_stub(
        equity=np.array([10.0, 20.0]),
        total_outstanding_loans=np.array([0.0, 0.0]),
        firms_fraction=np.array([0.5, 0.5]),
        hh_cons_fraction=np.array([0.3, 0.3]),
        mortgage_fraction=np.array([0.2, 0.2]),
        capital_adequacy_ratio=0.1,
    )

    _append_credit_supply_caps_to_banks_ts(
        banks=banks,
        current_npl_firm_loans=0.0,
        current_npl_hh_cons_loans=0.0,
        current_npl_mortgages=0.0,
        credit_supply_temperature=0.0,
        total_target_short_term_credit=20.0,
        total_target_long_term_credit=80.0,
    )

    assert np.allclose(banks.ts.current("credit_supply_cap_total"), np.array([100.0, 200.0]))
    assert np.allclose(banks.ts.current("credit_supply_cap_firms"), np.array([50.0, 100.0]))
    assert np.allclose(banks.ts.current("credit_supply_cap_firms_short_term"), np.array([10.0, 20.0]))
    assert np.allclose(banks.ts.current("credit_supply_cap_firms_long_term"), np.array([40.0, 80.0]))
    assert np.allclose(banks.ts.current("credit_supply_cap_households_consumption"), np.array([30.0, 60.0]))
    assert np.allclose(banks.ts.current("credit_supply_cap_mortgages"), np.array([20.0, 40.0]))

    assert np.allclose(banks.ts.current("total_credit_supply_cap_total")[0], 300.0)
    assert np.allclose(banks.ts.current("total_credit_supply_cap_firms")[0], 150.0)
    assert np.allclose(banks.ts.current("total_credit_supply_cap_firms_short_term")[0], 30.0)
    assert np.allclose(banks.ts.current("total_credit_supply_cap_firms_long_term")[0], 120.0)
    assert np.allclose(banks.ts.current("total_credit_supply_cap_households_consumption")[0], 90.0)
    assert np.allclose(banks.ts.current("total_credit_supply_cap_mortgages")[0], 60.0)
