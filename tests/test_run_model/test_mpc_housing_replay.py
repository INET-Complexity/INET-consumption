from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from src.experiments.mpc_housing_replay import HousingReplayMismatch, HousingReplayTape


def country(cash=4.0):
    return SimpleNamespace(
        housing_market=SimpleNamespace(
            states={"current_sales": pd.DataFrame({"buyer_id": [1], "price_or_rent": [10.0]})}
        ),
        households=SimpleNamespace(
            current_net_housing_cash_need=lambda: np.array([cash]), current_housing_lfa_draw=lambda: np.array([2.0])
        ),
    )


def test_replay_copies_and_verifies_native_results():
    tape = HousingReplayTape(60)
    baseline = country()
    treated = country()
    tape.capture_proposals(1, 60, baseline)
    tape.record_or_verify(1, 60, baseline, treatment=False)
    baseline.housing_market.states["current_sales"].iloc[0, 0] = 99
    tape.replay_proposals(1, 60, treated)
    assert treated.housing_market.states["current_sales"].iloc[0, 0] == 1
    tape.record_or_verify(1, 60, treated, treatment=True)
    assert tape.audit[0]["matched"]


def test_rejects_cash_mismatch_instead_of_overwriting():
    tape = HousingReplayTape(60, require_financing_match=True)
    tape.record_or_verify(1, 60, country(), treatment=False)
    treated = country(9.0)
    with pytest.raises(HousingReplayMismatch, match="Housing cash differs"):
        tape.record_or_verify(1, 60, treated, treatment=True)
    assert treated.households.current_net_housing_cash_need()[0] == 9.0


def test_rejects_changed_completed_transaction():
    tape = HousingReplayTape(60)
    tape.record_or_verify(1, 60, country(), treatment=False)
    treated = country()
    treated.housing_market.states["current_sales"].iloc[0, 1] = 12.0
    with pytest.raises(HousingReplayMismatch, match="transactions differ"):
        tape.record_or_verify(1, 60, treated, treatment=True)


def test_instrumentation_restores_methods_after_failure():
    from src.experiments import mpc
    from src.experiments.mpc_housing_replay import housing_replay

    from macromodel.country.country import Country

    before = (mpc._run_one, Country.clear_housing_market, Country.process_housing_market_clearing)
    with pytest.raises(ValueError, match="stop"):
        with housing_replay(HousingReplayTape(60)):
            raise ValueError("stop")
    assert before == (mpc._run_one, Country.clear_housing_market, Country.process_housing_market_clearing)


def test_transaction_control_allows_endogenous_financing():
    tape = HousingReplayTape(60)
    tape.record_or_verify(1, 60, country(), treatment=False)
    tape.record_or_verify(1, 60, country(9.0), treatment=True)
    assert tape.audit[0]["matched"]
    assert not tape.audit[0]["financing_matched"]


def test_replays_completed_trades_instead_of_rejected_proposals():
    tape = HousingReplayTape(60)
    baseline = country()
    treated = country()
    baseline.housing_market.states["current_sales"] = pd.DataFrame({"buyer_id": [1, 2], "price_or_rent": [10.0, 20.0]})
    tape.capture_proposals(1, 60, baseline)
    baseline.housing_market.states["current_sales"] = baseline.housing_market.states["current_sales"].iloc[:1].copy()
    tape.record_or_verify(1, 60, baseline, treatment=False)
    tape.replay_proposals(1, 60, treated)
    assert len(treated.housing_market.states["current_sales"]) == 1
