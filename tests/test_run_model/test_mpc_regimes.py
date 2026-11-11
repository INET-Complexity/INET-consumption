import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

RUN_MODEL_PATH = Path(__file__).resolve().parents[2] / "run_model"
if str(RUN_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL_PATH))

from src.mpc_analysis import (  # noqa: E402
    MPC_REGIME_CONSTRAINED,
    MPC_REGIME_FLOOR_SELF_FUNDED,
    MPC_REGIME_INPUTS,
    MPC_REGIME_SUPPORTED,
    MPC_REGIME_UNCONSTRAINED,
    MPC_REGIME_UNKNOWN,
    classify_mpc_regime,
    labor_income_gap_share,
    read_household_regime_inputs,
    summarize_mpc_regimes,
)


def test_classify_mpc_regime_precedence_and_unknown():
    # Floor F = 100 for every household.
    # 0: support 30, consumes the floor            -> supported
    # 1: desired 60 < F, consumes F, no support    -> floor-bound self-funded
    # 2: desired 180, cut by 40                    -> constrained
    # 3: desired 150 met, 5 of debt service unpaid -> constrained
    # 4: desired 150 met, nothing unpaid           -> unconstrained
    # 5: supported and with unpaid service         -> supported (precedence)
    # 6: non-finite input                          -> unknown
    regime = classify_mpc_regime(
        support=np.array([30.0, 0.0, 0.0, 0.0, 0.0, 10.0, np.nan]),
        consumption_before_floor=np.array([80.0, 60.0, 180.0, 150.0, 150.0, 120.0, 150.0]),
        consumption_after_floor=np.array([100.0, 100.0, 140.0, 150.0, 150.0, 100.0, 150.0]),
        consumption_cut=np.array([0.0, 0.0, 40.0, 0.0, 0.0, 20.0, 0.0]),
        unpaid_service=np.array([0.0, 0.0, 0.0, 5.0, 0.0, 7.0, 0.0]),
        subsistence_floor=np.full(7, 100.0),
    )
    assert regime.tolist() == [
        MPC_REGIME_SUPPORTED,
        MPC_REGIME_FLOOR_SELF_FUNDED,
        MPC_REGIME_CONSTRAINED,
        MPC_REGIME_CONSTRAINED,
        MPC_REGIME_UNCONSTRAINED,
        MPC_REGIME_SUPPORTED,
        MPC_REGIME_UNKNOWN,
    ]


def test_classify_mpc_regime_ignores_rounding_noise_relative_to_the_floor():
    regime = classify_mpc_regime(
        support=np.array([1e-6]),
        consumption_before_floor=np.array([2e7]),
        consumption_after_floor=np.array([2e7]),
        consumption_cut=np.array([1e-6]),
        unpaid_service=np.array([0.0]),
        subsistence_floor=np.array([1e7]),
    )
    assert regime.tolist() == [MPC_REGIME_UNCONSTRAINED]


def test_classify_mpc_regime_rejects_unequal_shapes():
    with pytest.raises(ValueError, match="one value per household"):
        classify_mpc_regime(
            support=np.zeros(2),
            consumption_before_floor=np.zeros(3),
            consumption_after_floor=np.zeros(2),
            consumption_cut=np.zeros(2),
            unpaid_service=np.zeros(2),
            subsistence_floor=np.ones(2),
        )


def test_read_household_regime_inputs_reads_one_row_or_reports_absence(tmp_path):
    full, legacy = tmp_path / "full.h5", tmp_path / "legacy.h5"
    with h5py.File(full, "w") as handle:
        households = handle.create_group("FRA/households")
        for offset, name in enumerate(MPC_REGIME_INPUTS):
            households.create_dataset(name, data=np.arange(6, dtype=float).reshape(3, 2) + 10 * offset)
        handle.create_dataset("FRA/economy/subsistence_consumption", data=np.full((3, 2), 7.0))
    with h5py.File(legacy, "w") as handle:
        handle.create_group("FRA/households").create_dataset("consumption", data=np.zeros((3, 2)))

    inputs = read_household_regime_inputs(full, "FRA", row=1)
    assert inputs["stage5_subsistence_support"].tolist() == [2.0, 3.0]
    assert inputs["unpaid_mortgage_payment"].tolist() == [52.0, 53.0]
    assert inputs["subsistence_floor"].tolist() == [7.0, 7.0]
    assert read_household_regime_inputs(legacy, "FRA", row=1) is None


def _labour_panel():
    return pd.DataFrame(
        {
            "shock_amount": [100.0, 100.0, 100.0],
            # Household 0: identical in both runs, although its wage moves over time.
            # Household 1: runs differ by 2 and 3 -> gap share 0.05.
            # Household 2: runs differ by 30 -> gap share 0.30.
            "labor_income_t0": [500.0, 500.0, 500.0],
            "labor_income_t1": [520.0, 500.0, 500.0],
            "labor_income_t2": [540.0, 500.0, 500.0],
            "labor_income_shock_t0": [500.0, 500.0, 500.0],
            "labor_income_shock_t1": [520.0, 502.0, 530.0],
            "labor_income_shock_t2": [540.0, 497.0, 500.0],
            "labor_income_t1_bin": [1, 1, 1],
        }
    )


def test_labor_income_gap_share_compares_runs_within_the_same_period():
    np.testing.assert_allclose(labor_income_gap_share(_labour_panel()), [0.0, 0.05, 0.30])


def test_labor_income_gap_share_requires_both_runs():
    with pytest.raises(ValueError, match="labor_income_shock_t2"):
        labor_income_gap_share(_labour_panel().drop(columns="labor_income_shock_t2"))
    with pytest.raises(ValueError, match="no labor_income_t"):
        labor_income_gap_share(pd.DataFrame({"shock_amount": [1.0]}))


def test_summarize_mpc_regimes_shares_means_and_identified_cumulative_sample():
    panel = pd.DataFrame(
        {
            "income_bin": ["Q1"] * 4 + ["Q2"] * 2,
            "mpc_regime": [
                MPC_REGIME_SUPPORTED,
                MPC_REGIME_SUPPORTED,
                MPC_REGIME_CONSTRAINED,
                MPC_REGIME_UNCONSTRAINED,
                MPC_REGIME_UNCONSTRAINED,
                MPC_REGIME_UNCONSTRAINED,
            ],
            "mpc_impact": [0.0, 0.01, 0.75, 0.40, 0.30, 0.50],
            "real_cmpc_4q": [0.0, 0.1, 0.8, 0.5, 0.2, 0.6],
            "cumulative_mpc_identified": [True, False, True, True, False, True],
        }
    )
    summary = summarize_mpc_regimes(panel).set_index(["income_bin", "regime"])

    q1_all = summary.loc[("Q1", "all")]
    assert q1_all["n"] == 4 and q1_all["share"] == 1.0
    assert q1_all["impact_mean"] == pytest.approx((0.0 + 0.01 + 0.75 + 0.40) / 4)
    assert q1_all["impact_share_zero"] == pytest.approx(0.5)
    assert q1_all["n_cumulative_identified"] == 3
    assert q1_all["cumulative_mean"] == pytest.approx((0.0 + 0.8 + 0.5) / 3)

    supported = summary.loc[("Q1", MPC_REGIME_SUPPORTED)]
    assert supported["share"] == pytest.approx(0.5)
    assert supported["impact_median"] == pytest.approx(0.005)
    assert supported["n_cumulative_identified"] == 1 and supported["cumulative_mean"] == pytest.approx(0.0)

    q2 = summary.loc[("Q2", MPC_REGIME_UNCONSTRAINED)]
    assert q2["share"] == 1.0 and q2["cumulative_median"] == pytest.approx(0.6)
    assert ("Q2", MPC_REGIME_SUPPORTED) not in summary.index
    # Regime shares add to one within each group.
    shares = summary.drop(index="all", level="regime").groupby(level="income_bin")["share"].sum()
    np.testing.assert_allclose(shares, 1.0)


def test_summarize_mpc_regimes_uses_everyone_without_an_identification_column():
    panel = pd.DataFrame(
        {
            "income_bin": ["Q1", "Q1"],
            "mpc_regime": [MPC_REGIME_UNCONSTRAINED] * 2,
            "mpc_impact": [0.2, 0.4],
            "real_cmpc_4q": [0.3, 0.5],
        }
    )
    row = summarize_mpc_regimes(panel).set_index("regime").loc["all"]
    assert row["n_cumulative_identified"] == 2 and row["cumulative_mean"] == pytest.approx(0.4)
    with pytest.raises(ValueError, match="mpc_regime"):
        summarize_mpc_regimes(panel.drop(columns="mpc_regime"))
