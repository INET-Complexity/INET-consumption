import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "run_model"))
from src.mpc_paper import write_paper_summaries


def test_ratio_weights_and_seed_uncertainty(tmp_path):
    panel = pd.DataFrame(
        {"seed": [1, 1, 2], "household_id": [0, 1, 0], "shock_amount": [1.0, 1.0, 6.0], "mpc_impact": [0.0, 0.0, 1.0]}
    )
    write_paper_summaries(panel, tmp_path, sample="test")
    result = pd.read_csv(tmp_path / "paper_mpc_summary_test.csv").iloc[0]
    assert result.transfer_weighted_mpc == 0.75
    assert np.isclose(result.seed_cluster_se, 0.375)
    assert np.isclose(result["mean"], 1 / 3)


def test_cumulative_identification_and_real_weights(tmp_path):
    panel = pd.DataFrame(
        {
            "seed": [1, 2],
            "household_id": [0, 0],
            "shock_amount": [5.0, 5.0],
            "real_shock_amount": [1.0, 3.0],
            "real_mpc_impact": [-1.0, 1.0],
            "real_cmpc_4q": [-1.0, 1.0],
            "cumulative_mpc_identified": [True, False],
        }
    )
    write_paper_summaries(panel, tmp_path, sample="test")
    result = pd.read_csv(tmp_path / "paper_mpc_summary_test.csv").set_index("outcome")
    assert result.loc["real_mpc_impact", "transfer_weighted_mpc"] == 0.5
    assert result.loc["real_cmpc_4q", "n"] == 1
    assert result.loc["real_cmpc_4q", "transfer_weighted_mpc"] == -1


def test_channel_counts_overlap_and_ignore_missing(tmp_path):
    from src.mpc_paper import write_negative_mpc_diagnostics

    panel = pd.DataFrame(
        {
            "real_mpc_impact": [-1.0, -2.0, 1.0],
            "diagnostic_income_t1": [10.0, np.nan, 10.0],
            "diagnostic_income_shock_t1": [8.0, 12.0, 20.0],
        }
    )
    write_negative_mpc_diagnostics(panel, tmp_path, sample="test")
    r = pd.read_csv(tmp_path / "negative_mpc_channels_test.csv").iloc[0]
    assert r.negative_n == 2
    assert r.observed_n == 1
    assert r.lower_n == 1
    assert r.higher_n == 0
