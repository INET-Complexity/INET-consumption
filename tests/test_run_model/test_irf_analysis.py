import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

RUN_MODEL_PATH = Path(__file__).resolve().parents[2] / "run_model"
if str(RUN_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL_PATH))

from src.irf_analysis import IRFVariable, build_irf_panel, summarize_irf_panel  # noqa: E402


def _write_h5(path: Path, *, values: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("FRA/households/consumption", data=values)


def test_build_irf_panel_computes_delta_and_percent_delta(tmp_path):
    baseline = tmp_path / "baseline.h5"
    shock = tmp_path / "shock.h5"
    _write_h5(baseline, values=np.array([[10.0, 20.0], [12.0, 22.0], [13.0, 23.0]]))
    _write_h5(shock, values=np.array([[10.0, 20.0], [15.0, 23.0], [15.0, 25.0]]))

    panel = build_irf_panel(
        baseline_h5=baseline,
        shock_h5=shock,
        seed=12,
        shock_name="test",
        shock_kind="policy_rate",
        shock_period=1,
        shock_magnitude=0.01,
        horizon_periods=2,
        country_code="FRA",
        variables=(IRFVariable("consumption", "{country}/households/consumption", "sum"),),
    )

    assert panel["horizon"].tolist() == [0, 1]
    assert panel["delta"].tolist() == pytest.approx([4.0, 4.0])
    assert panel["pct_delta"].tolist() == pytest.approx([4.0 / 34.0, 4.0 / 36.0])


def test_summarize_irf_panel_aggregates_across_seeds(tmp_path):
    baseline = tmp_path / "baseline.h5"
    shock = tmp_path / "shock.h5"
    _write_h5(baseline, values=np.array([[10.0], [10.0]]))
    _write_h5(shock, values=np.array([[10.0], [12.0]]))

    first = build_irf_panel(
        baseline_h5=baseline,
        shock_h5=shock,
        seed=1,
        shock_name="test",
        shock_kind="policy_rate",
        shock_period=1,
        shock_magnitude=0.01,
        horizon_periods=1,
        country_code="FRA",
        variables=(IRFVariable("consumption", "{country}/households/consumption", "sum"),),
    )
    second = first.assign(seed=2, delta=4.0, pct_delta=0.4)

    summary = summarize_irf_panel(pd.concat([first, second], ignore_index=True))

    assert summary.loc[0, "n"] == 2
    assert summary.loc[0, "delta_mean"] == pytest.approx(3.0)
