import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

RUN_MODEL_PATH = Path(__file__).resolve().parents[2] / "run_model"
if str(RUN_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL_PATH))

from src.diagnostics.household_investment_counterfactual import _period_total  # noqa: E402


def _write_series(path: Path, values: np.ndarray) -> h5py.File:
    h5_file = h5py.File(path, "w")
    h5_file.create_dataset("FRA/households/series", data=values)
    return h5_file


def test__period_total_returns_finite_household_panel_total(tmp_path: Path) -> None:
    h5_file = _write_series(tmp_path / "series.h5", np.array([[1.0, 2.0], [3.0, 4.0]]))
    try:
        assert _period_total(h5_file, "FRA", "households/series", 1) == 7.0
    finally:
        h5_file.close()


def test__period_total_rejects_missing_series_and_invalid_period(tmp_path: Path) -> None:
    h5_file = _write_series(tmp_path / "series.h5", np.array([[1.0, 2.0]]))
    try:
        with pytest.raises(KeyError, match="Missing HDF5 series"):
            _period_total(h5_file, "FRA", "households/missing", 0)
        with pytest.raises(IndexError, match="Period 1 is unavailable"):
            _period_total(h5_file, "FRA", "households/series", 1)
    finally:
        h5_file.close()


def test__period_total_rejects_non_finite_values(tmp_path: Path) -> None:
    h5_file = _write_series(tmp_path / "series.h5", np.array([[1.0, np.nan]]))
    try:
        with pytest.raises(ValueError, match="Non-finite values"):
            _period_total(h5_file, "FRA", "households/series", 0)
    finally:
        h5_file.close()
