"""Selected serialization must preserve values and exclude unrelated datasets."""

import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "run_model"))
from src.experiments.selected_output import select_names, write_selected

from macromodel.timeseries import TimeSeries


def test_selected_serialization_preserves_multidimensional_data(tmp_path):
    ts = TimeSeries(target=np.arange(6).reshape(2, 3), unused=np.ones(100))
    ts.target.append(np.arange(6, 12).reshape(2, 3))
    model = SimpleNamespace(countries={"FRA": SimpleNamespace(households=SimpleNamespace(ts=ts))})
    path = tmp_path / "selected.h5"
    write_selected(model, path, "FRA", ["FRA/households/target"])
    with h5py.File(path) as handle:
        assert set(handle["FRA/households"]) == {"target", "target_columns"}
        np.testing.assert_array_equal(handle["FRA/households/target"], np.arange(12).reshape(2, 6))
    assert len(ts.target) == 2
    assert "unused" in ts.dicts


@pytest.mark.parametrize("names", [[], ["bad"], ["cash", "cash"], "cash"])
def test_invalid_selection(names):
    with pytest.raises(ValueError):
        select_names(names, ["cash"])
