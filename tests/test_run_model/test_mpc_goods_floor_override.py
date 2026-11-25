import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "run_model"))
import src.experiments.mpc as mpc  # noqa: E402


class _Stop(Exception):
    pass


@pytest.mark.parametrize("override, expected", [(False, False), (True, True), (None, True)])
def test_goods_floor_override_applies_to_experiment_config_only(monkeypatch, override, expected):
    cfg = SimpleNamespace(household_support={"mode": "minimum_income", "goods_floor": True})
    data = SimpleNamespace(configuration=SimpleNamespace(year=2014))
    monkeypatch.setattr(mpc, "_prepare_mpc_inputs", lambda **_: (None, data, {"FRA": cfg}, "FRA", Path(".")))

    def stop(*_args, **_kwargs):
        raise _Stop

    monkeypatch.setattr(mpc.DataPaths, "default_paths", stop)
    with pytest.raises(_Stop):
        mpc.run_mpc_experiment(
            seeds=[1],
            t_max=5,
            shock_period=2,
            horizon_periods=1,
            shock_fraction=0.1,
            output_dir=Path("unused"),
            goods_floor=override,
        )
    assert cfg.household_support["goods_floor"] is expected
    assert cfg.household_support["mode"] == "minimum_income"
