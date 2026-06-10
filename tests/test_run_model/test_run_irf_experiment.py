import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import pandas as pd
import pytest

RUN_MODEL_PATH = Path(__file__).resolve().parents[2] / "run_model"
if str(RUN_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL_PATH))

import run_irf_experiment as irf_runner  # noqa: E402


def _write_irf_smoke_h5(path: Path, *, shock_kind: str | None = None) -> None:
    values_by_path = {
        "FRA/households/consumption": [[100.0], [100.0], [100.0], [100.0]],
        "FRA/households/target_consumption": [[110.0], [110.0], [110.0], [110.0]],
        "FRA/households/income": [[90.0], [90.0], [90.0], [90.0]],
        "FRA/government_entities/total_consumption": [[20.0], [20.0], [20.0], [20.0]],
        "FRA/government_entities/desired_consumption_in_lcu": [[25.0], [25.0], [25.0], [25.0]],
        "FRA/central_Bank/policy_rate": [[0.02], [0.02], [0.02], [0.02]],
        "FRA/economy/cpi_fixed_basket": [[1.0], [1.0], [1.0], [1.0]],
        "FRA/economy/gdp_output": [[200.0], [200.0], [200.0], [200.0]],
        "FRA/economy/gdp_expenditure": [[200.0], [200.0], [200.0], [200.0]],
        "FRA/economy/gdp_income": [[200.0], [200.0], [200.0], [200.0]],
        "FRA/economy/unemployment_rate": [[0.05], [0.05], [0.05], [0.05]],
    }
    if shock_kind == "government_consumption":
        values_by_path["FRA/government_entities/total_consumption"][1][0] += 5.0
        values_by_path["FRA/economy/gdp_expenditure"][1][0] += 5.0
    elif shock_kind == "income_tax":
        values_by_path["FRA/households/income"][1][0] -= 3.0
        values_by_path["FRA/households/consumption"][1][0] -= 1.0
    elif shock_kind == "policy_rate":
        values_by_path["FRA/central_Bank/policy_rate"][1][0] += 0.01
        values_by_path["FRA/households/consumption"][1][0] -= 2.0
    elif shock_kind == "unemployment_rate":
        values_by_path["FRA/economy/unemployment_rate"][1][0] += 0.05
        values_by_path["FRA/households/income"][1][0] -= 4.0

    with h5py.File(path, "w") as handle:
        for dataset_path, values in values_by_path.items():
            handle.create_dataset(dataset_path, data=values)


def test_run_irf_experiment_writes_nonzero_responses_for_supported_shocks(tmp_path, monkeypatch):
    data = SimpleNamespace()
    country_cfg = SimpleNamespace()
    shock_specs = (
        irf_runner.ShockSpec(name="gov", kind="government_consumption", period=0, magnitude=5.0),
        irf_runner.ShockSpec(name="tax", kind="income_tax", period=0, magnitude=0.01),
        irf_runner.ShockSpec(name="rate", kind="policy_rate", period=0, magnitude=0.01),
        irf_runner.ShockSpec(name="unemp", kind="unemployment_rate", period=0, magnitude=0.05),
    )
    run_calls = []

    monkeypatch.setattr(
        irf_runner,
        "_prepare_irf_inputs",
        lambda **kwargs: (
            SimpleNamespace(),
            data,
            {"FRA": country_cfg},
            "FRA",
        ),
    )

    def fake_run_one(**kwargs):
        shock_spec = kwargs["shock_spec"]
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"seed-{kwargs['seed']}" / "multi_country_simulation.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_irf_smoke_h5(path, shock_kind=None if shock_spec is None else shock_spec.kind)
        run_calls.append((kwargs["seed"], None if shock_spec is None else shock_spec.kind))
        return path

    monkeypatch.setattr(irf_runner, "_run_one", fake_run_one)

    def fake_write_irf_plots(summary, output_dir, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(irf_runner, "write_irf_plots", fake_write_irf_plots)

    paths = irf_runner.run_irf_experiment(
        seeds=[12],
        t_max=4,
        shock_specs=shock_specs,
        horizon_periods=2,
        output_dir=tmp_path,
        country_iso3="FRA",
        n_jobs=1,
    )

    panel = pd.read_csv(paths["analysis_dir"] / "irf_panel.csv")
    summary = pd.read_csv(paths["analysis_dir"] / "irf_summary.csv")
    run_index = pd.read_csv(paths["analysis_dir"] / "irf_run_index.csv")

    assert run_calls == [
        (12, None),
        (12, "government_consumption"),
        (12, "income_tax"),
        (12, "policy_rate"),
        (12, "unemployment_rate"),
    ]
    assert run_index["shock_name"].tolist() == ["gov", "tax", "rate", "unemp"]
    assert panel.loc[
        (panel["shock_name"] == "gov") & (panel["variable"] == "government_consumption") & (panel["horizon"] == 0),
        "delta",
    ].iat[0] == 5.0
    assert panel.loc[
        (panel["shock_name"] == "tax") & (panel["variable"] == "household_income") & (panel["horizon"] == 0),
        "delta",
    ].iat[0] == -3.0
    assert panel.loc[
        (panel["shock_name"] == "rate") & (panel["variable"] == "policy_rate") & (panel["horizon"] == 0),
        "delta",
    ].iat[0] == pytest.approx(0.01)
    assert panel.loc[
        (panel["shock_name"] == "unemp") & (panel["variable"] == "unemployment_rate") & (panel["horizon"] == 0),
        "delta",
    ].iat[0] == 0.05
    assert set(summary["shock_name"]) == {"gov", "tax", "rate", "unemp"}
