"""Run a frozen FRA benchmark with explicit phi=0.05 and unique outputs."""

import argparse
import ast
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[4]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--phase", choices=["smoke", "full"], required=True)
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--revision", default="dfde7397")
args = parser.parse_args()
ROOT = args.source_root.resolve()
os.environ["RAW_DATA_PATH"] = str(WORKSPACE / "run_model/data/raw_data")
os.environ["OUTPUT_DATA_PATH"] = str(args.output.resolve())
sys.path.insert(0, str(ROOT / "run_model"))
sys.path.insert(0, str(ROOT))

from src.notebook_workflow import NotebookRunConfig, build_country_config  # noqa: E402
from src.visual_helpers import build_macro_output_df  # noqa: E402

from macro_data import DataWrapper  # noqa: E402
from macro_data.readers.default_readers import DataPaths  # noqa: E402
from macromodel.configurations import SimulationConfiguration  # noqa: E402
from macromodel.simulation import Simulation  # noqa: E402

destination = args.output.resolve()
phase_dir = destination / args.phase
phase_dir.mkdir(parents=True, exist_ok=False)
snapshot = destination / "config"
source_cache = WORKSPACE / "run_model/data/output_data/data_benchmark.pkl"
cache = destination / "data_benchmark.pkl"
if not cache.exists():
    shutil.copy2(source_cache, cache)
raw = WORKSPACE / "run_model/data/raw_data"
revision = subprocess.check_output(["git", "rev-parse", args.revision], cwd=WORKSPACE, text=True).strip()
assert Path(sys.modules[Simulation.__module__].__file__).is_relative_to(ROOT)
if not snapshot.exists():
    snapshot.mkdir()
    for path in sorted((ROOT / "run_model/config").glob("*.yaml")):
        shutil.copy2(path, snapshot / path.name)
    manifest = {
        "git_commit": revision,
        "source_archive_root": str(ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "data_cache": str(cache),
        "source_data_cache": str(source_cache),
        "data_cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "yaml_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob("*.yaml")},
        "explicit_overrides": {"target_capital_inputs_fraction": 0.05},
        "seeds": [40, 41, 42, 43],
        "periods": 50,
        "workers": 1,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / "environment.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
manifest = json.loads((destination / "manifest.json").read_text())
assert hashlib.sha256(cache.read_bytes()).hexdigest() == manifest["data_cache_sha256"]
assert revision == manifest["git_commit"]
for name, digest in manifest["yaml_sha256"].items():
    assert hashlib.sha256((snapshot / name).read_bytes()).hexdigest() == digest
seeds = [40] if args.phase == "smoke" else manifest["seeds"]
periods = 2 if args.phase == "smoke" else manifest["periods"]
if args.phase == "full":
    assert json.loads((destination / "smoke/summary.json").read_text())["provenance_pass"]
records = []
started = time.monotonic()
for seed in seeds:
    assert time.monotonic() - started < 1800, "Thirty-minute compute budget exceeded"
    data = DataWrapper.init_from_pickle(str(cache))
    notebook_config = NotebookRunConfig(seed=seed, t_max=periods, country_iso3="FRA", config_dir=snapshot)
    countries = build_country_config(
        data,
        notebook_config,
        overrides={"firms.functions.target_capital_inputs.parameters['target_capital_inputs_fraction']": 0.05},
    )
    configuration = SimulationConfiguration(country_configurations=countries, seed=seed, t_max=periods)
    effective = configuration.model_dump()
    phi = effective["country_configurations"]["FRA"]["firms"]["functions"]["target_capital_inputs"]["parameters"][
        "target_capital_inputs_fraction"
    ]
    assert phi == 0.05
    (phase_dir / f"configuration_seed{seed}.json").write_text(json.dumps(effective, indent=2) + "\n")
    model = Simulation.from_datawrapper(
        datawrapper=data,
        simulation_configuration=configuration,
        data_paths=DataPaths.default_paths(raw, [data.configuration.year]),
    )
    model.run()
    frame = build_macro_output_df(model, country_code="FRA")
    for key in ["real_gdp", "gdp", "cpi_fixed_basket"]:
        assert np.isfinite(frame[key]).all() and (frame[key] > 0).all(), key
    exposure = np.asarray(model.countries["FRA"].firms.ts.historic("total_credit_exposure"))
    assert len(exposure) == periods + 1
    frame.to_pickle(phase_dir / f"macro_seed{seed}.pkl")
    if seed == 40:
        model.save(save_dir=phase_dir, file_name=f"FRA_phi005_seed{seed}_t{periods}.h5")
        with h5py.File(phase_dir / f"FRA_phi005_seed{seed}_t{periods}.h5") as h5:
            embedded = ast.literal_eval(h5.attrs["configuration"])
            assert embedded == effective
            assert h5["FRA/firms/total_credit_exposure"].shape[0] == periods + 1
    real = np.log(frame["real_gdp"].iloc[1:].to_numpy())
    trend = np.polyval(np.polyfit(np.arange(periods), real, 1), np.arange(periods))
    last = frame.iloc[-1]
    active = frame.iloc[1:]
    record = {
        "seed": seed,
        "phi": phi,
        "detrended_real_gdp_sd_pct": float(100 * np.std(real - trend)),
        "peak_unemployment_pct": float(100 * active["unemployment_rate"].max()),
        "bank_insolvency_max": float(active["bank_insolvency_rate"].max()),
        "mean_deficit_pct_gdp": float(100 * active["deficit_to_gdp"].mean()),
        "terminal_debt_pct_annual_gdp": float(100 * last["debt_to_gdp"]),
        "mean_cpi_yoy_pct": float(100 * frame["cpi_fixed_basket_yoy_change"].iloc[min(5, periods) :].mean()),
        "lt_credit_fulfillment_pct": float(
            100 * active["firm_credit_received_long_term"].sum() / active["firm_credit_demand_long_term"].sum()
        ),
        "terminal_real_gdp_ratio": float(last["real_gdp"] / frame["real_gdp"].iloc[0]),
    }
    records.append(record)
    pd.DataFrame(records).to_csv(phase_dir / "metrics.csv", index=False)
    print("VALIDATED", json.dumps(record), flush=True)
    del model, data, frame
    gc.collect()
summary = {
    "phase": args.phase,
    "provenance_pass": True,
    "stability_pass": bool(
        all(
            r["detrended_real_gdp_sd_pct"] < 3 and r["peak_unemployment_pct"] < 20 and r["bank_insolvency_max"] == 0
            for r in records
        )
        and np.mean([r["detrended_real_gdp_sd_pct"] for r in records]) < 2
    ),
    "elapsed_seconds": time.monotonic() - started,
    "metrics": records,
}
(phase_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("COMPLETE", json.dumps(summary), flush=True)
