# Imports below CLI path setup select the requested repository.
# ruff: noqa: E402
"""Capture an unchanged two-period FRA baseline and exact GDP call inputs."""

import argparse
import hashlib
import inspect
import json
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = args.repo.resolve()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "run_model"))
from src.helpers import align_country_configuration_to_data

from macro_data import DataWrapper
from macro_data.readers.default_readers import DataPaths
from macromodel.configurations import SimulationConfiguration, load_country_configuration
from macromodel.economy.economy import Economy
from macromodel.simulation import Simulation


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


cache = root / "run_model/data/output_data/data_benchmark.pkl"
config_path = root / "run_model/config/country_config_FRA.yaml"
data = DataWrapper.init_from_pickle(str(cache))
syn = data.synthetic_countries["FRA"]
manifest = {
    "model_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    "seed": 15,
    "t_max": 2,
    "python": platform.python_version(),
    "cache": str(cache),
    "cache_sha256": sha(cache),
    "config_sha256": sha(config_path),
    "synthetic_gdp": {k: float(getattr(syn, k)) for k in ["gdp_output", "gdp_expenditure", "gdp_income"]},
    "synthetic_rent_imputed": float(syn.population.household_data["Rent Imputed"].sum()),
}
cfg = load_country_configuration(config_path, country_iso3="FRA")
cfg = align_country_configuration_to_data(cfg, n_industries=data.n_industries, n_firms=len(syn.firms.firm_data))
configuration = SimulationConfiguration(country_configurations={"FRA": cfg}, t_max=2, seed=15)
(out / "resolved_configuration.json").write_text(configuration.model_dump_json(indent=2))
random.seed(15)
np.random.seed(15)
paths = DataPaths.default_paths(root / "run_model/data/raw_data", [data.configuration.year])
model = Simulation.from_datawrapper(datawrapper=data, simulation_configuration=configuration, data_paths=paths)
country = model.countries["FRA"]
fields = [
    "gdp_output",
    "gdp_expenditure",
    "gdp_income",
    "gdp_expenditure_prebalancing_residual",
    "total_household_fce",
    "total_real_rent_paid",
    "total_imp_rent_paid",
]
manifest["time_unit"] = country.economy.time_unit
manifest["scale"] = country.scale
manifest["n_households"] = int(country.households.ts.current("n_households"))
manifest["initial_economy"] = {k: country.economy.ts.current(k) for k in fields}
manifest["initial_household_sums"] = {
    k: float(np.asarray(country.households.ts.current(k)).sum())
    for k in ["consumption", "total_consumption", "rent", "rent_imputed", "cacf_real_consumption_budget"]
}
original = Economy.compute_gdp
signature = inspect.signature(original)
calls = []


def capture(self, *args, **kwargs):
    bound = signature.bind(self, *args, **kwargs)
    bound.apply_defaults()
    inputs = {k: v for k, v in bound.arguments.items() if k != "self"}
    original(self, *args, **kwargs)
    results = {
        k: np.asarray(self.ts.current(k)).copy()
        for k in fields
        if k not in ["total_real_rent_paid", "total_imp_rent_paid"]
    }
    assert all(np.isfinite(v).all() for v in results.values())
    calls.append({"inputs": inputs, "outputs": results})


Economy.compute_gdp = capture
try:
    model.run()
finally:
    Economy.compute_gdp = original
model.save(save_dir=out, file_name="baseline.h5")
manifest["hdf5_sha256"] = sha(out / "baseline.h5")
manifest["calls"] = calls
manifest["source_hashes"] = {
    str(p.relative_to(root)): sha(p)
    for p in [
        root / "macromodel/economy/economy.py",
        root / "macromodel/economy/economy_ts.py",
        root / "macromodel/country/country.py",
        root / "macromodel/agents/households/households.py",
        root / "macromodel/agents/households/func/financial_feasibility.py",
    ]
}
(out / "manifest.json").write_text(json.dumps(manifest, default=serial, indent=2) + "\n")
(out / "packages.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
print(json.dumps({k: v for k, v in manifest.items() if k not in ["source_hashes", "calls"]}, default=serial, indent=2))
print("CAPTURE_COMPLETE", len(calls))
