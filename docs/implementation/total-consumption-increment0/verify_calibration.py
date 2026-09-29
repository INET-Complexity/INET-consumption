# Imports below CLI path setup select the requested repository.
# ruff: noqa: E402
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Verify persisted HFCS consumption against source components.")
parser.add_argument("--calibration-repo", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = args.calibration_repo.resolve()
sys.path.insert(0, str(root))
from src.hfcs_reader_consumption import HFCSReaderConsumption, build_consumption_components_from_hfcs

p = root / "data/hfcs/mpc_dataset_FRA.parquet"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


out = {"parquet_sha256": sha(p), "sources": {}, "waves": []}
persisted = pd.read_parquet(p)
cols = ["SA0010", "IM0100", "HW0010", "HI0220", "HB2300", "HB0410", "DA1110", "HB0900", "DA1130", "DA1131", "HC0110"]
for wave in [2014, 2017, 2021]:
    reader = HFCSReaderConsumption.from_csv(
        country_name="France",
        country_name_short="FR",
        year=wave,
        hfcs_data_path=root / "data/hfcs",
        num_surveys=5,
        subset_vars_dict={c: c for c in cols},
    )
    raw = reader.households_df.apply(pd.to_numeric, errors="coerce")
    # Same forced household median collapse and sorted SA0010 order as build_wave_df.
    hh = raw.groupby("SA0010", sort=True).median(numeric_only=True)
    rebuilt = build_consumption_components_from_hfcs(hh, sna_data_path=root / "data/hfcs/consumption_data_SNA_FRA.yaml")
    saved = persisted.loc[persisted.wave == wave].reset_index(drop=True)
    assert len(hh) == len(saved)
    # Validate row order independently using residence stocks and renter cash flows.
    np.testing.assert_allclose(hh.HB0900, saved.hmr_value, rtol=2e-6, atol=0.02, equal_nan=True)
    np.testing.assert_allclose(3 * hh.HB2300, saved.rent_quart, rtol=2e-6, atol=0.02, equal_nan=True)
    diff = np.abs(rebuilt.consumption.to_numpy() - saved.consumption.to_numpy())
    np.testing.assert_allclose(rebuilt.consumption, saved.consumption, rtol=2e-6, atol=0.02)
    real_error = np.abs(saved.real_consumption - saved.consumption / (saved.hfce_deflator / 100))
    np.testing.assert_allclose(
        saved.real_consumption, saved.consumption / (saved.hfce_deflator / 100), rtol=2e-6, atol=0.02
    )
    out["waves"].append(
        {
            "wave": wave,
            "rows": len(saved),
            "max_consumption_error": float(diff.max()),
            "max_deflation_error": float(real_error.max()),
            "annual_consumption_sum": float(saved.consumption.sum()),
            "annual_imputed_rent_sum": float(rebuilt.imputed_rental_HMR.sum()),
            "annual_actual_rent_sum": float(12 * hh[["HB2300", "HB0410"]].fillna(0).sum().sum()),
            "hfce_deflator": saved.hfce_deflator.unique().tolist(),
        }
    )
    for f in sorted((root / "data/hfcs" / str(wave)).glob("*.csv")):
        if f.stem.lower().rstrip("12345") in ["h", "d", "hn"]:
            out["sources"][str(f.relative_to(root))] = sha(f)
for rel in [
    "src/hfcs_reader_consumption.py",
    "src/SNA_reader.py",
    "src/portfolio_allocation_helpers.py",
    "src/consumption_calibration.py",
    "src/helpers.py",
    "consumption_parameters_calibration.ipynb",
    "show_distributions_and_MPCs.ipynb",
    "data/hfcs/consumption_data_SNA_FRA.yaml",
]:
    out["sources"][rel] = sha(root / rel)
args.output.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out["waves"], indent=2))
