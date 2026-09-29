"""Verify the registered output identities and Increment 1 compatibility smoke."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline", type=Path, required=True)
parser.add_argument("--candidate", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
with h5py.File(args.baseline, "r") as baseline, h5py.File(args.candidate, "r") as candidate:
    hh = candidate["FRA/households"]
    goods = hh["consumption"][:]
    factor = hh["total_consumption"][:] / hh["total_consumption_before_vat"][:]
    cash = hh["consumption_cash_expenditure"][:]
    total = hh["consumption_including_housing"][:]
    assert cash.shape == total.shape == goods.shape == (3, 5760)
    assert np.isfinite(cash).all() and np.isfinite(total).all()
    cash_failures = int(
        np.count_nonzero(~np.isclose(cash, goods * factor + np.maximum(hh["rent"][:], 0), rtol=1e-10, atol=1e-6))
    )
    total_failures = int(
        np.count_nonzero(~np.isclose(total, cash + np.maximum(hh["rent_imputed"][:], 0), rtol=1e-10, atol=1e-6))
    )
    checked = []
    differences = []

    def compare(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        old, new = obj[:], candidate[name][:]
        # The initial unevaluated total target now has the explicit NaN marker.
        if name == "FRA/households/target_consumption_calibrated_total":
            assert np.isnan(new[0]).all()
            old, new = old[1:], new[1:]
        matches = (
            np.allclose(old, new, rtol=1e-10, atol=1e-6, equal_nan=True)
            if np.issubdtype(old.dtype, np.number)
            else np.array_equal(old, new)
        )
        if old.shape != new.shape or not matches:
            differences.append(name)
        checked.append(name)

    baseline.visititems(compare)
    result = {
        "base_commit": "89c9e118",
        "seed": 15,
        "t_max": 2,
        "household_rows_including_initial": int(goods.size),
        "cash_identity_failures": cash_failures,
        "total_identity_failures": total_failures,
        "legacy_datasets_compared": len(checked),
        "legacy_differences": differences,
        "initial_target_marker": "NaN (previously zero placeholder)",
    }
for name, path in (("baseline", args.baseline), ("candidate", args.candidate)):
    with path.open("rb") as handle:
        result[f"{name}_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
assert cash_failures == total_failures == 0
assert not differences
