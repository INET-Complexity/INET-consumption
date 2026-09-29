"""Verify fixed-input bridges separately from endogenous smoke differences."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidate", type=Path, required=True)
parser.add_argument("--baseline", type=Path, default=Path("/private/tmp/inet-increment2-smoke"))
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
manifest = json.loads((args.candidate / "manifest.json").read_text())
base_manifest = json.loads((args.baseline / "manifest.json").read_text())
report = {"candidate": str(args.candidate), "baseline": str(args.baseline), "t_max": manifest["t_max"]}


def check(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-3)


# No identity claim depends on the model's trade-balancing plug.
bridges = []
for call in manifest["calls"]:
    i, o = call["inputs"], call["outputs"]
    h = i["rent_imputed"]
    old_output = (
        i["total_output"]
        - sum(i["sectoral_intermediate_consumption"])
        - i["taxes_on_production"]
        + i["taxes_on_products"]
        + i["rent_paid"]
    )
    old_income = (
        i["operating_surplus"]
        + i["wages"]
        + i["taxes_on_products"]
        + i["rent_received"]
        + i["central_government_rent_received"]
    )
    old_raw_expenditure = (
        i["change_in_inventories"]
        + i["gross_fixed_capital_formation"]
        + i["hh_consumption"]
        + i["gov_consumption"]
        + i["exports"]
        - i["imports"]
        + i["rent_paid"]
    )
    residual = old_raw_expenditure - old_output
    check(o["gdp_output"], [old_output + h])
    check(o["gdp_income"], [old_income + h])
    expected_expenditure = old_output if i["always_adjust"] or i["running_multiple_countries"] else old_raw_expenditure
    check(o["gdp_expenditure"], [expected_expenditure + h])
    check(o["gdp_expenditure_prebalancing_residual"], [residual])
    check(o["total_household_fce"], [i["hh_consumption"] + i["rent_paid"] + h])
    for name in (
        "owner_occupied_housing_output",
        "owner_occupied_housing_value_added",
        "owner_occupied_housing_operating_income",
    ):
        check(o[name], [h])
    bridges.append(
        {
            "gdp_output": o["gdp_output"][0],
            "housing": h,
            "raw_expenditure_residual": residual,
            "raw_income_residual": old_income - old_output,
        }
    )
report["fixed_input_bridges"] = bridges

with h5py.File(args.candidate / "baseline.h5") as candidate, h5py.File(args.baseline / "baseline.h5") as baseline:
    hh = candidate["FRA/households"]
    ec = candidate["FRA/economy"]
    cash, total = hh["consumption_cash_expenditure"][:], hh["consumption_including_housing"][:]
    h, rent = hh["rent_imputed"][:], hh["rent"][:]
    vat_factor = hh["total_consumption"][:] / hh["total_consumption_before_vat"][:]
    assert np.isfinite(cash).all() and np.isfinite(total).all() and np.isfinite(h).all()
    check(cash, hh["consumption"][:] * vat_factor + rent)
    check(total, cash + h)
    check(total.sum(axis=1), ec["total_household_fce"][:, 0])
    check(hh["consumption"][0].sum(), hh["total_consumption_before_vat"][0, 0])
    # The active fixed-basket CPI has initial level one.
    check(hh["cacf_real_consumption_budget"][0] * ec["cpi_fixed_basket"][0, 0], total[0])
    report["household_rows_checked"] = int(total.size)
    report["identity_failures"] = 0
    for name in ("gdp_output", "gdp_expenditure", "gdp_income"):
        assert np.isfinite(ec[name][:]).all()
        check(ec[name + "_growth"][1:, 0], ec[name][1:, 0] / ec[name][:-1, 0] - 1)
        check(ec[name][0], baseline["FRA/economy/" + name][0] + h[0].sum())
    check(
        ec["total_household_fce_growth"][1:, 0],
        ec["total_household_fce"][1:, 0] / ec["total_household_fce"][:-1, 0] - 1,
    )
    unchanged_initial = [
        "households/liquid_financial_assets",
        "households/income",
        "households/income_rental",
        "households/debt",
        "households/investment",
        "households/rent",
        "households/rent_imputed",
        "households/target_consumption",
        "central_government/taxes_vat",
        "central_government/taxes_on_products",
    ]
    for key in unchanged_initial:
        path = "FRA/" + key
        if path not in candidate:
            raise AssertionError("Missing invariant dataset: " + path)
        np.testing.assert_array_equal(candidate[path][0], baseline[path][0])
    report["unchanged_initial_datasets"] = unchanged_initial
    dynamic = {}
    fields = [
        "economy/gdp_output",
        "economy/gdp_expenditure",
        "economy/gdp_income",
        "economy/gdp_expenditure_prebalancing_residual",
        "economy/total_household_fce",
        "economy/cpi_fixed_basket",
        "economy/ppi",
        "households/target_consumption",
        "households/total_consumption",
        "households/consumption_cash_expenditure",
        "households/consumption_including_housing",
        "households/total_received_consumption_loans",
        "households/total_consumption_loan_debt",
        "households/total_liquid_financial_assets",
        "households/formula_implied_mpc",
        "households/target_consumption_total_mpc",
    ]
    for key in fields:
        a, b = candidate["FRA/" + key][:], baseline["FRA/" + key][:]
        # MPC diagnostics are household means; quantities are household sums.
        reducer = np.mean if "mpc" in key else np.sum
        a = reducer(a.reshape(len(a), -1), axis=1)
        b = reducer(b.reshape(len(b), -1), axis=1)
        n = min(len(a), len(b))
        dynamic[key] = {"baseline": b[:n].tolist(), "candidate": a[:n].tolist(), "difference": (a[:n] - b[:n]).tolist()}
    report["dynamic_comparison"] = dynamic

assert manifest["cache_sha256"] == base_manifest["cache_sha256"]
assert manifest["config_sha256"] == base_manifest["config_sha256"]
assert manifest["synthetic_gdp"] == base_manifest["synthetic_gdp"]
report["initial_goods_alignment_factor"] = manifest["initial_goods_alignment_factor"]
report["initial_economy"] = manifest["initial_economy"]
report["verdict"] = "PASS: Increment 3 accounting and initialization contracts; no macro-validity claim"
args.output.parent.mkdir(parents=True, exist_ok=True)


# JSON null explicitly represents unavailable initial formula diagnostics.
def json_finite(value):
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_finite(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


args.output.write_text(json.dumps(json_finite(report), indent=2, allow_nan=False) + "\n")
print(report["verdict"])
print("Household rows:", report["household_rows_checked"])
