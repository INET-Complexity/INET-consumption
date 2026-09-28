"""Pre-change GDP fixtures captured before the atomic housing migration.

Increment 3 replays these frozen inputs: all GDP approaches gain H, FCE gains
R+H, and raw residuals and trade balancing stay unchanged.
"""

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE_PATH = (
    Path(__file__).resolve().parents[5] / "docs/implementation/total-consumption-increment0/gdp_prechange_fixtures.json"
)
CASES = json.loads(FIXTURE_PATH.read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_national_accounts_migration_against_prechange_fixture(test_economy, case):
    inputs = dict(case["inputs"])
    for name in ("sectoral_sales", "sectoral_intermediate_consumption"):
        inputs[name] = np.asarray(inputs[name], dtype=float)
    test_economy.compute_gdp(**inputs)
    for name, expected in case["outputs"].items():
        expected = np.asarray(expected, dtype=float)
        if name in ("gdp_output", "gdp_income", "gdp_expenditure"):
            expected = expected + inputs["rent_imputed"]
        elif name == "total_household_fce":
            expected = expected + inputs["rent_paid"] + inputs["rent_imputed"]
        np.testing.assert_allclose(
            test_economy.ts.current(name),
            expected,
            rtol=1e-12,
            atol=1e-3,
            err_msg=f"{case['name']}: {name}",
        )
