"""Pre-change GDP fixtures captured before the atomic housing migration.

Increment 3 must update these expectations explicitly: all GDP approaches gain
H, FCE gains R+H, and raw residuals and trade balancing stay unchanged.
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
def test_prechange_gdp_fixture(test_economy, case):
    inputs = dict(case["inputs"])
    for name in ("sectoral_sales", "sectoral_intermediate_consumption"):
        inputs[name] = np.asarray(inputs[name], dtype=float)
    test_economy.compute_gdp(**inputs)
    for name, expected in case["outputs"].items():
        np.testing.assert_allclose(
            test_economy.ts.current(name),
            expected,
            rtol=1e-12,
            atol=1e-3,
            err_msg=f"{case['name']}: {name}",
        )
