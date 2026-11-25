import numpy as np
import pandas as pd

from run_model.src.household_classes import classify_households


def _panel(rows):
    cols = [
        "income",
        "liquid_financial_assets",
        "consumption_loan_debt",
        "illiquid_financial_assets",
        "housing_wealth",
        "mortgage_debt",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["seed"] = 1
    return df


def test_kvw_kinks_and_illiquid_split():
    # quarterly income 600 -> monthly pay 200 -> tau = 100; credit limit (if credit) = 200
    panel = _panel(
        [
            [600, 50, 0, 0, 0, 0],  # 0 <= m <= tau, a = 0 -> P-HtM
            [600, 100, 0, 0, 500, 100],  # m = tau exactly, a > 0 -> R-HtM
            [600, 0, 150, 0, 0, 0],  # m = -150 <= -q + tau = -100 -> P-HtM
            [600, 0, 50, 0, 0, 0],  # m = -50 > -100 -> not HtM
            [600, 0, 50, 0, 0, 0],
            [600, 101, 0, 0, 0, 0],  # just above tau -> not HtM
            [600, 5000, 0, 0, 9000, 0],  # top m and a -> DW
            [600, 4000, 0, 0, 8000, 0],  # top two quintiles -> DW
            [600, 3000, 0, 0, 0, 0],
            [600, 200, 0, 0, 0, 0],
        ]
    )
    out = classify_households(panel)
    assert list(out[:3]) == ["P-HtM", "R-HtM", "P-HtM"]
    assert out[3] == "Other" and out[5] == "Other"
    assert out[6] == "DW" and out[7] == "DW"
    assert set(out) <= {"P-HtM", "R-HtM", "DW", "Other"}


def test_negative_income_and_missing_values():
    panel = _panel([[-10, 0, 0, 0, 0, 0], [-10, 5, 0, 0, 0, 0], [np.nan, 1, 0, 0, 0, 0]])
    out = classify_households(panel)
    assert out[0] == "P-HtM"  # tau = 0, m = 0 is the zero-liquidity kink
    assert out[1] != "P-HtM" and out[1] != "R-HtM"
    assert pd.isna(out[2])
