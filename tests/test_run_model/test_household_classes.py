import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "run_model"))
from src.household_classes import classify_households  # noqa: E402


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


def test_paper_eight_groups_median_split_and_three_group_collapse():
    from src.household_classes import classify_balance_sheet_groups

    # income 100; ratios: NLA = (LFA - consumer - mortgage)/y, IFA/y, HA/y
    panel = _panel(
        [
            [100, 0, 50, 0, 0, 0],  # NLA -0.5, IFA 0, HA 0 -> LLL
            [100, 0, 0, 0, 900, 400],  # NLA -4 (mortgage counted), HA 9 -> LLH
            [100, 500, 0, 300, 0, 0],  # NLA 5, IFA 3 -> HHL
            [100, 500, 0, 300, 900, 0],  # HHH
            [0, 1, 0, 0, 0, 0],  # non-positive income -> unclassified
        ]
    )
    out = classify_balance_sheet_groups(panel)
    assert list(out.group_code[:4]) == ["LLL", "LLH", "HHL", "HHH"]
    assert out.group_label[1] == "Housing-rich hand-to-mouth"
    assert list(out.group_3[:4]) == [
        "Poor hand-to-mouth",
        "Wealthy hand-to-mouth",
        "Doubly wealthy",
        "Doubly wealthy",
    ]
    assert pd.isna(out.group_code[4]) and pd.isna(out.group_3[4])


def test_median_split_is_within_seed():
    from src.household_classes import classify_balance_sheet_groups

    panel = pd.concat([_panel([[100, 10, 0, 0, 0, 0], [100, 20, 0, 0, 0, 0]])] * 2, ignore_index=True)
    panel.loc[2:, "seed"] = 2
    panel.loc[2:, "liquid_financial_assets"] *= 100  # seed 2 richer: split still within seed
    out = classify_balance_sheet_groups(panel)
    assert list(out.group_code.str[0]) == ["L", "H", "L", "H"]
