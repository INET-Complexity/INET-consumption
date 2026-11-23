import sys
from pathlib import Path

import h5py
import numpy as np

RUN_MODEL_PATH = Path(__file__).resolve().parents[2] / "run_model"
if str(RUN_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL_PATH))

from src.household_diagnostics import (  # noqa: E402
    build_household_diagnostics,
    display_label,
    household_diagnostic_summary,
    plot_household_diagnostics,
)

ROWS, N = 3, 4


def _run(path, block):
    with h5py.File(path, "w") as handle:
        households = handle.create_group("FRA/households")
        households.create_dataset("liquid_financial_assets", data=np.ones((ROWS, N)))
        households.create_dataset("income_minimum_income", data=np.arange(ROWS)[:, None] * np.full((ROWS, N), 1e9))
        households.create_dataset("unpaid_rent", data=np.full((ROWS, N), 2e9))
        households.create_dataset("unmet_floor", data=np.full((ROWS, N), 3e9))
        households.create_dataset("remaining_subsistence_shortfall", data=np.full((ROWS, N), 4e9))
        handle.attrs["configuration"] = "{'country_configurations': {'FRA': {'household_support': " + block + "}}}"
    return path


def test_diagnostics_report_top_up_unpaid_rent_and_unmet_floor_and_the_support_mode(tmp_path):
    frame, coverage = build_household_diagnostics(
        _run(tmp_path / "mi.h5", "{'mode': 'minimum_income'}"), country_code="FRA"
    )
    assert frame.attrs["support_mode"] == "minimum_income" and frame.attrs["goods_floor"] is True
    assert coverage.loc[["income_minimum_income", "unpaid_rent", "unmet_floor"], "status"].eq("available").all()
    # LCU bn per period, summed over the four households; row 0 holds initialisation placeholders.
    assert frame["income_minimum_income"].tolist()[1:] == [4.0, 8.0]
    assert frame["unpaid_rent"].tolist()[1:] == [8.0, 8.0] and frame["unmet_floor"].tolist()[1:] == [12.0, 12.0]
    assert frame.loc[0, ["income_minimum_income", "unpaid_rent", "unmet_floor"]].isna().all()
    summary = household_diagnostic_summary({"mi": frame})
    assert {"income_minimum_income", "unpaid_rent", "unmet_floor"} <= set(summary.columns)


def test_display_labels_change_only_in_minimum_income_mode_and_keys_are_kept(tmp_path):
    mi, _ = build_household_diagnostics(_run(tmp_path / "mi.h5", "{'mode': 'minimum_income'}"), country_code="FRA")
    legacy, _ = build_household_diagnostics(_run(tmp_path / "legacy.h5", "{}"), country_code="FRA")
    assert legacy.attrs["support_mode"] == "legacy_stage5"
    assert display_label("remaining_subsistence_shortfall", mi) == "unpaid debt service (no support)"
    assert display_label("remaining_subsistence_shortfall", legacy) == "remaining subsistence shortfall"
    assert display_label("unpaid_rent", mi) == "unpaid rent"
    assert "remaining_subsistence_shortfall" in mi.columns  # key unchanged

    figure = plot_household_diagnostics({"mi": mi, "legacy": legacy}, section="Credit and payment constraints")
    names = {trace.name for trace in figure.data}
    assert "unpaid debt service (no support)" in names
