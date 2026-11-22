import inspect
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

RUN_MODEL_PATH = Path(__file__).resolve().parents[2] / "run_model"
if str(RUN_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL_PATH))

from src.household_support import (  # noqa: E402
    SUPPORT_LEGACY,
    SUPPORT_MINIMUM_INCOME,
    minimum_income_metrics,
    resolve_household_support,
)

N, ROWS = 10, 4


def _write_run(path, *, configuration=None, drop=(), **overrides):
    """Ten households ranked 1..10 by cash income before support, four rows (row 0 = initialisation)."""
    base = np.tile(100.0 * np.arange(1, N + 1), (ROWS, 1))
    top_up = np.zeros((ROWS, N))
    top_up[1:, 0] = top_up[1:, 1] = 50.0  # Q1: T/M = 0.5
    top_up[1:, 2] = 20.0  # Q2: T/M = 0.2
    social = np.zeros((ROWS, N))
    social[:, 5] = [0, 1, 0, 1]  # enters at 1, leaves at 2, re-enters at 3
    social[:, 6] = [1, 0, 0, 0]  # leaves at 1
    social[:, 7] = [0, 0, 0, 1]  # enters at 3, never left
    social[:, 8] = [1, 0, 0, 1]  # leaves at 1, re-enters at 3
    unpaid = np.zeros((ROWS, N))
    unpaid[2, 0] = 5.0
    unpaid[1:3, 9] = 1.0
    unmet = np.zeros((ROWS, N))
    unmet[1:, 3] = 2.0
    guard = np.zeros((ROWS, N))
    guard[1, 0] = 1.0
    data = {
        "income": base + top_up,
        "income_minimum_income": top_up,
        "stage5_subsistence_support": np.zeros((ROWS, N)),
        "minimum_income_level": np.full((ROWS, N), 100.0),
        "income_guard_bound": guard,
        "unpaid_rent": unpaid,
        "social_housing_tenant": social,
        "unmet_floor": unmet,
    }
    data.update(overrides)
    with h5py.File(path, "w") as handle:
        for name, values in data.items():
            if name not in drop:
                handle.create_dataset(f"FRA/households/{name}", data=values)
        handle.create_dataset("FRA/economy/gdp_output", data=np.full((ROWS, 1), 1000.0))
        if configuration is not None:
            handle.attrs["configuration"] = configuration
    return path


def test_metrics_by_income_quintile_on_a_synthetic_run(tmp_path):
    table = minimum_income_metrics(_write_run(tmp_path / "run.h5"), "FRA", first_period=1)
    assert list(table.index) == ["Q1", "Q2", "Q3", "Q4", "Q5", "all"]
    assert table["n_household_periods"].tolist() == [6, 6, 6, 6, 6, 30]
    assert table["share_receiving_T"].tolist() == pytest.approx([1.0, 0.5, 0, 0, 0, 0.3])
    np.testing.assert_allclose(table["mean_T_over_M_recipients"].iloc[[0, 1, 5]], [0.5, 0.2, 0.4])
    assert table["mean_T_over_M_recipients"].iloc[2:5].isna().all()
    # Sum of T per period over GDP: Q1 100/1000, Q2 20/1000; the quintiles add up to "all".
    assert table["fiscal_cost_pct_gdp"].tolist() == pytest.approx([10.0, 2.0, 0, 0, 0, 12.0])
    assert table["guard_binding_share"].tolist() == pytest.approx([1 / 6, 0, 0, 0, 0, 1 / 30])
    assert table["unpaid_rent_events"].tolist() == [1, 0, 0, 0, 2, 3]
    assert table["social_housing_entries"].tolist() == [0, 0, 2, 1, 1, 4]
    assert table["re_entries_within_4q"].tolist() == [0, 0, 1, 0, 1, 2]
    assert table["unmet_floor_share"].tolist() == pytest.approx([0, 0.5, 0, 0, 0, 0.1])


def test_metrics_default_to_activation_plus_the_four_quarter_guard_window(tmp_path):
    assert inspect.signature(minimum_income_metrics).parameters["first_period"].default == 5
    with pytest.raises(ValueError, match="beyond"):  # the four-row synthetic run has no row 5
        minimum_income_metrics(_write_run(tmp_path / "run.h5"), "FRA")


def test_wealth_ratio_clipped_share_counts_a_household_period_once_if_any_flag_is_set(tmp_path):
    lfa, drag = np.zeros((ROWS, N)), np.zeros((ROWS, N))
    lfa[1, 0] = lfa[2, 0] = 1.0  # household 0 (Q1)
    drag[2, 0] = 1.0  # same household-period as an lfa flag: counted once
    drag[3, 9] = 1.0  # household 9 (Q5)
    path = _write_run(tmp_path / "run.h5")
    with h5py.File(path, "a") as handle:
        handle.create_dataset("FRA/households/target_consumption_lfa_ratio_winsorized", data=lfa)
        handle.create_dataset("FRA/households/target_consumption_wealth_drag_clipped", data=drag)
    table = minimum_income_metrics(path, "FRA", first_period=1)
    assert table["wealth_ratio_clipped_share"].tolist() == pytest.approx([2 / 6, 0, 0, 0, 1 / 6, 3 / 30])
    # Without any of the flags in the file the metric is missing, not zero.
    assert (
        minimum_income_metrics(_write_run(tmp_path / "bare.h5"), "FRA", first_period=1)["wealth_ratio_clipped_share"]
        .isna()
        .all()
    )


def test_metrics_use_exit_history_before_first_period_and_count_from_first_period(tmp_path):
    table = minimum_income_metrics(_write_run(tmp_path / "run.h5"), "FRA", first_period=3)
    # Only the moves into social housing at row 3 count; households 5 and 8 left earlier, 7 never did.
    assert table.loc["all", "social_housing_entries"] == 3
    assert table.loc["all", "re_entries_within_4q"] == 2
    assert table.loc["all", "n_household_periods"] == N


def test_metrics_rank_households_before_support_so_recipients_keep_their_place(tmp_path):
    # Household 0 receives a top-up that lifts its cash income above everyone's: it must still be in Q1.
    top_up = np.zeros((ROWS, N))
    top_up[1:, 0] = 5000.0
    base = np.tile(100.0 * np.arange(1, N + 1), (ROWS, 1))
    path = _write_run(tmp_path / "run.h5", income=base + top_up, income_minimum_income=top_up)
    table = minimum_income_metrics(path, "FRA", first_period=1)
    assert table.loc["Q1", "share_receiving_T"] == 0.5
    assert table.loc["Q5", "share_receiving_T"] == 0.0


def test_metrics_report_missing_series_as_nan_and_validate_arguments(tmp_path):
    path = _write_run(tmp_path / "run.h5", drop=("social_housing_tenant", "unmet_floor", "income_guard_bound"))
    table = minimum_income_metrics(path, "FRA", first_period=1)
    for column in ("social_housing_entries", "re_entries_within_4q", "unmet_floor_share", "guard_binding_share"):
        assert table[column].isna().all()
    assert table.loc["all", "unpaid_rent_events"] == 3
    with pytest.raises(ValueError, match="first_period must be at least 1"):
        minimum_income_metrics(path, "FRA", first_period=0)
    with pytest.raises(ValueError, match="beyond"):
        minimum_income_metrics(path, "FRA", first_period=ROWS)
    with pytest.raises(ValueError, match="required"):
        minimum_income_metrics(_write_run(tmp_path / "bad.h5", drop=("income_minimum_income",)), "FRA", first_period=1)


def test_metrics_of_a_legacy_run_have_no_recipients(tmp_path):
    zeros = np.zeros((ROWS, N))
    path = _write_run(tmp_path / "run.h5", income_minimum_income=zeros, minimum_income_level=zeros)
    table = minimum_income_metrics(path, "FRA", first_period=1)
    assert (table["share_receiving_T"] == 0).all() and (table["fiscal_cost_pct_gdp"] == 0).all()
    assert table["mean_T_over_M_recipients"].isna().all()


@pytest.mark.parametrize(
    "block, expected",
    [
        ("{}", (SUPPORT_LEGACY, True)),
        ("{'mode': 'minimum_income'}", (SUPPORT_MINIMUM_INCOME, True)),
        ("{'mode': 'minimum_income', 'goods_floor': False}", (SUPPORT_MINIMUM_INCOME, False)),
    ],
)
def test_resolve_support_reads_the_saved_configuration(tmp_path, block, expected):
    configuration = "{'t_max': 12, 'country_configurations': {'FRA': {'household_support': " + block + "}}}"
    # Saved configuration wins over the series, which here look like a minimum_income run.
    path = _write_run(tmp_path / "run.h5", configuration=configuration)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolved = resolve_household_support(path, "FRA")
    assert (resolved["mode"], resolved["goods_floor"], resolved["source"]) == (*expected, "saved_configuration")


def test_resolve_support_prefers_the_argument_and_rejects_unknown_modes(tmp_path):
    path = _write_run(tmp_path / "run.h5")
    resolved = resolve_household_support(path, "FRA", {"mode": "minimum_income", "goods_floor": False})
    assert (resolved["mode"], resolved["goods_floor"], resolved["source"]) == ("minimum_income", False, "argument")
    assert resolve_household_support(path, "FRA", {})["mode"] == SUPPORT_LEGACY
    with pytest.raises(ValueError, match="Unknown household_support.mode"):
        resolve_household_support(path, "FRA", {"mode": "stage6"})


def test_resolve_support_infers_the_mode_only_without_a_saved_configuration(tmp_path):
    with pytest.warns(UserWarning, match="assuming goods_floor=True"):
        inferred = resolve_household_support(_write_run(tmp_path / "a.h5"), "FRA")
    assert (inferred["mode"], inferred["goods_floor"], inferred["source"]) == (SUPPORT_MINIMUM_INCOME, True, "inferred")
    support = np.zeros((ROWS, N))
    support[2, 3] = 1.0
    with_stage5 = _write_run(tmp_path / "b.h5", stage5_subsistence_support=support)
    with pytest.warns(UserWarning):
        assert resolve_household_support(with_stage5, "FRA")["mode"] == SUPPORT_LEGACY
    no_top_up = _write_run(tmp_path / "c.h5", income_minimum_income=np.zeros((ROWS, N)))
    with pytest.warns(UserWarning):
        assert resolve_household_support(no_top_up, "FRA")["mode"] == SUPPORT_LEGACY
    unparseable = _write_run(tmp_path / "d.h5", configuration="not a python literal {")
    with pytest.warns(UserWarning):
        assert resolve_household_support(unparseable, "FRA")["source"] == "inferred"
