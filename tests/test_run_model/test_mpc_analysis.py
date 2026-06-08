import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

RUN_MODEL_PATH = Path(__file__).resolve().parents[2] / "run_model"
if str(RUN_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL_PATH))

from src.mpc_analysis import (  # noqa: E402
    MPCFilterConfig,
    add_mpc_bins,
    build_household_mpc_panel,
    filter_mpc_panel,
    make_distribution_plot,
    period_to_year_month,
    read_household_target_consumption_sum,
    summarize_mpc_bins,
)


def _write_household_h5(path, *, country_code, consumption, target_consumption, cpi=None):
    with h5py.File(path, "w") as handle:
        country_group = handle.create_group(country_code)
        economy_group = country_group.create_group("economy")
        if cpi is None:
            cpi = np.ones(consumption.shape[0])
        economy_group.create_dataset("cpi_fixed_basket", data=np.asarray(cpi, dtype=float).reshape(-1, 1))
        group = country_group.create_group("households")
        group.create_dataset("consumption", data=consumption)
        flattened = target_consumption.reshape(target_consumption.shape[0], -1)
        columns = []
        for household_id in range(target_consumption.shape[1]):
            for industry_id in range(target_consumption.shape[2]):
                columns.append([household_id, industry_id])
        group.create_dataset("target_consumption", data=flattened)
        group.create_dataset("target_consumption_columns", data=np.asarray(columns, dtype=int))


def test_period_to_year_month_uses_zero_based_simulation_periods():
    assert period_to_year_month(2014, 3, 0) == (2014, 1)
    assert period_to_year_month(2014, 3, 1) == (2014, 4)
    assert period_to_year_month(2014, 3, 4) == (2015, 1)


def test_read_household_target_consumption_sum_reconstructs_flattened_h5(tmp_path):
    path = tmp_path / "simulation.h5"
    target = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    _write_household_h5(
        path,
        country_code="FRA",
        consumption=np.zeros((2, 2)),
        target_consumption=target,
    )

    result = read_household_target_consumption_sum(path, "FRA")

    np.testing.assert_allclose(result, np.array([[3.0, 7.0], [11.0, 15.0]]))


def test_build_household_mpc_panel_computes_impact_and_cumulative(tmp_path):
    baseline_path = tmp_path / "baseline.h5"
    shock_path = tmp_path / "shock.h5"
    baseline_consumption = np.array(
        [
            [10.0, 20.0],
            [10.0, 20.0],
            [10.0, 20.0],
            [10.0, 20.0],
        ]
    )
    shock_consumption = baseline_consumption + np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ]
    )
    baseline_target = np.repeat(baseline_consumption[:, :, None] / 2.0, 2, axis=2)
    shock_target = np.repeat(shock_consumption[:, :, None] / 2.0, 2, axis=2)
    _write_household_h5(
        baseline_path,
        country_code="FRA",
        consumption=baseline_consumption,
        target_consumption=baseline_target,
    )
    _write_household_h5(
        shock_path,
        country_code="FRA",
        consumption=shock_consumption,
        target_consumption=shock_target,
    )
    metadata = pd.DataFrame(
        {
            "seed": [1, 1],
            "household_id": [0, 1],
            "income": [100.0, 300.0],
            "net_wealth": [0.0, 10.0],
            "net_liquid_financial_assets": [0.0, 5.0],
            "illiquid_financial_assets": [0.0, 8.0],
            "housing_wealth": [0.0, 50.0],
            "housing_tenure": ["renter", "owner_mortgage"],
        }
    )

    panel = build_household_mpc_panel(
        baseline_h5=baseline_path,
        shock_h5=shock_path,
        metadata=metadata,
        country_code="FRA",
        shock_row=1,
        horizon_periods=3,
        shock_fraction=0.10,
    )

    # Median positive income is 200, so the equal household shock is 20.
    assert panel["shock_amount"].iloc[0] == pytest.approx(20.0)
    assert panel.loc[0, "mpc_impact"] == pytest.approx(1.0 / 20.0)
    assert panel.loc[1, "mpc_impact"] == pytest.approx(2.0 / 20.0)
    assert panel.loc[0, "cmpc_4q"] == pytest.approx((1.0 + 2.0 + 3.0) / 20.0)
    assert panel.loc[1, "target_cmpc_4q"] == pytest.approx((2.0 + 3.0 + 4.0) / 20.0)
    assert panel.loc[0, "real_mpc_impact"] == pytest.approx(panel.loc[0, "mpc_impact"])
    assert panel.loc[1, "target_real_cmpc_4q"] == pytest.approx(panel.loc[1, "target_cmpc_4q"])


def test_build_household_mpc_panel_deflates_each_path_by_own_cpi(tmp_path):
    baseline_path = tmp_path / "baseline.h5"
    shock_path = tmp_path / "shock.h5"
    baseline_consumption = np.array([[10.0], [10.0], [10.0]])
    shock_consumption = np.array([[10.0], [12.0], [12.0]])
    baseline_target = baseline_consumption[:, :, None]
    shock_target = shock_consumption[:, :, None]
    _write_household_h5(
        baseline_path,
        country_code="FRA",
        consumption=baseline_consumption,
        target_consumption=baseline_target,
        cpi=np.array([1.0, 1.0, 1.0]),
    )
    _write_household_h5(
        shock_path,
        country_code="FRA",
        consumption=shock_consumption,
        target_consumption=shock_target,
        cpi=np.array([1.0, 2.0, 2.0]),
    )
    metadata = pd.DataFrame({"household_id": [0], "income": [100.0]})

    panel = build_household_mpc_panel(
        baseline_h5=baseline_path,
        shock_h5=shock_path,
        metadata=metadata,
        country_code="FRA",
        shock_row=1,
        horizon_periods=2,
        shock_fraction=0.10,
        cumulative_mpc_column="cmpc_2p",
        target_cumulative_mpc_column="target_cmpc_2p",
        real_cumulative_mpc_column="real_cmpc_2p",
        target_real_cumulative_mpc_column="target_real_cmpc_2p",
    )

    assert panel.loc[0, "cmpc_2p"] == pytest.approx((2.0 + 2.0) / 10.0)
    # Own-path real consumption deltas are (12/2 - 10/1) in both response rows.
    # The real shock denominator is 10/2 because the income shock is received in
    # the shocked path at row 1.
    assert panel.loc[0, "real_cmpc_2p"] == pytest.approx((-4.0 + -4.0) / 5.0)


def test_build_household_mpc_panel_supports_custom_cumulative_column_names(tmp_path):
    baseline_path = tmp_path / "baseline.h5"
    shock_path = tmp_path / "shock.h5"
    baseline_consumption = np.array([[10.0], [10.0], [10.0]])
    shock_consumption = np.array([[10.0], [11.0], [12.0]])
    baseline_target = baseline_consumption[:, :, None]
    shock_target = shock_consumption[:, :, None]
    _write_household_h5(
        baseline_path,
        country_code="FRA",
        consumption=baseline_consumption,
        target_consumption=baseline_target,
    )
    _write_household_h5(
        shock_path,
        country_code="FRA",
        consumption=shock_consumption,
        target_consumption=shock_target,
    )
    metadata = pd.DataFrame({"household_id": [0], "income": [100.0]})

    panel = build_household_mpc_panel(
        baseline_h5=baseline_path,
        shock_h5=shock_path,
        metadata=metadata,
        country_code="FRA",
        shock_row=1,
        horizon_periods=2,
        shock_fraction=0.10,
        cumulative_mpc_column="cmpc_2p",
        target_cumulative_mpc_column="target_cmpc_2p",
    )

    assert "cmpc_4q" not in panel.columns
    assert panel.loc[0, "cmpc_2p"] == pytest.approx((1.0 + 2.0) / 10.0)
    assert panel.loc[0, "target_cmpc_2p"] == pytest.approx((1.0 + 2.0) / 10.0)


def test_add_mpc_bins_handles_repeated_values():
    panel = pd.DataFrame(
        {
            "income": [1.0, 1.0, 1.0, 2.0],
            "net_wealth": [0.0, 0.0, 0.0, 0.0],
            "net_liquid_financial_assets": [0.0, 1.0, 2.0, 3.0],
            "illiquid_financial_assets": [0.0, 0.0, 1.0, 1.0],
            "housing_wealth": [0.0, 0.0, 10.0, 20.0],
            "housing_tenure": ["renter", "renter", "owner", "owner"],
        }
    )

    result = add_mpc_bins(panel, n_bins=5)

    assert result["net_wealth_bin"].tolist() == ["all", "all", "all", "all"]
    assert result["housing_tenure_bin"].tolist() == ["renter", "renter", "owner", "owner"]


def test_make_distribution_plot_sorts_quantile_bins():
    panel = pd.DataFrame(
        {
            "income_bin": ["Q5", "Q1", "Q3", "Q2", "Q4"],
            "income": [5, 1, 3, 2, 4],
            "cmpc_4q": [0.5, 0.1, 0.3, 0.2, 0.4],
        }
    )

    fig = make_distribution_plot(panel, variable="income")

    assert list(fig.layout.xaxis.categoryarray) == ["Q1", "Q2", "Q3", "Q4", "Q5"]


def test_make_distribution_plot_orders_housing_tenure_labels():
    panel = pd.DataFrame(
        {
            "housing_tenure_bin": ["renter", "unknown", "owner_mortgage", "owner_outright", "owner_other"],
            "housing_tenure": ["renter", "unknown", "owner_mortgage", "owner_outright", "owner_other"],
            "cmpc_4q": [0.3, 0.5, 0.2, 0.1, 0.4],
        }
    )

    fig = make_distribution_plot(panel, variable="housing_tenure")

    assert list(fig.layout.xaxis.categoryarray) == [
        "owner_outright",
        "owner_mortgage",
        "renter",
        "owner_other",
        "unknown",
    ]


def test_make_distribution_plot_can_clip_y_axis_to_quantiles():
    panel = pd.DataFrame(
        {
            "income_bin": ["Q1", "Q1", "Q2", "Q2"],
            "income": [1, 2, 3, 4],
            "cmpc_4q": [0.0, 1.0, 2.0, 100.0],
        }
    )

    fig = make_distribution_plot(panel, variable="income", y_quantiles=(0.25, 0.75))

    assert list(fig.layout.yaxis.range) == pytest.approx([0.75, 26.5])


def test_make_distribution_plot_can_winsorise_display_values():
    panel = pd.DataFrame(
        {
            "income_bin": ["Q1", "Q1", "Q2", "Q2"],
            "income": [1, 2, 3, 4],
            "cmpc_4q": [0.0, 1.0, 2.0, 100.0],
        }
    )

    fig = make_distribution_plot(panel, variable="income", plot_kind="box", winsor_quantiles=(0.25, 0.75))

    plotted_values = np.concatenate([np.asarray(trace.y, dtype=float) for trace in fig.data])
    assert plotted_values.min() == pytest.approx(0.75)
    assert plotted_values.max() == pytest.approx(26.5)


def test_make_distribution_plot_rejects_conflicting_y_display_filters():
    panel = pd.DataFrame(
        {
            "income_bin": ["Q1"],
            "income": [1],
            "cmpc_4q": [0.5],
        }
    )

    with pytest.raises(ValueError, match="either y_quantiles or y_range"):
        make_distribution_plot(panel, variable="income", y_quantiles=(0.01, 0.99), y_range=(-1, 1))


def test_summarize_mpc_bins_weights_real_mpcs_by_real_shock_amount():
    panel = add_mpc_bins(
        pd.DataFrame(
            {
                "income": [1.0, 2.0],
                "net_wealth": [1.0, 2.0],
                "net_liquid_financial_assets": [1.0, 2.0],
                "illiquid_financial_assets": [1.0, 2.0],
                "housing_wealth": [1.0, 2.0],
                "housing_tenure": ["renter", "renter"],
                "real_cmpc_4q": [0.0, 1.0],
                "cmpc_4q": [0.0, 1.0],
                "shock_amount": [100.0, 100.0],
                "real_shock_amount": [1.0, 3.0],
            }
        ),
        n_bins=1,
    )

    real_summary = summarize_mpc_bins(panel, mpc_column="real_cmpc_4q")
    nominal_summary = summarize_mpc_bins(panel, mpc_column="cmpc_4q")

    real_income = real_summary.loc[real_summary["variable"] == "income"].iloc[0]
    nominal_income = nominal_summary.loc[nominal_summary["variable"] == "income"].iloc[0]
    assert real_income["weighted_mean"] == pytest.approx(0.75)
    assert nominal_income["weighted_mean"] == pytest.approx(0.5)


def test_filter_mpc_panel_applies_core_analysis_sample_rules():
    panel = pd.DataFrame(
        {
            "household_id": [0, 1, 2, 3],
            "income": [200.0, 50.0, 300.0, 400.0],
            "baseline_consumption_impact": [100.0, 40.0, -1.0, 120.0],
            "shock_consumption_impact": [105.0, 42.0, 2.0, 125.0],
            "baseline_consumption_to_income": [0.50, 0.80, -0.01, 0.30],
            "income_growth": [0.02, 0.02, 0.02, 6.50],
            "debt_asset_ratio": [0.40, 0.10, 0.20, 0.30],
            "gross_wealth": [1000.0, 1000.0, 1000.0, 1000.0],
            "head_age": [40.0, 40.0, 40.0, 40.0],
            "abs_income_change": [4.0, 1.0, 6.0, 8.0],
            "abs_income_change_income_share": [0.02, 0.02, 0.02, 0.02],
            "shock_amount": [4.0, 4.0, 4.0, 4.0],
            "real_cmpc_4q": [0.2, 0.3, 0.4, 0.5],
        }
    )
    config = MPCFilterConfig(
        min_income=100.0,
        income_top_quantile=None,
        consumption_income_ratio_quantiles=None,
        income_growth_min=-0.8,
        income_growth_max=5.0,
        max_debt_asset_ratio=1.0,
        gross_wealth_top_quantile=None,
        head_age_min=25.0,
        head_age_max=75.0,
        min_abs_income_change_income_share=0.01,
    )

    filtered, report = filter_mpc_panel(panel, config, required_mpc_columns=["real_cmpc_4q"])

    assert filtered["household_id"].tolist() == [0]
    dropped_by_filter = dict(zip(report["filter"], report["dropped"], strict=False))
    assert dropped_by_filter["minimum_income"] == 1
    assert dropped_by_filter["positive_baseline_consumption"] == 1
    assert dropped_by_filter["income_growth_ceiling"] == 1


def test_filter_mpc_panel_reports_skipped_optional_missing_columns():
    panel = pd.DataFrame(
        {
            "household_id": [0],
            "income": [200.0],
            "baseline_consumption_impact": [100.0],
            "shock_consumption_impact": [105.0],
            "real_cmpc_4q": [0.2],
        }
    )
    config = MPCFilterConfig(
        income_top_quantile=None,
        consumption_income_ratio_quantiles=None,
        income_growth_min=None,
        income_growth_max=None,
        max_debt_asset_ratio=1.0,
        gross_wealth_top_quantile=None,
        head_age_min=None,
        head_age_max=None,
    )

    filtered, report = filter_mpc_panel(panel, config, required_mpc_columns=["real_cmpc_4q"])

    assert filtered["household_id"].tolist() == [0]
    skipped = report.loc[report["skipped"]]
    assert "debt_asset_ratio_ceiling" in set(skipped["filter"])
    assert "Missing column: debt_asset_ratio." in set(skipped["reason"])


def test_filter_mpc_panel_keeps_zero_asset_zero_debt_households():
    panel = pd.DataFrame(
        {
            "household_id": [0, 1],
            "income": [200.0, 200.0],
            "baseline_consumption_impact": [100.0, 100.0],
            "shock_consumption_impact": [105.0, 105.0],
            "debt_asset_ratio": [0.0, np.inf],
            "real_cmpc_4q": [0.2, 0.3],
        }
    )
    config = MPCFilterConfig(
        min_income=None,
        income_top_quantile=None,
        consumption_income_ratio_quantiles=None,
        income_growth_min=None,
        income_growth_max=None,
        max_debt_asset_ratio=1.0,
        gross_wealth_top_quantile=None,
        head_age_min=None,
        head_age_max=None,
    )

    filtered, report = filter_mpc_panel(panel, config, required_mpc_columns=["real_cmpc_4q"])

    assert filtered["household_id"].tolist() == [0]
    dropped_by_filter = dict(zip(report["filter"], report["dropped"], strict=False))
    assert dropped_by_filter["debt_asset_ratio_ceiling"] == 1


def test_make_distribution_plot_rebins_positive_only_asset_variables():
    panel = add_mpc_bins(
        pd.DataFrame(
            {
                "housing_wealth": [0.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
                "income": [1.0] * 7,
                "net_wealth": [1.0] * 7,
                "net_liquid_financial_assets": [1.0] * 7,
                "illiquid_financial_assets": [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "housing_tenure": ["renter"] * 7,
                "cmpc_4q": [0.1] * 7,
            }
        )
    )

    housing_fig = make_distribution_plot(panel, variable="housing_wealth")
    illiquid_fig = make_distribution_plot(panel, variable="illiquid_financial_assets")

    assert list(housing_fig.layout.xaxis.categoryarray) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert list(illiquid_fig.layout.xaxis.categoryarray) == ["Q1", "Q2", "Q3", "Q4", "Q5"]


def test_make_distribution_plot_can_use_box_only_traces():
    panel = pd.DataFrame(
        {
            "income_bin": ["Q1", "Q1", "Q2", "Q2"],
            "income": [1, 2, 3, 4],
            "cmpc_4q": [0.1, 0.2, 0.3, 0.4],
        }
    )

    fig = make_distribution_plot(panel, variable="income", plot_kind="box")

    assert {trace.type for trace in fig.data} == {"box"}


def test_build_household_mpc_panel_requires_complete_household_ids(tmp_path):
    baseline_path = tmp_path / "baseline.h5"
    shock_path = tmp_path / "shock.h5"
    consumption = np.zeros((2, 2))
    target = np.zeros((2, 2, 1))
    _write_household_h5(
        baseline_path,
        country_code="FRA",
        consumption=consumption,
        target_consumption=target,
    )
    _write_household_h5(
        shock_path,
        country_code="FRA",
        consumption=consumption,
        target_consumption=target,
    )
    metadata = pd.DataFrame(
        {
            "household_id": [0, 2],
            "income": [100.0, 200.0],
        }
    )

    with pytest.raises(ValueError, match="household_id"):
        build_household_mpc_panel(
            baseline_h5=baseline_path,
            shock_h5=shock_path,
            metadata=metadata,
            country_code="FRA",
            shock_row=1,
            horizon_periods=1,
        )
