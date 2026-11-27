import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "run_model"))
from src.mpc_paper import write_paper_summaries


def test_ratio_weights_and_seed_uncertainty(tmp_path):
    panel = pd.DataFrame(
        {"seed": [1, 1, 2], "household_id": [0, 1, 0], "shock_amount": [1.0, 1.0, 6.0], "mpc_impact": [0.0, 0.0, 1.0]}
    )
    write_paper_summaries(panel, tmp_path, sample="test")
    result = pd.read_csv(tmp_path / "paper_mpc_summary_test.csv").iloc[0]
    assert result.transfer_weighted_mpc == 0.75
    assert np.isclose(result.seed_cluster_se, 0.375)
    assert np.isclose(result["mean"], 1 / 3)


def test_cumulative_identification_and_real_weights(tmp_path):
    panel = pd.DataFrame(
        {
            "seed": [1, 2],
            "household_id": [0, 0],
            "shock_amount": [5.0, 5.0],
            "real_shock_amount": [1.0, 3.0],
            "real_mpc_impact": [-1.0, 1.0],
            "real_cmpc_4q": [-1.0, 1.0],
            "cumulative_mpc_identified": [True, False],
        }
    )
    write_paper_summaries(panel, tmp_path, sample="test")
    result = pd.read_csv(tmp_path / "paper_mpc_summary_test.csv").set_index("outcome")
    assert result.loc["real_mpc_impact", "transfer_weighted_mpc"] == 0.5
    assert result.loc["real_cmpc_4q", "n"] == 1
    assert result.loc["real_cmpc_4q", "transfer_weighted_mpc"] == -1


def test_channel_counts_overlap_and_ignore_missing(tmp_path):
    from src.mpc_paper import write_negative_mpc_diagnostics

    panel = pd.DataFrame(
        {
            "real_mpc_impact": [-1.0, -2.0, 1.0],
            "diagnostic_income_t1": [10.0, np.nan, 10.0],
            "diagnostic_income_shock_t1": [8.0, 12.0, 20.0],
        }
    )
    write_negative_mpc_diagnostics(panel, tmp_path, sample="test")
    r = pd.read_csv(tmp_path / "negative_mpc_channels_test.csv").iloc[0]
    assert r.negative_n == 2
    assert r.observed_n == 1
    assert r.lower_n == 1
    assert r.higher_n == 0


def test_missing_channel_data_still_writes_readable_csv(tmp_path):
    from src.mpc_paper import write_negative_mpc_diagnostics

    write_negative_mpc_diagnostics(pd.DataFrame({"real_mpc_impact": [-1.0]}), tmp_path, sample="old")
    result = pd.read_csv(tmp_path / "negative_mpc_channels_old.csv")
    assert result.empty
    assert "observed_n" in result.columns


def test_figure_preserves_constant_groups_and_readable_borrowing_label(tmp_path):
    from src.mpc_paper_figures import write_paper_figures

    panel = pd.DataFrame(
        {
            "income": [1.0, 1.0],
            "labor_income": [0.0, 0.0],
            "non_property_income": [1.0, 1.0],
            "property_income": [0.0, 0.0],
            "liquid_financial_assets": [2.0, 2.0],
            "net_liquid_financial_assets_all_debt": [2.0, 2.0],
            "illiquid_financial_assets": [3.0, 3.0],
            "housing_wealth": [4.0, 4.0],
            "housing_tenure": [None, "owner_other"],
            "activity_bracket_t0": [1, 1],
            "consumption_loan_debt": [0.0, 0.0],
            "mortgage_debt": [0.0, 0.0],
            "mpc_regime": ["unconstrained", "unconstrained"],
            "real_mpc_impact": [0.2, 0.8],
            "real_shock_amount": [1.0, 3.0],
        }
    )
    write_paper_figures(panel, tmp_path)
    groups = pd.read_csv(tmp_path / "paper_figure_group_means.csv")
    assert groups.loc[groups.variable.eq("income_paper_group"), "n"].sum() == 2
    assert groups.loc[groups.variable.eq("illiquid_financial_assets_paper_group"), "n"].sum() == 2
    assert groups.loc[groups.variable.eq("borrowing"), "group"].tolist() == ["No borrowing"]
    assert groups.loc[groups.variable.eq("tenure"), "group"].tolist() == ["Other tenure"]
    assert (tmp_path / "paper_impact_mpc.png").stat().st_size > 0


def test_summary_and_figure_use_same_positive_asset_groups(tmp_path):
    from src.mpc_paper_figures import write_paper_figures

    panel = pd.DataFrame(
        {
            "seed": [1] * 10,
            "household_id": range(10),
            "illiquid_financial_assets": [0.0] * 5 + [1.0, 2.0, 3.0, 4.0, 5.0],
            # Legacy whole-sample bins must not substitute for positive-holder groups.
            "illiquid_financial_assets_bin": ["Q1", "Q1", "Q2", "Q2", "Q3", "Q3", "Q4", "Q4", "Q5", "Q5"],
            "real_mpc_impact": np.arange(10) / 10,
            "real_shock_amount": np.arange(10) + 1.0,
        }
    )
    write_paper_summaries(panel, tmp_path, sample="groups")
    write_paper_figures(panel, tmp_path)
    summary = pd.read_csv(tmp_path / "paper_mpc_summary_groups.csv")
    figure = pd.read_csv(tmp_path / "paper_figure_group_means.csv")
    matched = figure.merge(summary, on=["variable", "group"], validate="one_to_one", suffixes=("_figure", "_summary"))
    assert len(matched) == len(figure) == 5
    np.testing.assert_array_equal(matched.n_figure, matched.n_summary)
    np.testing.assert_allclose(matched.mean_figure, matched.mean_summary)
    np.testing.assert_allclose(matched.transfer_weighted_mpc_figure, matched.transfer_weighted_mpc_summary)
    assert "Zero" not in figure.group.tolist()
    assert summary.set_index("group").loc["Zero", "n"] == 5


def test_consumption_constraint_excludes_arrears_only_and_other_regimes():
    from src.mpc_paper_groups import prepare_paper_groups

    panel = pd.DataFrame(
        {
            "mpc_regime": [
                "liquidity_constrained",
                "liquidity_constrained",
                "unconstrained",
                "minimum_income_recipient",
                "floor_bound_self_funded",
            ],
            "baseline_consumption_cut_amount": [10.0, 0.0, 0.0, 10.0, 0.0],
            "baseline_subsistence_floor": [100.0] * 5,
        }
    )
    grouped, _ = prepare_paper_groups(panel)
    assert grouped.constraint.iloc[0] == "Constrained"
    assert pd.isna(grouped.constraint.iloc[1])
    assert grouped.constraint.iloc[2] == "Unconstrained"
    assert grouped.constraint.iloc[3:].isna().all()


def test_constraint_requires_evidence_and_accepts_saved_diagnostics():
    import pytest
    from src.mpc_paper_groups import prepare_paper_groups

    panel = pd.DataFrame({"mpc_regime": ["liquidity_constrained"], "baseline_subsistence_floor": [100.0]})
    with pytest.raises(ValueError, match="baseline consumption cuts"):
        prepare_paper_groups(panel)
    panel["diagnostic_consumption_cut_amount_t1"] = [10.0]
    grouped, _ = prepare_paper_groups(panel)
    assert grouped.constraint.iloc[0] == "Constrained"


def test_empty_filtered_summary_has_readable_schema(tmp_path):
    panel = pd.DataFrame(columns=["seed", "shock_amount", "mpc_impact"])
    write_paper_summaries(panel, tmp_path, sample="empty")
    result = pd.read_csv(tmp_path / "paper_mpc_summary_empty.csv")
    assert result.empty
    assert {"outcome", "n", "mean", "transfer_weighted_mpc", "seed_cluster_se"}.issubset(result.columns)


def test_cumulative_figure_uses_identified_rows_without_rebinning(tmp_path):
    from src.mpc_paper_figures import write_paper_figures

    panel = pd.DataFrame(
        {
            "seed": [1] * 10,
            "income": np.arange(1, 11, dtype=float),
            "real_mpc_impact": [0.1] * 10,
            "real_cmpc_4q": np.arange(10) / 10 + 1.0,
            "real_shock_amount": [2.0] * 10,
            "cumulative_mpc_identified": [False] * 5 + [True] * 5,
        }
    )
    write_paper_figures(panel, tmp_path, measure="cumulative_4q", show_transfer_weighted_mean=False)
    groups = pd.read_csv(tmp_path / "paper_cumulative_4q_figure_group_means.csv").set_index("group")
    assert groups.index.tolist() == ["Q3", "Q4", "Q5"]
    assert groups.n.tolist() == [1, 2, 2]
    np.testing.assert_allclose(groups["mean"], [1.5, 1.65, 1.85])
    write_paper_summaries(panel, tmp_path, sample="cumulative")
    table = pd.read_csv(tmp_path / "paper_mpc_summary_cumulative.csv")
    table = table[(table.variable == "income_paper_group") & (table.outcome == "real_cmpc_4q")].set_index("group")
    np.testing.assert_allclose(groups["mean"], table.loc[groups.index, "mean"])
    assert (tmp_path / "paper_cumulative_4q_mpc.png").stat().st_size > 0
    assert not (tmp_path / "paper_impact_mpc.png").exists()
