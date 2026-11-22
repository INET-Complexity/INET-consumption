import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.errors import PerformanceWarning

RUN_MODEL_PATH = Path(__file__).resolve().parents[2] / "run_model"
if str(RUN_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL_PATH))

from src.visual_helpers import (  # noqa: E402
    build_cpi_comparison_df,
    build_cumulative_insolvent_firms_by_sector_df,
    build_employment_by_sector_df,
    build_macro_output_df,
    build_production_by_sector_df,
    build_sector_tfp_investment_desired_mb_mc_ratio_df,
    plot_agent_timeseries,
    plot_cpi_comparison,
    plot_cumulative_insolvent_firms_by_sector,
    plot_employment_by_sector,
    plot_irf_panels,
    plot_ppi_comparison,
    plot_production_by_sector,
    plot_sector_tfp_investment_desired_mb_mc_ratio,
    split_firm_ts_balance_sheet_and_transaction_account_df,
)


def _ts(dicts):
    return SimpleNamespace(dicts=dicts)


def test_build_macro_output_df_uses_canonical_columns_and_expands_economy_series():
    index = pd.RangeIndex(3, name="t")
    shallow = pd.DataFrame(
        {
            "GDP_Expenditure": [100.0, 110.0, 121.0],
            "Household Consumption": [60.0, 66.0, 72.0],
            "Government Consumption": [20.0, 22.0, 24.0],
            "Exports": [10.0, 11.0, 12.0],
            "Imports": [5.0, 5.5, 6.0],
            "CPI": [1.0, 1.01, 1.02],
            "PPI": [1.0, 1.02, 1.04],
            "Consumption Expansion Loan Debt": [1.0, 2.0, 3.0],
            "Mortgage Debt": [4.0, 5.0, 6.0],
        },
        index=index,
    )
    gdp_components = pd.DataFrame({"+Gross_Fixed_Capital_Formation": [15.0, 16.0, 17.0]}, index=index)
    economy_ts = {
        "unemployment_rate": [0.1, 0.11, 0.12],
        "unemployment_rate_growth": [0.0, 0.1, 0.09],
        "participation_rate": [0.7, 0.71, 0.72],
        "participation_rate_growth": [0.0, 0.01, 0.01],
        "labour_input_shortfall_rate": [0.04, 0.041, 0.042],
        "labour_input_shortfall_rate_growth": [0.0, 0.03, 0.03],
        "unfilled_jobs": [12.0, 13.0, 14.0],
        "vacancy_rate": [0.03, 0.031, 0.032],
        "vacancy_rate_growth": [0.0, 0.02, 0.02],
        "job_reallocation_rate": [0.04, 0.041, 0.042],
        "job_reallocation_rate_growth": [0.0, 0.03, 0.03],
        "firm_insolvency_rate": [0.001, 0.002, 0.003],
        "bank_insolvency_rate": [0.0, 0.0, 0.01],
        "household_insolvency_rate": [0.01, 0.011, 0.012],
        "illiquid_financial_asset_return_rate": [np.nan, 0.02, -0.01],
        "total_growth": [0.0, 0.02, 0.03],
        "estimated_growth": [0.01, 0.02, 0.03],
        "cpi_transaction": [1.0, 1.01, 1.02],
        "cpi_transaction_pop_change": [0.0, 0.01, 0.01],
        "cpi_transaction_yoy_change": [0.0, 0.02, 0.03],
        "sectoral_growth": [np.array([0.01, 0.02]), np.array([0.03, 0.04]), np.array([0.05, 0.06])],
        "real_gross_output": [90.0, 95.0, 100.0],
        "potential_output": [92.0, 96.0, 101.0],
        "output_gap": [-0.02, -0.01, 0.0],
        "hpi": [1.0, 1.02, 1.04],
        "rpi_actual": [1.0, 1.0, 0.5],
        "rpi_imputed": [1.0, 2.0, 2.0],
        "social_rent_price": [1.0, 1.0, 4.0],
        "total_social_rent_paid": [np.nan, 1.0, 2.0],
        "hpi_inflation": [0.0, 0.02, 0.02],
        "estimated_hpi_inflation": [0.01, 0.011, 0.012],
        "total_real_rent_paid": [10.0, 11.0, 12.0],
        "total_imp_rent_paid": [3.0, 3.1, 3.2],
        "total_real_rent_rec": [9.0, 10.0, 11.0],
        "num_insolvent_firms_by_sector": [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])],
        "npl_firm_loans": [0.1, 0.2, 0.3],
        "npl_hh_cons_loans": [0.4, 0.5, 0.6],
    }
    country = SimpleNamespace(
        central_government=SimpleNamespace(
            ts=_ts(
                {
                    "revenue": [30.0, 31.0, 32.0],
                    "deficit": [2.0, 2.5, 3.0],
                    "debt": [100.0, 101.0, 102.0],
                    "interest_payments_on_debt": [1.0, 1.1, 1.2],
                    "total_unemployment_benefits": [4.0, 4.1, 4.2],
                    "total_public_pension_benefits": [3.0, 3.1, 3.2],
                    "total_other_social_transfers": [2.0, 2.1, 2.2],
                    "total_necessity_support": [0.1, 0.2, 0.3],
                    "total_minimum_income": [0.0, 1.5, 2.5],
                    "social_rent_written_off": [0.0, 0.2, 0.3],
                    "total_household_social_transfers": [5.0, 5.1, 5.2],
                    "taxes_vat": [6.0, 6.1, 6.2],
                    "taxes_corporate_income": [7.0, 7.1, 7.2],
                }
            )
        ),
        central_bank=SimpleNamespace(ts=_ts({"policy_rate": [0.01, 0.011, 0.012]})),
        households=SimpleNamespace(
            ts=_ts(
                {
                    "total_rent_unpaid_absorbed": [[0.0], [0.4], [0.6]],
                    "rent_unpaid_no_counterparty": [np.zeros(2), np.array([0.1, 0.2]), np.array([0.0, 0.5])],
                }
            )
        ),
        banks=SimpleNamespace(
            ts=_ts(
                {
                    "average_interest_rates_on_short_term_firm_loans": [0.02, 0.021, 0.022],
                    "average_interest_rates_on_long_term_firm_loans": [0.03, 0.031, 0.032],
                    "average_interest_rates_on_household_consumption_loans": [0.04, 0.041, 0.042],
                    "average_interest_rates_on_mortgages": [0.05, 0.051, 0.052],
                }
            )
        ),
        economy=SimpleNamespace(ts=_ts(economy_ts)),
        housing_market=SimpleNamespace(
            ts=_ts(
                {
                    "total_number_of_bought_houses": [[np.nan], [4.0], [5.0]],
                    "total_number_of_newly_rented_houses": [[np.nan], [2.0], [3.0]],
                    "total_number_of_houses_rented": [[10.0], [11.0], [12.0]],
                    "total_number_of_houses_owner_occupied": [[20.0], [21.0], [22.0]],
                    "total_number_of_houses_unoccupied": [[5.0], [4.0], [3.0]],
                }
            )
        ),
        firms=SimpleNamespace(
            industries=["agriculture", "services"],
            ts=SimpleNamespace(
                tfp_multiplier=[
                    np.array([1.0, 1.1, 1.2]),
                    np.array([1.1, 1.2, 1.3]),
                    np.array([1.2, 1.3, 1.4]),
                ],
                sector_tfp_investment_desired_mb_mc_ratio=[
                    np.array([0.9, 1.1]),
                    np.array([1.0, 1.2]),
                    np.array([1.1, 1.3]),
                ],
            ),
        ),
    )
    model = SimpleNamespace(
        timestep=SimpleNamespace(increment=1),
        countries={"FRA": country},
        shallow_df_dict=lambda: {"FRA": shallow},
        get_country_gdp_components_df=lambda country_code: gdp_components,
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        output = build_macro_output_df(model, "FRA")

    assert not [warning for warning in caught_warnings if issubclass(warning.category, PerformanceWarning)]

    expected_columns = {
        "unemployment_rate",
        "unemployment_rate_growth",
        "participation_rate",
        "participation_rate_growth",
        "labour_input_shortfall_rate",
        "labour_input_shortfall_rate_growth",
        "unfilled_jobs",
        "vacancy_rate",
        "vacancy_rate_growth",
        "job_reallocation_rate",
        "job_reallocation_rate_growth",
        "firm_insolvency_rate",
        "bank_insolvency_rate",
        "household_insolvency_rate",
        "illiquid_financial_asset_return_rate",
        "total_growth",
        "estimated_growth",
        "cpi_transaction",
        "cpi_transaction_pop_change",
        "cpi_transaction_yoy_change",
        "sectoral_growth",
        "sectoral_growth_agriculture",
        "sectoral_growth_services",
        "real_gross_output",
        "potential_output",
        "output_gap",
        "hpi",
        "hpi_inflation",
        "estimated_hpi_inflation",
        "total_real_rent_paid",
        "total_imp_rent_paid",
        "total_real_rent_rec",
        "num_insolvent_firms_by_sector",
        "num_insolvent_firms_by_sector_agriculture",
        "num_insolvent_firms_by_sector_services",
        "npl_firm_loans",
        "npl_hh_cons_loans",
        "sector_tfp_investment_desired_mb_mc_ratio",
        "sector_tfp_investment_desired_mb_mc_ratio_agriculture",
        "sector_tfp_investment_desired_mb_mc_ratio_services",
        "avg_tfp_multiplier",
        "fiscal_revenue",
        "fiscal_expenditure",
        "unemployment_benefits",
        "public_pension_benefits",
        "other_social_transfers",
        "necessity_support",
        "household_social_transfers",
        "public_pension_benefits_to_gdp",
        "other_social_transfers_to_gdp",
        "necessity_support_to_gdp",
        "household_social_transfers_to_gdp",
        "fiscal_revenue_vat",
        "fiscal_revenue_corporate_income_taxes",
    }
    assert expected_columns.issubset(output.columns)
    assert "other_benefits" not in output.columns
    assert "other_benefits_to_expenditure" not in output.columns
    assert output["sectoral_growth"].tolist() == [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]]
    assert output["illiquid_financial_asset_return_rate"].tolist()[1:] == [0.02, -0.01]
    assert output["sectoral_growth_services"].tolist() == [0.02, 0.04, 0.06]
    assert output["num_insolvent_firms_by_sector"].tolist() == [[1, 2], [3, 4], [5, 6]]
    assert output["num_insolvent_firms_by_sector_agriculture"].tolist() == [1, 3, 5]
    assert output["sector_tfp_investment_desired_mb_mc_ratio"].tolist() == [[0.9, 1.1], [1.0, 1.2], [1.1, 1.3]]
    assert output["sector_tfp_investment_desired_mb_mc_ratio_services"].tolist() == [1.1, 1.2, 1.3]
    # Each housing flow is deflated by its own index, the rest by the PPI (issue #197).
    gdp = np.array([100.0, 110.0, 121.0])
    ppi = np.array([1.0, 1.02, 1.04])
    social = np.array([0.0, 1.0, 2.0])
    actual = np.array([10.0, 11.0, 12.0]) - social
    imputed = np.array([3.0, 3.1, 3.2])
    housing = actual + imputed + social
    expected = (
        (gdp - housing) / ppi
        + actual / np.array([1.0, 1.0, 0.5])
        + imputed / np.array([1.0, 2.0, 2.0])
        + social / np.array([1.0, 1.0, 4.0])
    )
    np.testing.assert_allclose(output["real_gdp_ppi"], gdp / ppi)
    np.testing.assert_allclose(output["real_gdp"], expected)
    assert output["avg_tfp_multiplier"].tolist() == pytest.approx([1.1, 1.2, 1.3])
    assert output["public_pension_benefits"].tolist() == [3.0, 3.1, 3.2]
    assert output["other_social_transfers"].tolist() == [2.0, 2.1, 2.2]
    assert output["necessity_support"].tolist() == [0.1, 0.2, 0.3]
    assert output["household_social_transfers"].tolist() == [5.0, 5.1, 5.2]
    assert output["minimum_income"].tolist() == [0.0, 1.5, 2.5]
    assert output["social_rent_written_off"].tolist() == [0.0, 0.2, 0.3]
    assert output["landlord_rent_loss"].tolist() == [0.0, 0.4, 0.6]
    assert output["rent_unpaid_no_counterparty"].tolist() == pytest.approx([0.0, 0.3, 0.5])
    assert "social_rent_written_off_to_expenditure" not in output.columns
    assert output["social_rent_written_off_to_gdp"].tolist() == pytest.approx([0.0, 0.2 / 110, 0.3 / 121])
    assert output["minimum_income_to_gdp"].tolist() == pytest.approx([0.0, 1.5 / 110, 2.5 / 121])
    assert output["landlord_rent_loss_to_gdp"].tolist() == pytest.approx([0.0, 0.4 / 110, 0.6 / 121])
    assert output["public_pension_benefits_to_gdp"].tolist() == pytest.approx([0.03, 3.1 / 110, 3.2 / 121])
    assert output["total_number_of_bought_houses"].tolist() == pytest.approx([np.nan, 4.0, 5.0], nan_ok=True)
    assert output["total_number_of_newly_rented_houses"].tolist() == pytest.approx([np.nan, 2.0, 3.0], nan_ok=True)
    assert output["total_number_of_houses_rented"].tolist() == [10.0, 11.0, 12.0]
    assert output["total_number_of_houses_owner_occupied"].tolist() == [20.0, 21.0, 22.0]
    assert output["total_number_of_houses_unoccupied"].tolist() == [5.0, 4.0, 3.0]

    clutter_columns = {
        "revenue",
        "policy_rate",
        "government expenditure",
        "expected gdp growth",
        "unemployment rate",
        "average_interest_rates_on_short_term_firm_loans",
    }
    assert output.columns.intersection(clutter_columns).empty


def test_build_cpi_comparison_df_uses_explicit_cpi_series_names():
    economy_ts = {
        "cpi_transaction": [[1.0], [1.10], [1.21]],
        "cpi_fixed_basket": [[1.0], [1.08], [1.17]],
        "cpi_chained_basket": [[1.0], [1.09], [1.19]],
        "cpi_transaction_pop_change": [[0.0], [0.10], [0.10]],
        "cpi_fixed_basket_pop_change": [[0.0], [0.08], [0.0833]],
        "cpi_chained_basket_pop_change": [[0.0], [0.09], [0.0917]],
        "cpi_transaction_yoy_change": [[0.0], [0.10], [0.21]],
        "cpi_fixed_basket_yoy_change": [[0.0], [0.08], [0.17]],
        "cpi_chained_basket_yoy_change": [[0.0], [0.09], [0.19]],
    }
    model = SimpleNamespace(
        timestep=SimpleNamespace(increment=3),
        countries={"FRA": SimpleNamespace(economy=SimpleNamespace(ts=_ts(economy_ts)))},
    )

    output = build_cpi_comparison_df(model=model, country_code="FRA")

    expected_columns = [
        "cpi_transaction",
        "cpi_fixed_basket",
        "cpi_chained_basket",
        "cpi_transaction_pop_change",
        "cpi_fixed_basket_pop_change",
        "cpi_chained_basket_pop_change",
        "cpi_transaction_yoy_change",
        "cpi_fixed_basket_yoy_change",
        "cpi_chained_basket_yoy_change",
        "cpi_fixed_basket_minus_transaction",
        "cpi_chained_basket_minus_transaction",
        "cpi_chained_basket_minus_fixed_basket",
        "cpi_fixed_basket_pop_change_minus_transaction",
        "cpi_chained_basket_pop_change_minus_transaction",
    ]
    assert output.columns.tolist() == expected_columns
    assert output["cpi_fixed_basket_minus_transaction"].iloc[1] == pytest.approx(-0.02)
    assert output["cpi_chained_basket_pop_change_minus_transaction"].iloc[2] == pytest.approx(-0.0083)


def test_plot_agent_timeseries_aggregates_vector_series_and_can_select_agent_id():
    index = pd.RangeIndex(3, name="t")
    shallow = pd.DataFrame({"GDP_Expenditure": [1.0, 1.0, 1.0]}, index=index)
    country = SimpleNamespace(
        firms=SimpleNamespace(
            ts=_ts(
                {
                    "total_sales": [10.0, 11.0, 12.0],
                    "target_short_term_credit": [
                        np.array([1.0, 2.0]),
                        np.array([3.0, 4.0]),
                        np.array([5.0, 6.0]),
                    ],
                }
            )
        )
    )
    model = SimpleNamespace(
        countries={"FRA": country},
        shallow_df_dict=lambda: {"FRA": shallow},
    )

    fig_sum = plot_agent_timeseries(
        model=model,
        country_code="FRA",
        agent_type="firms",
        variables=["target_short_term_credit", "total_sales"],
        agg="sum",
        show=False,
    )
    assert len(fig_sum.data) == 2
    assert list(fig_sum.data[0].y) == pytest.approx([3.0, 7.0, 11.0])

    fig_agent = plot_agent_timeseries(
        model=model,
        country_code="FRA",
        agent_type="firms",
        variables=["target_short_term_credit"],
        agent_id=1,
        show=False,
    )
    assert list(fig_agent.data[0].y) == pytest.approx([2.0, 4.0, 6.0])

    fig_multi = plot_agent_timeseries(
        model=model,
        country_code="FRA",
        agent_type="firms",
        variables=["target_short_term_credit", "total_sales"],
        agent_id=[0, 1],
        show=False,
    )
    assert len(fig_multi.data) == 3
    assert [trace.name for trace in fig_multi.data][:2] == ["id=0", "id=1"]
    assert list(fig_multi.data[0].y) == pytest.approx([1.0, 3.0, 5.0])
    assert list(fig_multi.data[1].y) == pytest.approx([2.0, 4.0, 6.0])


def test_plot_agent_timeseries_can_return_the_plotted_dataframe():
    index = pd.RangeIndex(3, name="t")
    shallow = pd.DataFrame({"GDP_Expenditure": [1.0, 1.0, 1.0]}, index=index)
    country = SimpleNamespace(
        firms=SimpleNamespace(
            ts=_ts(
                {
                    "total_sales": [10.0, 11.0, 12.0],
                    "target_short_term_credit": [
                        np.array([1.0, 2.0]),
                        np.array([3.0, 4.0]),
                        np.array([5.0, 6.0]),
                    ],
                }
            )
        )
    )
    model = SimpleNamespace(
        countries={"FRA": country},
        shallow_df_dict=lambda: {"FRA": shallow},
    )

    fig, df = plot_agent_timeseries(
        model=model,
        country_code="FRA",
        agent_type="firms",
        variables=["target_short_term_credit", "total_sales"],
        agent_id=[0, 1],
        show=False,
        return_df=True,
    )

    assert len(fig.data) == 3
    assert df.index.equals(index)
    assert df.columns.tolist() == [
        "firms.target_short_term_credit[id=0]",
        "firms.target_short_term_credit[id=1]",
        "firms.total_sales",
    ]
    assert df.iloc[:, 0].tolist() == pytest.approx([1.0, 3.0, 5.0])
    assert df.iloc[:, 1].tolist() == pytest.approx([2.0, 4.0, 6.0])
    assert df.iloc[:, 2].tolist() == pytest.approx([10.0, 11.0, 12.0])


def test_split_firm_ts_balance_sheet_and_transaction_account_df_uses_curated_columns():
    index = pd.RangeIndex(2, name="t")
    model = SimpleNamespace(
        countries={
            "FRA": SimpleNamespace(
                firms=SimpleNamespace(
                    ts=_ts(
                        {
                            "inventory": [1.0, 2.0],
                            "deposits": [3.0, 4.0],
                            "equity": [5.0, 6.0],
                            "production": [7.0, 8.0],
                            "total_sales": [9.0, 10.0],
                            "received_short_term_credit": [11.0, 12.0],
                            "target_short_term_credit": [13.0, 14.0],
                            "expected_profits": [15.0, 16.0],
                        }
                    )
                )
            )
        },
        shallow_df_dict=lambda: {"FRA": pd.DataFrame(index=index)},
    )

    balance_sheet_df, transaction_account_df = split_firm_ts_balance_sheet_and_transaction_account_df(model, "FRA")

    assert balance_sheet_df.columns.tolist() == ["inventory", "deposits", "equity"]
    assert transaction_account_df.columns.tolist() == [
        "production",
        "total_sales",
        "received_short_term_credit",
    ]
    assert "target_short_term_credit" not in transaction_account_df.columns
    assert "expected_profits" not in transaction_account_df.columns


def test_plot_agent_timeseries_uses_bare_variable_names_and_side_legend():
    index = pd.RangeIndex(3, name="t")
    shallow = pd.DataFrame({"GDP_Expenditure": [1.0, 1.0, 1.0]}, index=index)
    country = SimpleNamespace(
        firms=SimpleNamespace(
            ts=_ts(
                {
                    "total_sales": [10.0, 11.0, 12.0],
                }
            )
        ),
        households=SimpleNamespace(
            ts=_ts(
                {
                    "target_consumption": [4.0, 5.0, 6.0],
                }
            )
        ),
    )
    model = SimpleNamespace(
        countries={"FRA": country},
        shallow_df_dict=lambda: {"FRA": shallow},
    )

    fig = plot_agent_timeseries(
        model=model,
        country_code="FRA",
        agent_type="firms",
        variables=["households.target_consumption", "firms.total_sales"],
        show_legend=True,
        show=False,
    )

    assert [trace.name for trace in fig.data] == ["Target Consumption", "Total Sales"]
    assert fig.layout.legend.orientation == "v"
    assert fig.layout.legend.x == pytest.approx(1.02)
    assert fig.layout.legend.y == pytest.approx(1.0)


def test_plot_agent_timeseries_uses_panel_annotations_without_agent_prefixes():
    index = pd.RangeIndex(3, name="t")
    shallow = pd.DataFrame({"GDP_Expenditure": [1.0, 1.0, 1.0]}, index=index)
    country = SimpleNamespace(
        firms=SimpleNamespace(
            ts=_ts(
                {
                    "total_sales": [10.0, 11.0, 12.0],
                }
            )
        ),
        households=SimpleNamespace(
            ts=_ts(
                {
                    "target_consumption": [4.0, 5.0, 6.0],
                }
            )
        ),
    )
    model = SimpleNamespace(
        countries={"FRA": country},
        shallow_df_dict=lambda: {"FRA": shallow},
    )

    fig = plot_agent_timeseries(
        model=model,
        country_code="FRA",
        agent_type="firms",
        variables=[
            ["households.target_consumption", "firms.total_sales"],
            ["firms.total_sales"],
        ],
        panel_titles=["household panel", "firm panel"],
        no_cols=1,
        show_legend=True,
        show=False,
    )

    legend_annotations = [annotation for annotation in fig.layout.annotations if annotation.bgcolor]
    assert len(legend_annotations) == 2
    assert all("." not in annotation.text for annotation in legend_annotations)
    assert any("Target Consumption" in annotation.text for annotation in legend_annotations)
    assert all("Series" in annotation.text for annotation in legend_annotations)
    assert any("■" in annotation.text for annotation in legend_annotations)
    assert all(annotation.x > 0.9 for annotation in legend_annotations)
    assert fig.layout.legend.orientation is None


def test_plot_agent_timeseries_uses_readable_default_panel_titles():
    index = pd.RangeIndex(3, name="t")
    shallow = pd.DataFrame({"GDP_Expenditure": [1.0, 1.0, 1.0]}, index=index)
    country = SimpleNamespace(
        firms=SimpleNamespace(
            ts=_ts(
                {
                    "total_sales": [10.0, 11.0, 12.0],
                }
            )
        ),
        households=SimpleNamespace(
            ts=_ts(
                {
                    "target_consumption": [4.0, 5.0, 6.0],
                }
            )
        ),
    )
    model = SimpleNamespace(
        countries={"FRA": country},
        shallow_df_dict=lambda: {"FRA": shallow},
    )

    fig = plot_agent_timeseries(
        model=model,
        country_code="FRA",
        agent_type="firms",
        variables=[
            ["households.target_consumption", "firms.total_sales"],
            ["firms.total_sales"],
        ],
        no_cols=1,
        show_legend=True,
        show=False,
    )

    subplot_titles = [annotation.text for annotation in fig.layout.annotations if not annotation.bgcolor]
    assert any(title == "Target Consumption + Total Sales" for title in subplot_titles)
    assert any(title == "Total Sales" for title in subplot_titles)


def test_cumulative_insolvent_firms_by_sector_uses_expanded_sector_columns():
    df = pd.DataFrame(
        {
            "num_insolvent_firms_by_sector": [[1, 2], [3, 4], [5, 6]],
            "num_insolvent_firms_by_sector_A": [1, 3, 5],
            "num_insolvent_firms_by_sector_C": [2, 4, 6],
            "firm_insolvency_rate": [0.1, 0.2, 0.3],
        },
        index=pd.RangeIndex(3, name="t"),
    )

    cumulative = build_cumulative_insolvent_firms_by_sector_df(df)
    fig = plot_cumulative_insolvent_firms_by_sector(df, show=False)

    assert cumulative.columns.tolist() == ["A", "C"]
    assert cumulative["A"].tolist() == [1, 4, 9]
    assert cumulative["C"].tolist() == [2, 6, 12]
    assert [trace.name for trace in fig.data] == [
        "A: Agriculture, forestry and fishing",
        "C: Manufacturing",
    ]
    assert fig.layout.yaxis.title.text == "cumulative insolvent firms"


def test_employment_by_sector_plot_uses_labour_market_series_and_sector_labels():
    model = SimpleNamespace(
        countries={
            "FRA": SimpleNamespace(
                labour_market=SimpleNamespace(
                    ts=SimpleNamespace(num_employed_individuals_by_sector=[[10, 20], [11, 22], [12, 24]])
                ),
                firms=SimpleNamespace(industries=["A", "C"]),
            )
        }
    )

    employment = build_employment_by_sector_df(model, "FRA")
    fig = plot_employment_by_sector(model, "FRA", show=False)

    assert employment.columns.tolist() == ["A", "C"]
    assert employment["A"].tolist() == [10.0, 11.0, 12.0]
    assert employment["C"].tolist() == [20.0, 22.0, 24.0]
    assert [trace.name for trace in fig.data] == [
        "A: Agriculture, forestry and fishing",
        "C: Manufacturing",
    ]
    assert [trace.mode for trace in fig.data] == ["lines+markers", "lines+markers"]
    assert fig.layout.yaxis.title.text == "Number of Employed Individuals"


def test_production_by_sector_plot_sums_firm_production_and_uses_sector_labels():
    class DummyFirmTS:
        def __init__(self, production):
            self._production = production

        def historic(self, name):
            if name != "production":
                raise KeyError(name)
            return self._production

    # 2 sectors, 3 firms (two in A, one in C)
    production = [
        [1.0, 2.0, 3.0],
        [1.5, 2.5, 3.5],
        [2.0, 3.0, 4.0],
    ]
    industries = ["A", "C"]
    firm_industry_idx = np.array([0, 0, 1])
    model = SimpleNamespace(
        countries={
            "FRA": SimpleNamespace(
                firms=SimpleNamespace(
                    industries=industries,
                    states={"Industry": firm_industry_idx},
                    ts=DummyFirmTS(production),
                )
            )
        }
    )

    sector_production = build_production_by_sector_df(model, "FRA")
    fig = plot_production_by_sector(model, "FRA", show=False)

    assert sector_production.columns.tolist() == ["A", "C"]
    assert sector_production["A"].tolist() == [3.0, 4.0, 5.0]
    assert sector_production["C"].tolist() == [3.0, 3.5, 4.0]
    assert [trace.name for trace in fig.data] == [
        "A: Agriculture, forestry and fishing",
        "C: Manufacturing",
    ]
    assert fig.layout.yaxis.title.text == "Production"


def test_sector_tfp_investment_desired_mb_mc_ratio_plot_expands_mc_sector_series():
    index = pd.MultiIndex.from_product([[12, 13], [0, 1]], names=["seed", "time"])
    combined = pd.DataFrame(
        {
            "sector_tfp_investment_desired_mb_mc_ratio": [
                [0.9, 1.1],
                [1.0, 1.2],
                [0.8, 1.0],
                [0.95, 1.15],
            ]
        },
        index=index,
    )

    ratios = build_sector_tfp_investment_desired_mb_mc_ratio_df(combined, sector_labels=["A", "C"])
    fig = plot_sector_tfp_investment_desired_mb_mc_ratio(combined, sector_labels=["A", "C"], show=False)

    assert ratios.columns.tolist() == ["A", "C"]
    assert ratios.loc[(12, 1), "C"] == pytest.approx(1.2)
    assert [trace.name for trace in fig.data[:4]] == [
        "A: Agriculture, forestry and fishing",
        "A: Agriculture, forestry and fishing",
        "C: Manufacturing",
        "C: Manufacturing",
    ]
    assert [trace.showlegend for trace in fig.data[:4]] == [True, False, True, False]
    assert fig.layout.yaxis.title.text == "desired MB/MC ratio"


def test_sector_tfp_investment_desired_mb_mc_ratio_plot_reads_live_model_firms_ts():
    model = SimpleNamespace(
        countries={
            "FRA": SimpleNamespace(
                firms=SimpleNamespace(
                    industries=["A", "C"],
                    ts=SimpleNamespace(
                        sector_tfp_investment_desired_mb_mc_ratio=[
                            np.array([0.9, 1.1]),
                            np.array([1.0, 1.2]),
                        ]
                    ),
                )
            )
        }
    )

    ratios = build_sector_tfp_investment_desired_mb_mc_ratio_df(model, "FRA")
    fig = plot_sector_tfp_investment_desired_mb_mc_ratio(model, "FRA", show=False)

    assert ratios.columns.tolist() == ["A", "C"]
    assert ratios["C"].tolist() == [1.1, 1.2]
    assert [trace.name for trace in fig.data] == [
        "A: Agriculture, forestry and fishing",
        "C: Manufacturing",
    ]
    assert fig.data[0].x.tolist() == [0, 1]


def test_cpi_ppi_comparison_plots_use_fixed_colors_and_specific_legends():
    index = pd.RangeIndex(3, name="t")
    cpi_df = pd.DataFrame(
        {
            "cpi_transaction": [1.0, 1.1, 1.2],
            "cpi_fixed_basket": [1.0, 1.08, 1.15],
            "cpi_chained_basket": [1.0, 1.09, 1.18],
            "cpi_transaction_pop_change": [0.0, 0.1, 0.09],
            "cpi_fixed_basket_pop_change": [0.0, 0.08, 0.07],
            "cpi_chained_basket_pop_change": [0.0, 0.09, 0.08],
            "cpi_transaction_yoy_change": [0.0, 0.1, 0.2],
            "cpi_fixed_basket_yoy_change": [0.0, 0.08, 0.15],
            "cpi_chained_basket_yoy_change": [0.0, 0.09, 0.18],
        },
        index=index,
    )
    ppi_df = pd.DataFrame(
        {
            "model_ppi": [1.0, 1.1, 1.2],
            "fixed_ppi": [1.0, 1.08, 1.15],
            "chained_ppi": [1.0, 1.09, 1.18],
            "model_pop": [0.0, 0.1, 0.09],
            "fixed_pop": [0.0, 0.08, 0.07],
            "chained_pop": [0.0, 0.09, 0.08],
            "model_yoy": [0.0, 0.1, 0.2],
            "fixed_yoy": [0.0, 0.08, 0.15],
            "chained_yoy": [0.0, 0.09, 0.18],
        },
        index=index,
    )

    cpi_fig = plot_cpi_comparison(cpi_df, show=False)
    ppi_fig = plot_ppi_comparison(ppi_df, show=False)

    assert [trace.name for trace in cpi_fig.data] == [
        "Level: transaction CPI",
        "Level: fixed-basket CPI",
        "Level: chained-basket CPI",
        "PoP: transaction CPI",
        "PoP: fixed-basket CPI",
        "PoP: chained-basket CPI",
        "YoY: transaction CPI",
        "YoY: fixed-basket CPI",
        "YoY: chained-basket CPI",
    ]
    assert [trace.name for trace in ppi_fig.data] == [
        "Level: model PPI",
        "Level: fixed-basket PPI",
        "Level: chained-basket PPI",
        "PoP: model PPI",
        "PoP: fixed-basket PPI",
        "PoP: chained-basket PPI",
        "YoY: model PPI",
        "YoY: fixed-basket PPI",
        "YoY: chained-basket PPI",
    ]
    assert [trace.line.color for trace in cpi_fig.data[:3]] == ["#1f77b4", "#2ca02c", "#d62728"]
    assert [trace.line.color for trace in ppi_fig.data[:3]] == ["#1f77b4", "#2ca02c", "#d62728"]


@pytest.mark.parametrize("baseline_income", [200.0, 0.0, -1.0, np.nan])
def test_net_interest_receipts_use_paired_baseline_income(baseline_income):
    from src.visual_helpers import _prepare_irf_plot_panel

    rows = []
    for seed, income, change in [(2, 400.0, 8.0), (1, baseline_income, -10.0)]:
        for variable, baseline, delta in [
            ("household_deposit_interest", 0.001, change),
            ("household_income", income, 0.0),
        ]:
            rows.append(
                dict(
                    seed=seed,
                    shock_name="policy_rate_100bp",
                    horizon=0,
                    variable=variable,
                    baseline=baseline,
                    delta=delta,
                    pct_delta=12345.0,
                )
            )
    result = _prepare_irf_plot_panel(pd.DataFrame(rows))
    receipts = result.loc[result.variable == "household_deposit_interest"].set_index("seed").plot_value
    assert receipts.loc[2] == pytest.approx(-2.0)
    if baseline_income > 0.0:
        assert receipts.loc[1] == pytest.approx(5.0)
    else:
        assert np.isnan(receipts.loc[1])


def test_plot_irf_panels_compares_uncertainty_index_with_real_consumption_percent():
    panel = pd.DataFrame(
        [
            {
                "seed": 1,
                "shock_name": "common_uncertainty",
                "horizon": 0,
                "variable": variable,
                "pct_delta": pct_delta,
                "delta": delta,
            }
            for variable, delta, pct_delta in (
                ("income_uncertainty_delta", 0.02, np.nan),
                ("consumption_uncertainty_contribution", 0.004, np.nan),
                ("household_consumption", 100.0, 0.10),
                ("cpi", 0.02, 0.02),
            )
        ]
    )

    figures = plot_irf_panels(panel, ["common_uncertainty"], show=False)

    figure = next(fig for fig in figures if fig._suptitle.get_text().startswith("Income uncertainty:"))
    axes = figure.axes
    assert len(axes) == 2
    assert axes[0].get_title() == "Income uncertainty change (index units)"
    assert axes[1].get_title() == "Real hh consumption, %"
    assert axes[0].get_ylabel() == "Common income\nuncertainty"
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), [0.02])
    np.testing.assert_allclose(axes[1].lines[0].get_ydata(), [100 * (1.10 / 1.02 - 1)])
    for fig in figures:
        assert fig._suptitle.get_fontweight() == "bold"
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        column_titles = [ax.title.get_window_extent(renderer) for ax in fig.axes if ax.get_title()]
        assert fig._suptitle.get_window_extent(renderer).y0 > max(title.y1 for title in column_titles) + 8
    without_uncertainty = panel.loc[
        ~panel.variable.isin(["income_uncertainty_delta", "consumption_uncertainty_contribution"])
    ]
    macro_only = plot_irf_panels(without_uncertainty, ["common_uncertainty"], show=False)
    assert len(macro_only) == 1
    assert macro_only[0]._suptitle.get_text().startswith("Macro and investment:")


@pytest.mark.parametrize("historical", [False, True])
def test_macro_facility_diagnostics_sum_agents_preserve_t0_and_missing_history(historical):
    from unittest.mock import patch

    from config import OPERATING_FACILITY_COLUMNS
    from src.visual_helpers import plot_output

    index = pd.Index([0, 1, 2], name="t")
    shallow = pd.DataFrame({"GDP_Expenditure": [100.0] * 3}, index=index)
    fields = (
        "operating_revolving_interest_base",
        "received_operating_refinance_credit",
        "operating_revolving_current_draw",
        "operating_revolving_repayment",
        "operating_revolving_closing_balance",
        "operating_revolving_interest_paid",
    )
    history = {} if historical else {f: [[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]] for f in fields}
    country = SimpleNamespace(
        central_government=SimpleNamespace(ts=_ts({})),
        central_bank=SimpleNamespace(ts=_ts({})),
        banks=SimpleNamespace(ts=_ts({})),
        economy=SimpleNamespace(ts=_ts({})),
        firms=SimpleNamespace(industries=[], ts=_ts(history)),
    )
    model = SimpleNamespace(
        timestep=SimpleNamespace(increment=3),
        countries={"FRA": country},
        shallow_df_dict=lambda: {"FRA": shallow},
        get_country_gdp_components_df=lambda country_code: pd.DataFrame(index=index),
    )
    output = build_macro_output_df(model, "FRA")
    columns = [c for c in OPERATING_FACILITY_COLUMNS if c in output]
    if historical:
        assert columns == []
    else:
        assert columns == list(OPERATING_FACILITY_COLUMNS)
        for c in columns:
            np.testing.assert_array_equal(output[c], [0.0, 3.0, 7.0])
        with patch("plotly.graph_objects.Figure.show") as show:
            plot_output(output[columns], no_rows=2, no_cols=3, country_code="FRA")
        assert len(show.call_args.args) == 0
        show.assert_called_once()
