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
    plot_ppi_comparison,
    plot_production_by_sector,
    plot_sector_tfp_investment_desired_mb_mc_ratio,
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
                    "total_household_social_transfers": [5.0, 5.1, 5.2],
                }
            )
        ),
        central_bank=SimpleNamespace(ts=_ts({"policy_rate": [0.01, 0.011, 0.012]})),
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
    }
    assert expected_columns.issubset(output.columns)
    assert output["sectoral_growth"].tolist() == [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]]
    assert output["sectoral_growth_services"].tolist() == [0.02, 0.04, 0.06]
    assert output["num_insolvent_firms_by_sector"].tolist() == [[1, 2], [3, 4], [5, 6]]
    assert output["num_insolvent_firms_by_sector_agriculture"].tolist() == [1, 3, 5]
    assert output["sector_tfp_investment_desired_mb_mc_ratio"].tolist() == [[0.9, 1.1], [1.0, 1.2], [1.1, 1.3]]
    assert output["sector_tfp_investment_desired_mb_mc_ratio_services"].tolist() == [1.1, 1.2, 1.3]
    assert output["avg_tfp_multiplier"].tolist() == pytest.approx([1.1, 1.2, 1.3])

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
