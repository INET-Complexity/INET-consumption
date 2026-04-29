import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

RUN_MODEL_PATH = Path(__file__).resolve().parents[2] / "run_model"
if str(RUN_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL_PATH))

from src.visual_helpers import build_macro_output_df  # noqa: E402


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
        "vacancy_rate": [0.03, 0.031, 0.032],
        "vacancy_rate_growth": [0.0, 0.02, 0.02],
        "job_reallocation_rate": [0.04, 0.041, 0.042],
        "job_reallocation_rate_growth": [0.0, 0.03, 0.03],
        "firm_insolvency_rate": [0.001, 0.002, 0.003],
        "bank_insolvency_rate": [0.0, 0.0, 0.01],
        "household_insolvency_rate": [0.01, 0.011, 0.012],
        "total_growth": [0.0, 0.02, 0.03],
        "estimated_growth": [0.01, 0.02, 0.03],
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
        firms=SimpleNamespace(industries=["agriculture", "services"]),
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
        "vacancy_rate",
        "vacancy_rate_growth",
        "job_reallocation_rate",
        "job_reallocation_rate_growth",
        "firm_insolvency_rate",
        "bank_insolvency_rate",
        "household_insolvency_rate",
        "total_growth",
        "estimated_growth",
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
        "num_insolvent_firms_by_sector_agriculture",
        "num_insolvent_firms_by_sector_services",
        "npl_firm_loans",
        "npl_hh_cons_loans",
    }
    assert expected_columns.issubset(output.columns)
    assert output["sectoral_growth_services"].tolist() == [0.02, 0.04, 0.06]
    assert output["num_insolvent_firms_by_sector_agriculture"].tolist() == [1, 3, 5]

    clutter_columns = {
        "revenue",
        "policy_rate",
        "government expenditure",
        "expected gdp growth",
        "unemployment rate",
        "average_interest_rates_on_short_term_firm_loans",
    }
    assert output.columns.intersection(clutter_columns).empty
