"""Reader-to-runtime interest-rate reconciliation diagnostics."""

from __future__ import annotations

from typing import Any

import pandas as pd

from macro_data.readers import AGGREGATED_INDUSTRIES, ALL_INDUSTRIES
from macro_data.readers.default_readers import DataReaders


def _scalar(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.iloc[0, 0]
    if isinstance(value, pd.Series):
        return value.iloc[0]
    return value


def build_data_readers_for_run(prepared, *, single_icio_survey: bool = True) -> DataReaders:
    """Construct the raw-data reader bundle used by rate diagnostics."""
    data_config = prepared.data_config
    country_code = prepared.cfg.country_iso3
    country_names = [
        country
        for country in data_config.countries
        if (country.value if hasattr(country, "value") else str(country)) == country_code
    ]
    aggregate_industries = getattr(data_config, "aggregate_industries", False)
    industries = AGGREGATED_INDUSTRIES if aggregate_industries else ALL_INDUSTRIES
    scale_dict = {
        (country.value if hasattr(country, "value") else str(country)): data_config.country_configs[country].scale
        for country in country_names
    }
    proxy_country_dict = {}
    for country in data_config.countries:
        code = country.value if hasattr(country, "value") else str(country)
        country_config = data_config.country_configs[country]
        is_eu = getattr(country_config, "is_eu_country", None) or getattr(country, "is_eu_country", False)
        proxy = getattr(country_config, "eu_proxy_country", None)
        if not is_eu and proxy:
            proxy_country_dict[code] = proxy

    return DataReaders.from_raw_data(
        raw_data_path=prepared.raw_data_path,
        country_names=country_names,
        simulation_year=data_config.year,
        scale_dict=scale_dict,
        industries=industries,
        aggregate_industries=aggregate_industries,
        prune_date=data_config.prune_date,
        force_single_hfcs_survey=False,
        single_icio_survey=single_icio_survey,
        proxy_country_dict=proxy_country_dict,
        use_disagg_can_2014_reader=data_config.can_disaggregation,
        use_provincial_can_reader=False,
        regions_dict=None,
        allow_missing_emissions=False,
    )


def build_interest_rate_comparison(
    readers,
    data,
    *,
    country_code: str,
    year: int,
    quarter: int,
    time_unit: int,
) -> pd.DataFrame:
    """Compare annual reader rates with per-period synthetic-data rates."""
    if time_unit <= 0 or 12 % time_unit:
        raise ValueError("time_unit must be a positive divisor of 12.")
    periods_per_year = 12 // time_unit
    quarter_key = f"{year}-Q{quarter}"
    synthetic_country = data.synthetic_countries[country_code]
    banks = synthetic_country.banks
    central_bank_data = synthetic_country.central_bank.central_bank_data

    policy = float(_scalar(readers.policy_rates.get_policy_rates(country_code).loc[quarter_key, ["Policy Rate"]]))
    firm = float(_scalar(readers.ecb_reader.get_firm_rates(country_code).loc[quarter_key]))
    household = float(_scalar(readers.ecb_reader.get_household_consumption_rates(country_code).loc[quarter_key]))
    mortgage = float(_scalar(readers.ecb_reader.get_household_mortgage_rates(country_code).loc[quarter_key]))
    rows = [
        ["policy rate from reader", policy, policy / periods_per_year, None],
        [
            "central bank stored policy_rate",
            None,
            policy / periods_per_year,
            float(_scalar(central_bank_data[["policy_rate"]])) if "policy_rate" in central_bank_data else None,
        ],
        [
            "central bank stored smooth_policy_rate",
            None,
            policy / periods_per_year,
            float(_scalar(central_bank_data[["smooth_policy_rate"]]))
            if "smooth_policy_rate" in central_bank_data
            else None,
        ],
        [
            "firm deposit rate",
            None,
            policy / periods_per_year,
            float(_scalar(banks.bank_data[["Interest Rates on Firm Deposits"]])),
        ],
        [
            "household deposit rate",
            None,
            policy / periods_per_year,
            float(_scalar(banks.bank_data[["Interest Rates on Household Deposits"]])),
        ],
        ["firm loan rate", firm, firm / periods_per_year, float(banks.firm_rate)],
        ["household consumption loan rate", household, household / periods_per_year, float(banks.hh_consumption_rate)],
        ["mortgage rate", mortgage, mortgage / periods_per_year, float(banks.hh_mortgage_rate)],
    ]
    frame = pd.DataFrame(rows, columns=["rate", "raw_annual", "expected_per_period", "stored_in_data"])
    frame["stored_minus_expected"] = frame["stored_in_data"] - frame["expected_per_period"]
    frame.attrs.update(country_code=country_code, quarter=quarter_key, periods_per_year=periods_per_year)
    return frame


def build_initial_policy_rate_comparison(
    readers,
    data,
    *,
    country_code: str,
    year: int,
    quarter: int,
    time_unit: int,
) -> pd.DataFrame:
    """Compare the pre-start policy rate with initialized policy/deposit rates."""
    if time_unit <= 0 or 12 % time_unit:
        raise ValueError("time_unit must be a positive divisor of 12.")
    periods_per_year = 12 // time_unit
    start_period = pd.Period(f"{year}Q{quarter}", freq="Q")
    initial_period = start_period - 1
    initial_key = f"{initial_period.year}-Q{initial_period.quarter}"
    synthetic_country = data.synthetic_countries[country_code]
    central_bank_data = synthetic_country.central_bank.central_bank_data
    banks = synthetic_country.banks
    policy_annual = float(
        readers.policy_rates.get_policy_rates(country_code).loc[initial_key, ["Policy Rate"]].iloc[0, 0]
    )
    policy_period = policy_annual / periods_per_year
    frame = pd.DataFrame(
        [
            ["pre-start policy rate from reader", policy_annual, policy_period, None],
            [
                "stored smooth_policy_rate",
                None,
                policy_period,
                float(central_bank_data["smooth_policy_rate"].iloc[0])
                if "smooth_policy_rate" in central_bank_data
                else None,
            ],
            [
                "stored legacy policy_rate",
                None,
                policy_period,
                float(central_bank_data["policy_rate"].iloc[0]) if "policy_rate" in central_bank_data else None,
            ],
            ["firm deposit rate", None, policy_period, float(banks.bank_data["Interest Rates on Firm Deposits"].iloc[0])],
            [
                "household deposit rate",
                None,
                policy_period,
                float(banks.bank_data["Interest Rates on Household Deposits"].iloc[0]),
            ],
        ],
        columns=["rate", "raw_annual", "expected_per_period", "stored_in_data"],
    )
    frame["stored_minus_expected"] = frame["stored_in_data"] - frame["expected_per_period"]
    frame.attrs.update(country_code=country_code, start=str(start_period), initial_quarter=initial_key)
    return frame
