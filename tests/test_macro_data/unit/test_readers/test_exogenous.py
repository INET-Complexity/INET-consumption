import numpy as np
import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.readers.exogenous_data import (
    ExogenousCountryData,
    convert_growth_rates_to_model_period,
    convert_levels_to_model_period,
)


def test_convert_growth_rates_to_model_period_quarterly_noop():
    data = pd.DataFrame({"GDP": [0.03, 0.06]}, index=pd.to_datetime(["2020-01-01", "2020-04-01"]))

    converted = convert_growth_rates_to_model_period(data, time_unit=3)

    pd.testing.assert_frame_equal(converted, data)


def test_convert_growth_rates_to_model_period_monthly_compounds_to_quarterly_rate():
    data = pd.DataFrame({"GDP": [0.331]}, index=pd.to_datetime(["2020-01-01"]))

    converted = convert_growth_rates_to_model_period(data, time_unit=1)

    expected_monthly_rate = (1.331 ** (1.0 / 3.0)) - 1.0
    assert list(converted.index) == list(pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]))
    np.testing.assert_allclose(converted["GDP"].to_numpy(), expected_monthly_rate)
    np.testing.assert_allclose((1.0 + converted["GDP"]).prod() - 1.0, 0.331)


def test_convert_growth_rates_to_model_period_annual_compounds_quarters():
    data = pd.DataFrame(
        {"GDP": [0.1, 0.1, 0.1, 0.1]},
        index=pd.to_datetime(["2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01"]),
    )

    converted = convert_growth_rates_to_model_period(data, time_unit=12)

    assert list(converted.index) == list(pd.to_datetime(["2020-01-01"]))
    np.testing.assert_allclose(converted["GDP"].iloc[0], (1.1**4) - 1.0)


def test_convert_levels_to_model_period_interpolates_monthly_levels():
    data = pd.DataFrame({"Unemployment Rate (Value)": [0.03, 0.06]}, index=pd.to_datetime(["2020-01-01", "2020-04-01"]))

    converted = convert_levels_to_model_period(data, time_unit=1)

    expected = np.array([0.03, 0.04, 0.05, 0.06, 0.06, 0.06])
    assert list(converted.index) == list(
        pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01", "2020-06-01"])
    )
    np.testing.assert_allclose(converted["Unemployment Rate (Value)"].to_numpy(), expected)


class TestExogenous:
    def test__national_accounts_growth_uses_oecd_for_components_missing_from_imf(self, readers):
        country = Country("FRA")

        merged_growth = readers.get_national_accounts_growth(country)
        oecd_growth = readers.oecd_econ.get_na_growth_rates(country)
        imf_growth = readers.imf_reader.get_na_growth_rates(country)

        oecd_only_columns = {
            "Compensation of Employees",
            "Exports",
            "Gross Operating Surplus and Mixed Income",
            "Gross Value Added",
            "Gross Value Added - A",
            "Gross Value Added - B, C, D, E",
            "Gross Value Added - C",
            "Gross Value Added - F",
            "Gross Value Added - G, H, I",
            "Gross Value Added - G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U",
            "Gross Value Added - J",
            "Gross Value Added - K",
            "Gross Value Added - L",
            "Gross Value Added - M, N",
            "Gross Value Added - O, P, Q",
            "Gross Value Added - R, S, T, U",
            "HH Cons",
            "Imports",
            "Taxes less Subsidies on Production",
        }

        assert oecd_only_columns.issubset(oecd_growth.columns)
        assert oecd_only_columns.isdisjoint(imf_growth.columns)

        for column in oecd_only_columns:
            assert merged_growth[column].equals(oecd_growth.loc[merged_growth.index, column])

    def test__nominal_and_real_hh_to_gdp_ratios_are_consistent(self, readers, industry_data):
        country = Country("FRA")
        data = ExogenousCountryData.from_data_readers(
            country_name=country,
            readers=readers,
            year=2014,
            quarter=1,
            industry_vectors=industry_data[country]["industry_vectors"],
        )

        nominal_ratio = data.national_accounts["Household Consumption (Value)"] / data.national_accounts["GDP (Value)"]
        real_ratio = (
            data.national_accounts["Real Household Consumption (Value)"] / data.national_accounts["Real GDP (Value)"]
        )

        assert np.allclose(nominal_ratio, real_ratio, equal_nan=True)

    def test__exogenous(self, readers, industry_data):
        country = Country("FRA")
        data = ExogenousCountryData.from_data_readers(
            country_name=country,
            readers=readers,
            year=2014,
            quarter=1,
            industry_vectors=industry_data[country]["industry_vectors"],
        )

        assert data.inflation.shape[0] > 0

        calibration_data = data.get_calibration_data(2014, 1)

        assert (calibration_data[("FRA", "HPI (Value)")].dropna() > 0).all()
