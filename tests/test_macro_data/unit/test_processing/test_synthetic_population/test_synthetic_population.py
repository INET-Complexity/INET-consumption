import pathlib

import numpy as np
import pandas as pd
import pytest

from macro_data.configuration.countries import Country
from macro_data.processing.synthetic_population.hfcs_synthetic_population import (
    CONVERT_HH_COLS,
    SyntheticHFCSPopulation,
    compute_notebook_household_accounts,
    sample_households,
)
from macro_data.readers import AGGREGATED_INDUSTRIES
from macro_data.readers.population_data.hfcs_reader import HFCSReader, var_mapping, var_numerical

PARENT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()


def test__compute_notebook_household_accounts_uses_notebook_definitions():
    data = {
        "Wealth in Deposits": [10.0],
        "Mutual Funds": [1.0],
        "Bonds": [2.0],
        "Value of Private Businesses": [100.0],
        "Shares": [3.0],
        "Managed Accounts": [4.0],
        "Money owed to Households": [5.0],
        "Other Assets": [6.0],
        "Voluntary Pension": [7.0],
        "Value of the Main Residence": [50.0],
        "DA1122": [8.0],
        "Outstanding Balance of HMR Mortgages": [9.0],
        "Outstanding Balance of Mortgages on other Properties": [10.0],
        "Outstanding Balance of Non-Mortgage Debt": [11.0],
    }

    accounts = compute_notebook_household_accounts(pd.DataFrame(data))

    assert accounts.loc[0, "lfa"] == 10.0
    assert accounts.loc[0, "ifa"] == 28.0
    assert accounts.loc[0, "ha"] == 58.0
    assert accounts.loc[0, "mr"] == 19.0
    assert accounts.loc[0, "db"] == 11.0
    assert accounts.loc[0, "nw"] == 66.0


def test__compute_notebook_household_accounts_does_not_mutate_input():
    data = pd.DataFrame(
        {
            "Wealth in Deposits": [10.0],
            "Mutual Funds": [1.0],
            "Bonds": [2.0],
            "Shares": [3.0],
            "Managed Accounts": [4.0],
            "Money owed to Households": [5.0],
            "Other Assets": [6.0],
            "Voluntary Pension": [7.0],
            "Value of the Main Residence": [50.0],
            "DA1122": [8.0],
            "Outstanding Balance of HMR Mortgages": [9.0],
            "Outstanding Balance of Mortgages on other Properties": [10.0],
            "Outstanding Balance of Credit Line": [1.0],
            "Outstanding Balance of Credit Card Debt": [2.0],
            "Outstanding Balance of other Non-Mortgage Loans": [3.0],
        }
    )
    original = data.copy(deep=True)

    accounts = compute_notebook_household_accounts(data)

    assert accounts.loc[0, "db"] == 6.0
    pd.testing.assert_frame_equal(data, original)


def test__compute_notebook_household_accounts_preserves_zero_aggregate_debt():
    data = {
        "Wealth in Deposits": [10.0],
        "Mutual Funds": [1.0],
        "Bonds": [2.0],
        "Shares": [3.0],
        "Managed Accounts": [4.0],
        "Money owed to Households": [5.0],
        "Other Assets": [6.0],
        "Voluntary Pension": [7.0],
        "Value of the Main Residence": [50.0],
        "DA1122": [8.0],
        "Outstanding Balance of HMR Mortgages": [9.0],
        "Outstanding Balance of Mortgages on other Properties": [10.0],
        "Outstanding Balance of Non-Mortgage Debt": [0.0],
        "Outstanding Balance of Credit Line": [1.0],
        "Outstanding Balance of Credit Card Debt": [2.0],
        "Outstanding Balance of other Non-Mortgage Loans": [3.0],
    }

    accounts = compute_notebook_household_accounts(pd.DataFrame(data))

    assert accounts.loc[0, "db"] == 0.0


def test__compute_notebook_household_accounts_preserves_zero_aggregate_mortgage_debt():
    data = {
        "Wealth in Deposits": [10.0],
        "Mutual Funds": [1.0],
        "Bonds": [2.0],
        "Shares": [3.0],
        "Managed Accounts": [4.0],
        "Money owed to Households": [5.0],
        "Other Assets": [6.0],
        "Voluntary Pension": [7.0],
        "Value of the Main Residence": [50.0],
        "DA1122": [8.0],
        "Outstanding Balance of Mortgage Debt": [0.0],
        "Outstanding Balance of HMR Mortgages": [9.0],
        "Outstanding Balance of Mortgages on other Properties": [10.0],
        "Outstanding Balance of Non-Mortgage Debt": [11.0],
    }

    accounts = compute_notebook_household_accounts(pd.DataFrame(data))

    assert accounts.loc[0, "mr"] == 0.0


def test__compute_notebook_household_accounts_uses_reader_notebook_source_columns():
    data = {
        "Wealth in Deposits": [10.0],
        "Mutual Funds": [1.0],
        "Bonds": [2.0],
        "Shares": [3.0],
        "Managed Accounts": [4.0],
        "Money owed to Households": [5.0],
        "Other Assets": [6.0],
        "Voluntary Pension": [7.0],
        "Value of the Main Residence": [50.0],
        "Value of Other Non-Business Real Estate": [8.0],
        "Outstanding Balance of HMR Mortgages": [9.0],
        "Outstanding Balance of Mortgages on other Properties": [10.0],
        "Outstanding Balance of Non-Mortgage Debt": [11.0],
    }

    accounts = compute_notebook_household_accounts(pd.DataFrame(data))

    assert accounts.loc[0, "ha"] == 58.0
    assert accounts.loc[0, "db"] == 11.0


def test__hfcs_reader_preserves_notebook_account_source_columns():
    assert var_mapping["DA1122"] == "Value of Other Non-Business Real Estate"
    assert var_mapping["DL1100"] == "Outstanding Balance of Mortgage Debt"
    assert var_mapping["DL1200"] == "Outstanding Balance of Non-Mortgage Debt"
    assert "Value of Other Non-Business Real Estate" in var_numerical
    assert "Outstanding Balance of Mortgage Debt" in var_numerical
    assert "Outstanding Balance of Non-Mortgage Debt" in var_numerical


def test__hfcs_reader_converts_notebook_account_source_columns(tmp_path):
    class ExchangeRates:
        def from_eur_to_lcu(self, country, year):
            return 2.0

    csv_path = tmp_path / "H1.csv"
    pd.DataFrame(
        {
            "SA0100": ["FR", "DE"],
            "ID": [1, 2],
            "DA1122": [8.0, 999.0],
            "DL1100": [9.0, 999.0],
            "DL1200": [10.0, 999.0],
        }
    ).to_csv(csv_path, index=False)

    data = HFCSReader.read_csv(
        path=csv_path,
        country_name="France",
        country_name_short="FR",
        year=2014,
        exchange_rates=ExchangeRates(),
    )

    assert data.loc[1, "Value of Other Non-Business Real Estate"] == 16.0
    assert data.loc[1, "Outstanding Balance of Mortgage Debt"] == 18.0
    assert data.loc[1, "Outstanding Balance of Non-Mortgage Debt"] == 20.0


def test__hfcs_reader_computes_windfall_income(tmp_path):
    class ExchangeRates:
        def from_eur_to_lcu(self, country, year):
            return 1.0

    csv_path = tmp_path / "H1.csv"
    pd.DataFrame(
        {
            "SA0100": ["FR"],
            "ID": [1],
            "HH0100": [1],
            "HH0201": [2012],
            "HH0202": [2010],
            "HH0203": [2013],
            "HH0301a": [1],
            "HH0302a": [1],
            "HH0303a": [1],
            "HH0401": [20000.0],
            "HH0402": [1000.0],
            "HH0403": [25000.0],
            "SA0200": [2014],
        }
    ).to_csv(csv_path, index=False)

    data = HFCSReader.read_csv(
        path=csv_path,
        country_name="France",
        country_name_short="FR",
        year=2014,
        exchange_rates=ExchangeRates(),
    )
    data = HFCSReader._compute_windfall_income(data, windfall_threshold=20000.0, default_year=2014)

    assert data.loc[1, "windfall_income"] == 1


def test__proxy_conversion_includes_notebook_account_source_columns():
    assert "Managed Accounts" in CONVERT_HH_COLS
    assert "Value of Other Non-Business Real Estate" in CONVERT_HH_COLS
    assert "Outstanding Balance of Mortgage Debt" in CONVERT_HH_COLS
    assert "Outstanding Balance of Non-Mortgage Debt" in CONVERT_HH_COLS


class TestSyntheticPopulation:
    def test__init(self, readers, configuration, industry_data, exogenous_data):
        france = Country("FRA")

        population = SyntheticHFCSPopulation.from_readers(
            readers=readers,
            country_name=france,
            year=2014,
            scale=10000,
            country_name_short=france.to_two_letter_code(),
            industries=AGGREGATED_INDUSTRIES,
            industry_data=industry_data[france],
            rent_as_fraction_of_unemployment_rate=0.5,
            total_unemployment_benefits=1000.0,
            quarter=1,
            exogenous_data=exogenous_data,
        )

        # Check if we have all the necessary fields
        for ind_field in [
            "Gender",
            "Age",
            "Education",
            "Activity Status",
            "Employment Industry",
            "Employee Income",
            "Income from Unemployment Benefits",
            "Income",
            "Corresponding Household ID",
        ]:
            assert ind_field in population.individual_data.columns
        for hh_field in [
            "Tenure Status of the Main Residence",
            "Rent Paid",
            "Number of Properties other than Household Main Residence",
            "Type",
            "Rental Income from Real Estate",
            "Income from Pensions",
            "Regular Social Transfers",
            "Value of the Main Residence",
            "Value of other Properties",
            "Value of Household Vehicles",
            "Value of Household Valuables",
            "Value of Self-Employment Businesses",
            "Wealth in Deposits",
            "Mutual Funds",
            "Bonds",
            "Value of Private Businesses",
            "Shares",
            "Managed Accounts",
            "Money owed to Households",
            "Other Assets",
            "Voluntary Pension",
            "Outstanding Balance of HMR Mortgages",
            "Outstanding Balance of Mortgages on other Properties",
            "Outstanding Balance of Credit Line",
            "Outstanding Balance of Credit Card Debt",
            "Outstanding Balance of other Non-Mortgage Loans",
            "Consumption of Consumer Goods/Services as a Share of Income",
            "Corresponding Individuals ID",
            "Wealth Other Real Assets",
            "Wealth in Real Assets",
            "Wealth in Other Financial Assets",
            "Wealth in Financial Assets",
            "Wealth",
            "Debt",
            "Net Wealth",
            "Employee Income",
            "Income",
        ]:
            assert hh_field in population.household_data.columns

        # Check individual gender
        assert np.all(population.individual_data["Gender"].isin([1, 2]))

        # Check individual age
        assert np.all(population.individual_data["Age"] >= 0)


@pytest.mark.parametrize("country", [Country("CAN"), Country("USA")])
def test__household_consumption(multic_readers, multic_industry_data, configuration, country, exogenous_data):
    industries = AGGREGATED_INDUSTRIES

    proxy_country = Country("FRA")
    year = 2014
    population_ratio = multic_readers.world_bank.get_population(
        country=country, year=year
    ) / multic_readers.world_bank.get_population(country=proxy_country, year=year)

    exch_rate_proxy_to_lcu = multic_readers.exchange_rates.from_eur_to_lcu(country, year)

    SyntheticHFCSPopulation.from_readers(
        readers=multic_readers,
        country_name=proxy_country,
        year=2014,
        scale=10000,
        country_name_short=proxy_country.to_two_letter_code(),
        industries=industries,
        industry_data=multic_industry_data[country],
        rent_as_fraction_of_unemployment_rate=0.5,
        total_unemployment_benefits=1000.0,
        population_ratio=population_ratio,
        exch_rate=exch_rate_proxy_to_lcu,
        exogenous_data=exogenous_data,
        quarter=1,
    )

    assert True


def test__household_sampling(readers):
    country_name = "FRA"
    year = 2014
    scale = 10_000
    n_households = int(readers.eurostat.number_of_households(country_name, year) / scale)
    hfcs_individuals_data = readers.hfcs[country_name].individuals_df
    hfcs_households_data = readers.hfcs[country_name].households_df

    household_selection, individual_selection = sample_households(
        hfcs_households_data, hfcs_individuals_data, n_households
    )

    assert household_selection.shape[0] == n_households
    assert individual_selection["New Household ID"].nunique() == n_households
    assert np.all(
        individual_selection.groupby(["New Household ID"])["Gender"].count()
        == household_selection["Corresponding Individuals ID"].apply(len)
    )

    large_households = household_selection["Corresponding Individuals ID"].apply(len) > 2

    sample = np.random.choice(household_selection.index[large_households], size=10)

    for i in sample:
        individuals = household_selection.loc[i, "Corresponding Individuals ID"]
        assert individual_selection.loc[individuals, "Corresponding Household ID"].nunique() == 1
        assert np.all(individual_selection.loc[individuals, "Corresponding Household ID"] == i)
    assert True
