import pathlib

import numpy as np
import pandas as pd
import pytest

import macro_data.processing.synthetic_population.hfcs_individual_tools as individual_tools
import macro_data.processing.synthetic_population.hfcs_synthetic_population as hfcs_population_module
from macro_data.configuration.countries import Country
from macro_data.processing.synthetic_population.hfcs_individual_tools import initial_unemployment_benefit_per_recipient
from macro_data.processing.synthetic_population.hfcs_synthetic_population import (
    CONVERT_HH_COLS,
    SyntheticHFCSPopulation,
    compute_notebook_household_accounts,
    sample_households,
)
from macro_data.readers import AGGREGATED_INDUSTRIES
from macro_data.readers.population_data.hfcs_reader import HFCSReader, var_mapping, var_numerical

PARENT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()


def test__initial_unemployment_benefit_uses_the_macro_recipient_baseline_without_recipients():
    assert (
        initial_unemployment_benefit_per_recipient(
            total_unemployment_benefits=12.0,
            n_unemployed=0,
            fallback_recipient_count=3,
        )
        == 4.0
    )


def test__initial_unemployment_benefit_preserves_the_recipient_budget():
    assert initial_unemployment_benefit_per_recipient(total_unemployment_benefits=12.0, n_unemployed=3) == 4.0


def test__unemployment_benefit_fallback_uses_active_labour_force(monkeypatch):
    individual_data = pd.DataFrame(
        {
            "Gender": [1, 2] * 4,
            "Age": [40] * 8,
            "Education": [3] * 8,
            "Labour Status": [1] * 4 + [5] * 4,
            "Employee Income": [0.0] * 8,
            "Corresponding Household ID": range(8),
            "Relation to Reference Person": [1] * 8,
        }
    )
    monkeypatch.setattr(individual_tools, "remove_outliers", lambda data, **_: data)
    monkeypatch.setattr(individual_tools, "fill_missing_gender", lambda data: data)
    monkeypatch.setattr(individual_tools, "fill_individual_age", lambda data: data)
    monkeypatch.setattr(individual_tools, "fill_individual_education", lambda data: data)
    monkeypatch.setattr(individual_tools, "fill_individual_labour_status", lambda data: data)
    monkeypatch.setattr(
        individual_tools,
        "set_individual_activity_status",
        lambda individual_data, **_: individual_data.assign(**{"Activity Status": [1] * 4 + [3] * 4}),
    )
    monkeypatch.setattr(
        individual_tools,
        "fill_individual_nace",
        lambda data, *_: data.assign(**{"Employment Industry": 0}),
    )
    monkeypatch.setattr(individual_tools, "fill_individual_employee_income", lambda data, **_: data)
    monkeypatch.setattr(
        individual_tools,
        "set_individual_unemployed_income",
        lambda data, **_: data.assign(**{"Income from Unemployment Benefits": 0.0}),
    )

    processed = individual_tools.process_individual_data(
        individual_data=individual_data,
        industries=["A"],
        scale=1,
        total_unemployment_benefits=100.0,
        unemployment_rate=0.2,
        participation_rate=0.5,
        n_firms_by_industry=[1],
    )

    np.testing.assert_allclose(processed["Unemployment Benefit Entitlement"], 100.0)


@pytest.mark.parametrize("total_unemployment_benefits", [np.nan, np.inf, -1.0])
def test__initial_unemployment_benefit_rejects_invalid_totals(total_unemployment_benefits):
    with pytest.raises(ValueError, match="finite non-negative"):
        initial_unemployment_benefit_per_recipient(
            total_unemployment_benefits=total_unemployment_benefits,
            n_unemployed=1,
        )


def test__compute_notebook_household_accounts_uses_notebook_definitions():
    data = {
        "Wealth in Deposits": [10.0],
        "HD1320c": [20.0],
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

    assert accounts.loc[0, "lfa"] == 30.0
    assert accounts.loc[0, "ifa"] == 8.0
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
    assert var_mapping["DHAQ01"] == "Country quintile, gross wealth, among households"
    assert "Value of Other Non-Business Real Estate" in var_numerical
    assert "Outstanding Balance of Mortgage Debt" in var_numerical
    assert "Outstanding Balance of Non-Mortgage Debt" in var_numerical


def test__hfcs_reader_preserves_public_and_private_pension_components():
    households = pd.DataFrame(
        {
            "Income from Pensions": [999.0, 999.0],
            "Public Pension Income": [10.0, 20.0],
            "Occupational and Private Pension Income": [3.0, 4.0],
            "Regular Social Transfers": [5.0, 6.0],
        },
        index=[1, 2],
    )
    individuals = pd.DataFrame(index=[])

    data = HFCSReader._add_household_income_diagnostics(households, individuals)

    assert data["Public Pension Income"].tolist() == [10.0, 20.0]
    assert data["Occupational and Private Pension Income"].tolist() == [3.0, 4.0]
    assert data["Pension Income"].tolist() == [13.0, 24.0]


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


def test__normalise_household_consumption_caps_saving_rates_at_one(monkeypatch):
    population = SyntheticHFCSPopulation.__new__(SyntheticHFCSPopulation)
    population.household_data = pd.DataFrame(
        {
            "Income": [100.0, 200.0, 300.0],
            "Saving Rate": [0.2, 0.4, 1.2],
        }
    )
    population.consumption_weights = np.array([1.0])
    population.saving_rates_model = None
    monkeypatch.setattr(hfcs_population_module, "fit_linear", lambda **_kwargs: None)

    population.normalise_household_consumption(
        iot_hh_consumption=np.array([140.0]),
        vat=0.0,
        positive_saving_rates_only=False,
    )

    assert np.all(population.household_data["Saving Rate"] <= 1.0)
    assert np.all(population.household_data["Consumption"] >= 0.0)
    assert population.household_data.loc[2, "Consumption"] > 0.0


def test__social_transfer_initialisation_uses_hfcs_public_pension_and_other_weights(monkeypatch):
    population = SyntheticHFCSPopulation.__new__(SyntheticHFCSPopulation)
    population.scale = 10
    population.yearly_factor = 4.0
    population.social_transfers_model = object()
    population.household_data = pd.DataFrame(
        {
            "Type": [1, 1],
            "Net Wealth": [0.0, 0.0],
            "Income": [1.0, 1.0],
            "Debt": [0.0, 0.0],
            "Income from Pensions": [13.0, 24.0],
            "Pension Income": [13.0, 24.0],
            "Public Pension Income": [10.0, 20.0],
            "Regular Social Transfers": [1.0, 3.0],
            "Social Transfer Income": [1.0, 3.0],
        }
    )
    monkeypatch.setattr(
        hfcs_population_module,
        "apply_iterative_imputer",
        lambda data, *_args, **_kwargs: data,
    )
    monkeypatch.setattr(hfcs_population_module, "fit_linear", lambda **_kwargs: None)

    population.set_household_social_transfers(total_social_transfers=100.0)

    assert np.allclose(
        population.household_data["Allocated Public Pension Benefits"],
        [100.0 * 10.0 / 34.0, 100.0 * 20.0 / 34.0],
    )
    assert np.allclose(
        population.household_data["Allocated Other Social Transfers"],
        [100.0 * 1.0 / 34.0, 100.0 * 3.0 / 34.0],
    )
    assert np.isclose(population.household_data["Regular Social Transfers"].sum(), 100.0)


def test__public_pension_household_source_is_mapped_to_retired_individuals(monkeypatch):
    population = SyntheticHFCSPopulation.__new__(SyntheticHFCSPopulation)
    population.social_transfers_model = object()
    population.household_data = pd.DataFrame(
        {
            "Type": [1, 1],
            "Net Wealth": [0.0, 0.0],
            "Income": [1.0, 1.0],
            "Debt": [0.0, 0.0],
            "Income from Pensions": [10.0, 20.0],
            "Pension Income": [10.0, 20.0],
            "Public Pension Income": [10.0, 20.0],
            "Regular Social Transfers": [1.0, 3.0],
            "Social Transfer Income": [1.0, 3.0],
            "Corresponding Individuals ID": [[0, 1], [2]],
        }
    )
    population.individual_data = pd.DataFrame(
        {
            "Corresponding Household ID": [0, 0, 1],
            "Is Retired": [True, False, True],
        }
    )
    monkeypatch.setattr(hfcs_population_module, "apply_iterative_imputer", lambda data, *_args, **_kwargs: data)
    monkeypatch.setattr(hfcs_population_module, "fit_linear", lambda **_kwargs: None)

    population.set_household_social_transfers(total_social_transfers=100.0)

    np.testing.assert_allclose(population.individual_data["Public Pension Weight"], [1.0 / 3.0, 0.0, 2.0 / 3.0])
    np.testing.assert_allclose(
        population.individual_data["Public Pension Benefits"],
        [500.0 / 17.0, 0.0, 1000.0 / 17.0],
    )
    np.testing.assert_allclose(
        population.household_data["Allocated Public Pension Benefits"],
        [500.0 / 17.0, 1000.0 / 17.0],
    )


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

    def test__zero_unemployed_initialisation_uses_benefit_entitlement(
        self, monkeypatch, readers, industry_data, exogenous_data
    ):
        original_process_individual_data = hfcs_population_module.process_individual_data

        def process_without_unemployed(*args, **kwargs):
            individual_data = original_process_individual_data(*args, **kwargs)
            individual_data["Activity Status"] = 1
            individual_data["Income from Unemployment Benefits"] = 0.0
            individual_data["Unemployment Benefit Entitlement"] = 4.0
            return individual_data

        monkeypatch.setattr(hfcs_population_module, "process_individual_data", process_without_unemployed)
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

        assert population.social_housing_rent == 2.0


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
