import pathlib
from types import SimpleNamespace

import numpy as np
import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.processing.country_data import TaxData
from macro_data.processing.synthetic_central_government.default_synthetic_central_government import (
    DefaultSyntheticCGovernment,
)

PARENT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()


class TestSyntheticCentralGovernment:
    def test__create(self, readers):
        central_gov = DefaultSyntheticCGovernment.from_readers(readers=readers, country_name=Country("FRA"), year=2014)
        # Check if we have all the necessary fields
        for central_gov_field in [
            "Debt",
            "Total Unemployment Benefits",
            "Reader Non-Unemployment Social Benefits",
            "Public Pension Benefits",
            "Other Social Benefits",
        ]:
            assert central_gov_field in central_gov.central_gov_data.columns

        # Check if there are any missing values
        assert not np.any(pd.isna(central_gov.central_gov_data))

    def test__reader_non_unemployment_envelope_is_preserved_for_hfcs_component_split(self, readers):
        central_gov = DefaultSyntheticCGovernment.from_readers(
            readers=readers,
            country_name=Country("FRA"),
            year=2014,
        )
        assert np.isclose(
            central_gov.central_gov_data["Reader Non-Unemployment Social Benefits"].iloc[0],
            central_gov.central_gov_data["Other Social Benefits"].iloc[0],
        )
        assert central_gov.central_gov_data["Public Pension Benefits"].iloc[0] == 0.0

    def test__update_fields_grosses_up_net_employee_income_for_labour_taxes(self):
        central_gov = DefaultSyntheticCGovernment(
            country_name="FRA",
            year=2014,
            central_gov_data=pd.DataFrame({"Debt": [0.0], "Bank Equity Injection": [0.0]}),
            other_benefits_model=None,
            unemployment_benefits_model=None,
        )
        tax_data = TaxData(
            value_added_tax=0.0,
            export_tax=0.0,
            employer_social_insurance_tax=0.3,
            employee_social_insurance_tax=0.1,
            profit_tax=0.0,
            income_tax=0.2,
            capital_formation_tax=0.0,
            risk_premium=0.0,
        )
        net_employee_income = 72.0
        synthetic_population = SimpleNamespace(
            individual_data=pd.DataFrame({"Employee Income": [net_employee_income]}),
            household_data=pd.DataFrame(
                {
                    "Tenure Status of the Main Residence": [1],
                    "Rent Paid": [10.0],
                    "Income from Financial Assets": [20.0],
                }
            ),
            social_housing_rent=0.0,
        )
        synthetic_firms = SimpleNamespace(
            firm_data=pd.DataFrame({"Taxes paid on Production": [0.0], "Corporate Taxes Paid": [0.0]})
        )
        synthetic_banks = SimpleNamespace(bank_data=pd.DataFrame({"Corporate Taxes Paid": [0.0]}))
        industry_data = {
            "industry_vectors": pd.DataFrame(
                {
                    "Household Consumption in LCU": [0.0],
                    "Exports in LCU": [0.0],
                    "Household Capital Inputs in LCU": [0.0],
                }
            )
        }

        central_gov.update_fields(
            tax_data=tax_data,
            synthetic_population=synthetic_population,
            synthetic_firms=synthetic_firms,
            synthetic_banks=synthetic_banks,
            industry_data=industry_data,
        )

        net_factor = 1 - 0.1 - 0.2 * (1 - 0.1)
        gross_employee_income = net_employee_income / net_factor
        expected_employee_si = 0.1 * gross_employee_income
        expected_employer_si = 0.3 * gross_employee_income
        expected_income_tax = 0.2 * (1 - 0.1) * gross_employee_income + 0.2 * 10.0 + 0.2 * 20.0

        assert np.isclose(central_gov.central_gov_data["Employee SI Tax"].iloc[0], expected_employee_si)
        assert np.isclose(central_gov.central_gov_data["Employer SI Tax"].iloc[0], expected_employer_si)
        assert np.isclose(central_gov.central_gov_data["Income Taxes"].iloc[0], expected_income_tax)
        assert np.isclose(
            central_gov.central_gov_data["Revenue"].iloc[0],
            expected_employee_si + expected_employer_si + expected_income_tax,
        )
