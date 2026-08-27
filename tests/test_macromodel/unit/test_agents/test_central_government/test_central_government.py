import numpy as np
import pandas as pd
import pytest

from macromodel.agents.central_government.central_government_ts import create_central_government_timeseries
from macromodel.agents.central_government.func.debt_interest import (
    CurrentPolicyRateDebtInterest,
    SmoothedPolicyRateDebtInterest,
)
from macromodel.agents.central_government.func.social_benefits import DefaultSocialBenefitsSetter
from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.configurations import CentralGovernmentConfiguration
from macromodel.util.function_mapping import functions_from_model


class RecordingBenefitModel:
    def __init__(self, growth_ratio):
        self.growth_ratio = growth_ratio
        self.seen_features = []

    def predict(self, features):
        self.seen_features.append(features)
        return np.array([self.growth_ratio])


class TestCentralGovernment:
    def test__initial_unemployment_benefits_keep_a_baseline_without_recipients(self):
        data = {
            "Debt": [0.0],
            "Total Unemployment Benefits": [12.0],
            "Public Pension Benefits": [0.0],
            "Other Social Benefits": [0.0],
            "Taxes on Production": [0.0],
            "VAT": [0.0],
            "Capital Formation Taxes": [0.0],
            "Corporate Taxes": [0.0],
            "Export Taxes": [0.0],
            "Income Taxes": [0.0],
            "Rental Income Taxes": [0.0],
            "Employee SI Tax": [0.0],
            "Employer SI Tax": [0.0],
            "Taxes on Products": [0.0],
            "Total Social Housing Rent": [0.0],
            "Revenue": [0.0],
            "Bank Equity Injection": [0.0],
        }
        timeseries = create_central_government_timeseries(
            data=pd.DataFrame(data),
            number_of_unemployed_individuals=0,
            initial_unemployment_benefit=4.0,
        )

        assert timeseries.current("unemployment_benefits_by_individual")[0] == 4.0
        assert timeseries.current("total_unemployment_benefits")[0] == 0.0

    def test__initial_social_transfer_total_reconciles_all_components(self):
        data = {
            "Debt": [0.0],
            "Total Unemployment Benefits": [12.0],
            "Public Pension Benefits": [3.0],
            "Other Social Benefits": [4.0],
            "Taxes on Production": [0.0],
            "VAT": [0.0],
            "Capital Formation Taxes": [0.0],
            "Corporate Taxes": [0.0],
            "Export Taxes": [0.0],
            "Income Taxes": [0.0],
            "Rental Income Taxes": [0.0],
            "Employee SI Tax": [0.0],
            "Employer SI Tax": [0.0],
            "Taxes on Products": [0.0],
            "Total Social Housing Rent": [0.0],
            "Revenue": [0.0],
            "Bank Equity Injection": [0.0],
        }
        timeseries = create_central_government_timeseries(
            data=pd.DataFrame(data),
            number_of_unemployed_individuals=2,
            initial_unemployment_benefit=6.0,
        )

        assert timeseries.current("total_household_social_transfers")[0] == 19.0

    def test__create(self, test_central_government):
        assert test_central_government.country_name == "FRA"

    def test__central_government_states(self, test_central_government):
        assert test_central_government is not None
        for state in [
            "Value-added Tax",
            "Household Investment VAT Rate",
            "Export Tax",
            "Employer Social Insurance Tax",
            "Employee Social Insurance Tax",
            "Profit Tax",
            "Income Tax",
            "Household Capital Formation Tax",
            "Firm Capital Formation Tax",
            "Other Product Production Tax Rate",
            "Taxes Less Subsidies Rates",
        ]:
            assert state in test_central_government.states.keys()

    def test__central_government_ts(self, test_central_government):
        for ts_key in [
            "unemployment_benefits_by_individual",
            "total_other_benefits",
            "total_unemployment_benefits",
            "total_household_social_transfers",
            "interest_payments_on_debt",
            "debt_interest_rate",
        ]:
            assert ts_key in test_central_government.ts.get_keys()

    def test__distribute_unemployment_benefits_to_individuals(self, test_central_government):
        benefits = test_central_government.ts.current("unemployment_benefits_by_individual")
        assert np.allclose(
            test_central_government.distribute_unemployment_benefits_to_individuals(
                current_individual_activity_status=np.array([ActivityStatus.EMPLOYED, ActivityStatus.UNEMPLOYED]),
            ),
            np.array([0.0, benefits[0]]),
        )

    def test__update_benefits_uses_cpi_based_benefit_indexation_inflation(self, test_central_government):
        unemployment_model = RecordingBenefitModel(growth_ratio=1.1)
        transfers_model = RecordingBenefitModel(growth_ratio=1.2)
        test_central_government.functions["social_benefits"] = DefaultSocialBenefitsSetter()
        test_central_government.states["unemployment_benefits_model"] = unemployment_model
        test_central_government.states["other_benefits_model"] = transfers_model

        prev_unemployment_benefits = test_central_government.ts.current("unemployment_benefits_by_individual")[0]
        prev_other_benefits = test_central_government.ts.current("total_other_benefits")[0]

        test_central_government.update_benefits(
            historic_benefit_indexation_inflation=[np.array([0.10, 0.20])],
            exogenous_benefit_indexation_inflation=np.array([0.01, 0.02]),
            current_estimated_benefit_indexation_inflation=0.03,
            current_unemployment_rate=0.25,
            current_estimated_growth=0.0,
        )

        assert np.isclose(
            test_central_government.ts.current("unemployment_benefits_by_individual")[0],
            1.1 * prev_unemployment_benefits,
        )
        assert np.isclose(test_central_government.ts.current("total_other_benefits")[0], 1.2 * prev_other_benefits)
        assert unemployment_model.seen_features[-1]["Data CPI Inflation"].iloc[0] == 0.03
        assert transfers_model.seen_features[-1]["Data CPI Inflation"].iloc[0] == 0.03

    def test__compute_deficit_uses_realised_benefit_stock_once(self, test_central_government):
        current_cpi = 2.0
        current_ind_activity = np.array(
            [
                ActivityStatus.UNEMPLOYED,
                ActivityStatus.UNEMPLOYED,
                ActivityStatus.EMPLOYED,
            ]
        )
        unemployment_benefit = test_central_government.ts.current("unemployment_benefits_by_individual")[0]
        realised_public_pensions = 10.0
        realised_other_transfers = 20.0
        realised_necessity_support = 12.0
        realised_transfers = realised_public_pensions + realised_other_transfers + realised_necessity_support
        test_central_government.ts.total_public_pension_benefits.append([realised_public_pensions])
        test_central_government.ts.total_other_social_transfers.append([realised_other_transfers])
        test_central_government.ts.total_necessity_support.append([realised_necessity_support])
        test_central_government.ts.total_household_social_transfers.append([realised_transfers])
        test_central_government.ts.total_unemployment_benefits.append([2 * current_cpi * unemployment_benefit])
        current_government_spending = np.array([10.0, 20.0])
        interest_payments = 5.0
        revenue = test_central_government.ts.current("revenue")[0]

        deficit = test_central_government.compute_deficit(
            current_ind_activity=current_ind_activity,
            current_cpi=current_cpi,
            current_government_nominal_amount_spent=current_government_spending,
            interest_payments_on_debt=interest_payments,
        )

        expected_benefits = 2 * current_cpi * unemployment_benefit + realised_transfers
        expected_deficit = expected_benefits + current_government_spending.sum() + interest_payments - revenue
        assert np.isclose(deficit[0], expected_deficit)

    def test__compute_deficit_uses_realised_transfer_stock_including_stage5(self, test_central_government):
        current_cpi = 2.0
        current_ind_activity = np.array([ActivityStatus.EMPLOYED])
        current_government_spending = np.array([10.0])
        interest_payments = 5.0
        revenue = test_central_government.ts.current("revenue")[0]
        realised_public_pensions = 0.0
        realised_other_transfers = 30.0
        realised_necessity_support = 12.0
        realised_transfers = realised_public_pensions + realised_other_transfers + realised_necessity_support
        test_central_government.ts.total_public_pension_benefits.append([realised_public_pensions])
        test_central_government.ts.total_other_social_transfers.append([realised_other_transfers])
        test_central_government.ts.total_necessity_support.append([realised_necessity_support])
        test_central_government.ts.total_household_social_transfers.append([realised_transfers])
        test_central_government.ts.total_unemployment_benefits.append([0.0])

        deficit = test_central_government.compute_deficit(
            current_ind_activity=current_ind_activity,
            current_cpi=current_cpi,
            current_government_nominal_amount_spent=current_government_spending,
            interest_payments_on_debt=interest_payments,
        )

        expected_benefits = realised_transfers
        expected_deficit = expected_benefits + current_government_spending.sum() + interest_payments - revenue
        assert np.isclose(deficit[0], expected_deficit)

    def test__public_pensions_only_pay_eligible_weighted_recipients(self, test_central_government):
        test_central_government.ts.public_pension_benefits.append([90.0])

        payments = test_central_government.distribute_public_pension_benefits_to_individuals(
            retirement_eligibility=np.array([True, False, True]),
            public_pension_weights=np.array([2.0, 100.0, 1.0]),
        )

        np.testing.assert_allclose(payments, [60.0, 0.0, 30.0])
        assert np.isclose(payments.sum(), 90.0)

    # def test__compute_taxes_revenue_deficit_debt(self, test_central_government):
    #     test_central_government.compute_taxes(
    #         current_ind_employee_income=np.array([50.0, 100.0]),
    #         current_total_rent_paid=np.array([10.0, 30.0]),
    #         current_income_financial_assets=np.array([5.0, 5.0]),
    #         current_ind_activity=np.array([ActivityStatus.EMPLOYED, ActivityStatus.UNEMPLOYED]),
    #         current_ind_realised_cons=np.array([50.0, 100.0]),
    #         current_bank_profits=np.array([10.0]),
    #         current_firm_production=np.array([200.0]),
    #         current_firm_price=np.array([1.0]),
    #         current_firm_profits=np.array([20.0]),
    #         current_firm_industries=np.array([0]),
    #         current_household_new_real_wealth=np.array([15.0]),
    #         taxes_less_subsidies_rates=np.array([0.2]),
    #         current_total_exports=100.0,
    #     )
    #     test_central_government.ts["debt"] = np.array([50.0])
    #     test_central_government.ts["revenue"] = np.array([40.0])
    #     # assert test_central_government.compute_revenue(household_rent_paid_to_government=100.0) == pytest.approx(
    #     #     226.37, abs=1e-1
    #     # )

    def test__compute_taxes_grosses_up_net_employee_income_for_labour_taxes(self, test_central_government):
        test_central_government.states["Income Tax"] = 0.2
        test_central_government.states["Employee Social Insurance Tax"] = 0.1
        test_central_government.states["Employer Social Insurance Tax"] = 0.3
        test_central_government.states["Value-added Tax"] = 0.0
        test_central_government.states["Capital Formation Tax"] = 0.0
        test_central_government.states["Export Tax"] = 0.0
        test_central_government.states["Profit Tax"] = 0.0

        net_employee_income = 72.0
        unemployed_employee_income = 100.0
        net_factor = 1 - 0.1 - 0.2 * (1 - 0.1)
        gross_employee_income = net_employee_income / net_factor

        rent_paid = 10.0
        financial_income = np.array([5.0, 15.0])

        test_central_government.compute_taxes(
            current_ind_employee_income=np.array([net_employee_income, unemployed_employee_income]),
            current_total_rent_paid=rent_paid,
            current_income_financial_assets=financial_income,
            current_ind_activity=np.array([ActivityStatus.EMPLOYED, ActivityStatus.UNEMPLOYED]),
            current_ind_realised_cons=np.array([50.0, 100.0]),
            current_bank_profits=np.array([10.0]),
            current_firm_production=np.array([200.0]),
            current_firm_price=np.array([1.0]),
            current_firm_profits=np.array([20.0]),
            current_firm_industries=np.array([0]),
            current_household_new_real_wealth=np.array([15.0]),
            taxes_less_subsidies_rates=np.array([0.0]),
            current_total_exports=100.0,
        )

        expected_income_tax = (
            0.2 * (1 - 0.1) * gross_employee_income
            + 0.2 * financial_income.sum()
            + 0.2 * rent_paid
        )
        assert np.isclose(test_central_government.ts.current("taxes_income")[0], expected_income_tax)
        assert np.isclose(test_central_government.ts.current("taxes_employee_si")[0], 0.1 * gross_employee_income)
        assert np.isclose(test_central_government.ts.current("taxes_employer_si")[0], 0.3 * gross_employee_income)
        assert test_central_government.ts.current("taxes_rental_income")[0] == pytest.approx(0.2 * rent_paid)
        assert np.isclose(
            test_central_government.ts.current("diagnostic_taxes_rental_income")[0],
            0.2 * rent_paid,
        )

    def test__social_rent_is_a_government_receipt(self, test_central_government):
        expected_tax_revenue = sum(
            test_central_government.ts.current(name)[0]
            for name in (
                "taxes_production",
                "taxes_vat",
                "taxes_cf",
                "taxes_corporate_income",
                "taxes_exports",
                "taxes_income",
                "taxes_employee_si",
                "taxes_employer_si",
            )
        )

        revenue = test_central_government.compute_revenue(household_rent_paid_to_government=123.0)

        assert revenue == pytest.approx(expected_tax_revenue + 123.0)
        assert test_central_government.ts.current("total_rent_received")[0] == 123.0
        assert test_central_government.ts.current("diagnostic_total_rent_received")[0] == 123.0

    def test__compute_taxes_allocates_capital_formation_tax_to_households_and_firms(self, test_central_government):
        test_central_government.states["Household Capital Formation Tax"] = 0.0
        test_central_government.states["Firm Capital Formation Tax"] = 0.25
        test_central_government.states["Value-added Tax"] = 0.0
        test_central_government.states["Export Tax"] = 0.0
        test_central_government.states["Profit Tax"] = 0.0
        test_central_government.states["Income Tax"] = 0.0
        test_central_government.states["Employee Social Insurance Tax"] = 0.0
        test_central_government.states["Employer Social Insurance Tax"] = 0.0

        test_central_government.compute_taxes(
            current_ind_employee_income=np.array([0.0]),
            current_total_rent_paid=0.0,
            current_income_financial_assets=np.array([0.0]),
            current_ind_activity=np.array([ActivityStatus.EMPLOYED]),
            current_ind_realised_cons=np.array([0.0]),
            current_bank_profits=np.array([0.0]),
            current_firm_production=np.array([0.0]),
            current_firm_price=np.array([1.0]),
            current_firm_profits=np.array([0.0]),
            current_firm_industries=np.array([0]),
            current_household_new_real_wealth=np.array([100.0]),
            taxes_less_subsidies_rates=np.array([0.0]),
            current_total_exports=0.0,
            current_firm_capital_formation=40.0,
        )

        assert np.isclose(test_central_government.ts.current("taxes_cf")[0], 10.0)

    def test__compute_taxes_applies_configured_vat_to_household_investment(self, test_central_government):
        test_central_government.states["Value-added Tax"] = 0.13
        test_central_government.states["Household Investment VAT Rate"] = 0.13
        for key in [
            "Household Capital Formation Tax",
            "Firm Capital Formation Tax",
            "Export Tax",
            "Profit Tax",
            "Income Tax",
            "Employee Social Insurance Tax",
            "Employer Social Insurance Tax",
        ]:
            test_central_government.states[key] = 0.0

        test_central_government.compute_taxes(
            current_ind_employee_income=np.array([0.0]),
            current_total_rent_paid=0.0,
            current_income_financial_assets=np.array([0.0]),
            current_ind_activity=np.array([ActivityStatus.EMPLOYED]),
            current_ind_realised_cons=np.array([100.0]),
            current_bank_profits=np.array([0.0]),
            current_firm_production=np.array([0.0]),
            current_firm_price=np.array([1.0]),
            current_firm_profits=np.array([0.0]),
            current_firm_industries=np.array([0]),
            current_household_new_real_wealth=np.array([40.0]),
            taxes_less_subsidies_rates=np.array([0.0]),
            current_total_exports=0.0,
        )

        assert np.isclose(test_central_government.ts.current("taxes_vat")[0], 18.2)

    def test__compute_taxes_uses_configured_flat_product_production_rate(self, test_central_government):
        test_central_government.states["Other Product Production Tax Rate"] = 0.1
        test_central_government.states["Value-added Tax"] = 0.0
        test_central_government.states["Household Capital Formation Tax"] = 0.0
        test_central_government.states["Firm Capital Formation Tax"] = 0.0
        test_central_government.states["Export Tax"] = 0.0
        test_central_government.states["Profit Tax"] = 0.0
        test_central_government.states["Income Tax"] = 0.0
        test_central_government.states["Employee Social Insurance Tax"] = 0.0
        test_central_government.states["Employer Social Insurance Tax"] = 0.0

        test_central_government.compute_taxes(
            current_ind_employee_income=np.array([0.0]),
            current_total_rent_paid=0.0,
            current_income_financial_assets=np.array([0.0]),
            current_ind_activity=np.array([ActivityStatus.EMPLOYED]),
            current_ind_realised_cons=np.array([0.0]),
            current_bank_profits=np.array([0.0]),
            current_firm_production=np.array([200.0]),
            current_firm_price=np.array([2.0]),
            current_firm_profits=np.array([0.0]),
            current_firm_industries=np.array([0]),
            current_household_new_real_wealth=np.array([0.0]),
            taxes_less_subsidies_rates=np.array([0.0]),
            current_total_exports=0.0,
        )

        assert np.isclose(test_central_government.ts.current("taxes_production")[0], 40.0)

    def test__reconcile_initial_capital_tax_updates_product_tax_and_revenue(self, test_central_government):
        test_central_government.states["Household Capital Formation Tax"] = 0.0
        test_central_government.states["Firm Capital Formation Tax"] = 0.25
        initial_products = test_central_government.ts.current("taxes_on_products")[0]
        initial_revenue = test_central_government.ts.current("revenue")[0]
        initial_capital_tax = test_central_government.ts.current("taxes_cf")[0]

        test_central_government.reconcile_initial_capital_formation_tax(
            current_household_investment=np.array([100.0]),
            current_firm_capital_formation=40.0,
        )

        delta = 10.0 - initial_capital_tax
        assert np.isclose(test_central_government.ts.current("taxes_cf")[0], 10.0)
        assert np.isclose(test_central_government.ts.current("taxes_on_products")[0], initial_products + delta)
        assert np.isclose(test_central_government.ts.current("revenue")[0], initial_revenue + delta)

    def test__reconcile_initial_vat_updates_product_tax_and_revenue(self, test_central_government):
        test_central_government.states["Value-added Tax"] = 0.13
        test_central_government.states["Household Investment VAT Rate"] = 0.13
        initial_products = test_central_government.ts.current("taxes_on_products")[0]
        initial_revenue = test_central_government.ts.current("revenue")[0]
        initial_vat = test_central_government.ts.current("taxes_vat")[0]

        test_central_government.reconcile_initial_vat(
            current_household_consumption_before_vat=100.0,
            current_household_investment=np.array([40.0]),
        )

        delta = 18.2 - initial_vat
        assert np.isclose(test_central_government.ts.current("taxes_vat")[0], 18.2)
        assert np.isclose(test_central_government.ts.current("taxes_on_products")[0], initial_products + delta)
        assert np.isclose(test_central_government.ts.current("revenue")[0], initial_revenue + delta)

    def test__current_policy_rate_debt_interest_preserves_legacy_rule(self):
        rule = CurrentPolicyRateDebtInterest()
        assert (
            rule.compute_interest_rate(
                current_policy_rate=0.03,
                previous_debt_interest_rate=0.01,
                time_unit=3,
            )
            == 0.03
        )

    def test__current_policy_rate_debt_interest_ignores_stale_parameters(self):
        rule = CurrentPolicyRateDebtInterest(average_maturity_years=9.0)
        assert (
            rule.compute_interest_rate(
                current_policy_rate=0.03,
                previous_debt_interest_rate=0.01,
                time_unit=3,
            )
            == 0.03
        )

    def test__smoothed_policy_rate_debt_interest_smooths_policy_rate(self):
        rule = SmoothedPolicyRateDebtInterest(smoothing=0.9)
        assert (
            rule.compute_interest_rate(
                current_policy_rate=0.05,
                previous_debt_interest_rate=0.02,
                time_unit=3,
            )
            == 0.023
        )

    def test__smoothed_policy_rate_debt_interest_uses_maturity_and_time_unit(self):
        rule = SmoothedPolicyRateDebtInterest(average_maturity_years=10.0)
        assert np.isclose(
            rule.compute_interest_rate(
                current_policy_rate=0.05,
                previous_debt_interest_rate=0.01,
                time_unit=3,
            ),
            0.011,
        )

    def test__smoothed_policy_rate_debt_interest_initializes_from_policy_rate(self):
        rule = SmoothedPolicyRateDebtInterest(average_maturity_years=10.0)
        assert (
            rule.compute_interest_rate(
                current_policy_rate=0.05,
                previous_debt_interest_rate=np.nan,
                time_unit=3,
            )
            == 0.05
        )

    def test__debt_interest_rule_is_loaded_from_central_government_config(self):
        config = CentralGovernmentConfiguration()
        config.functions.debt_interest.name = "SmoothedPolicyRateDebtInterest"
        config.functions.debt_interest.parameters = {"average_maturity_years": 10.0}

        functions = functions_from_model(
            model=config.functions,
            loc="macromodel.agents.central_government",
        )

        assert isinstance(functions["debt_interest"], SmoothedPolicyRateDebtInterest)
        assert functions["debt_interest"].average_maturity_years == 10.0
