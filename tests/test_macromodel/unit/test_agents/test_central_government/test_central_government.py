import numpy as np

from macromodel.agents.central_government.func.debt_interest import (
    CurrentPolicyRateDebtInterest,
    SmoothedPolicyRateDebtInterest,
)
from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.configurations import CentralGovernmentConfiguration
from macromodel.util.function_mapping import functions_from_model


class TestCentralGovernment:
    def test__create(self, test_central_government):
        assert test_central_government.country_name == "FRA"

    def test__central_government_states(self, test_central_government):
        assert test_central_government is not None
        for state in [
            "Value-added Tax",
            "Export Tax",
            "Employer Social Insurance Tax",
            "Employee Social Insurance Tax",
            "Profit Tax",
            "Income Tax",
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
