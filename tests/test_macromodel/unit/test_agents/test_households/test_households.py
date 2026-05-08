import numpy as np
import pandas as pd


class TestHouseholds:
    def test__create(self, test_households):
        assert test_households.country_name == "FRA"

    def test__households_states(self, test_households):
        assert test_households is not None
        for state in [
            "saving_rates_model",
            "social_transfers_model",
            "average_saving_rate",
            "Type",
            "Corresponding Bank ID",
            "Corresponding Inhabited House ID",
            "Corresponding Property Owner",
            "Tenure Status of the Main Residence",
            "corr_individuals",
            "corr_renters",
        ]:
            assert state in test_households.states.keys()


    def test__mortgage_target_ignores_rentals(self, test_households):
        n_households = test_households.ts.current("n_households")

        # Make the mortgage target depend only on target_house_price.
        test_households.ts.override_current(
            "target_consumption",
            np.zeros_like(test_households.ts.current("target_consumption")),
        )
        test_households.ts.override_current(
            "expected_income",
            np.zeros_like(test_households.ts.current("expected_income")),
        )
        test_households.ts.override_current(
            "rent",
            np.zeros_like(test_households.ts.current("rent")),
        )
        test_households.ts.override_current(
            "wealth_financial_assets",
            np.zeros_like(test_households.ts.current("wealth_financial_assets")),
        )

        current_sales = pd.DataFrame(
            {
                "sales_types": ["Rental", "Sell"],
                "buyer_id": [0, 1],
                "price_or_rent": [100.0, 100000.0],
            }
        )
        test_households.compute_target_credit(current_sales=current_sales)

        target_mortgage = test_households.ts.current("target_mortgage")
        assert target_mortgage.shape == (n_households,)
        assert target_mortgage[0] == 0.0
        assert target_mortgage[1] == 100000.0
    # def test__households_ts(self, test_households):
    #     for ts_key in [
    #         "n_households",
    #         "target_consumption_before_ce",
    #         "target_consumption_ce",
    #         "target_consumption",
    #         "amount_bought",
    #         "consumption",
    #         "investment_in_other_real_assets",
    #         "income",
    #         "income_employee",
    #         "income_social_transfers",
    #         "income_rental",
    #         "price_paid_for_property",
    #         "rent",
    #         "max_price_willing_to_pay",
    #         "max_rent_willing_to_pay",
    #         "wealth",
    #         "wealth_real_assets",
    #         "wealth_main_residence",
    #         "wealth_other_properties",
    #         "wealth_other_real_assets",
    #         "wealth_deposits",
    #         "wealth_other_financial_assets",
    #         "wealth_financial_assets",
    #         "payday_loan_debt",
    #         "consumption_expansion_loan_debt",
    #         "mortgage_debt",
    #         "debt",
    #         "net_wealth",
    #         "target_payday_loans",
    #         "received_payday_loans",
    #         "target_consumption_expansion_loans",
    #         "received_consumption_expansion_loans",
    #         "target_mortgage",
    #         "received_mortgages",
    #         "debt_installments",
    #         "interest_paid_on_deposits",
    #         "interest_paid_on_loans",
    #         "interest_paid",
    #     ]:
    #         assert ts_key in test_households.ts.get_keys()
    #
    # def test__get_saving_rates_by_household(self, test_households):
    #     assert np.allclose(test_households.get_saving_rates_by_household(), np.full(18, 0.2))
    #
    # def test__get_social_transfers_by_household(self, test_households):
    #     assert np.allclose(
    #         test_households.compute_social_transfer_income(
    #             total_other_social_transfers=1000.0,
    #             central_government_init={
    #                 "functions": {
    #                     "household_social_transfers": {
    #                         "parameters": {
    #                             "independents": {"value": []},
    #                             "steps": {"value": 1},
    #                         }
    #                     }
    #                 }
    #             },
    #         ),
    #         np.full(18, 55.55555556),
    #     )
