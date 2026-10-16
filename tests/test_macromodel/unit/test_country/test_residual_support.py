from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.agents.households.households import Households
from macromodel.country.country import Country
from macromodel.timeseries import TimeSeries


@pytest.mark.parametrize("receipt,expected_support", [(80.0, 20.0), (70.0, 30.0), (120.0, 0.0)])
def test_final_support_uses_actual_receipts_and_posts_once(receipt, expected_support):
    hh = Households.__new__(Households)
    hh.uses_feasibility_resolver = True
    hh.post_grant_feasible_plan = SimpleNamespace(reserved_liquidation_total=np.zeros(1))
    hh.functions = {"wealth": SimpleNamespace(), "investment": SimpleNamespace()}
    hh.ts = TimeSeries(
        n_households=1,
        income_employee=np.array([receipt]),
        income_social_transfers=np.array([40.0]),
        income_rental=np.zeros(1),
        income_financial_assets=np.zeros(1),
        stage5_subsistence_support=np.array([40.0]),
        total_income_social_transfers=np.array([40.0]),
        nominal_amount_spent_in_lcu=np.array([[100.0]]),
        target_consumption=np.array([[100.0]]),
        rent=np.zeros(1),
        price_paid_for_property=np.zeros(1),
        interest_paid=np.zeros(1),
        debt_installments=np.zeros(1),
        liquid_financial_assets=np.zeros(1),
        received_consumption_loans=np.zeros(1),
        received_mortgages=np.zeros(1),
        expected_income=np.array([80.0]),
        financing_income_error=np.zeros(1),
    )
    country = Country.__new__(Country)
    country.households = hh
    country._stage5_subsistence_support_by_household = np.array([40.0])
    country._staged_household_income_shock = None
    country.central_government = SimpleNamespace(
        states={"Value-added Tax": 0.0, "Household Capital Formation Tax": 0.0, "Household Investment VAT Rate": 0.0}
    )
    country.reconcile_realised_household_support()
    assert hh.ts.current("stage5_subsistence_support")[0] == expected_support
    assert hh.ts.current("income_social_transfers")[0] == expected_support
    assert country.current_stage5_subsistence_support_fiscal_increment() == expected_support
    assert hh.ts.current("financing_income_error")[0] == receipt - 80
    assert hh.compute_income()[0] >= 100
    assert len(hh.ts.dicts["income_social_transfers"]) == 1
