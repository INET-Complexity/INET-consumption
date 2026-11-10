from types import SimpleNamespace

import numpy as np
import pytest

from macromodel.agents.households.households import Households
from macromodel.country.country import Country
from macromodel.timeseries import TimeSeries

_NO_HOUSE = dict(price=0.0, mortgage=0.0, lfa=0.0, purchase_sale=0.0)


@pytest.mark.parametrize(
    "receipt,sale_proceeds,liquidation,sale_cost,housing,expected_support",
    [
        (80.0, 0.0, 0.0, 0.0, _NO_HOUSE, 20.0),
        (70.0, 0.0, 0.0, 0.0, _NO_HOUSE, 30.0),
        (120.0, 0.0, 0.0, 0.0, _NO_HOUSE, 0.0),
        # A pure seller's proceeds are a resource (max(-X, 0)).
        (80.0, 15.0, 0.0, 0.0, _NO_HOUSE, 5.0),
        # A forced sale of 10 that pays a trade cost of 5 brings only 5 of cash.
        (80.0, 0.0, 10.0, 5.0, _NO_HOUSE, 15.0),
        # A buyer pays 50 with a mortgage of 30 and its 20 of liquid assets
        # (D = 20): the purchase neither raises nor lowers support.
        (80.0, 0.0, 0.0, 0.0, dict(price=50.0, mortgage=30.0, lfa=20.0, purchase_sale=0.0), 20.0),
        # Same buyer with 10 liquid: it sells 10 of illiquid assets plus the
        # trade cost of 3. None of that sale is cash for other needs.
        (80.0, 0.0, 13.0, 3.0, dict(price=50.0, mortgage=30.0, lfa=10.0, purchase_sale=10.0), 20.0),
        # A mover funds 40 of the 50 with its own sale and 10 with liquid assets.
        (80.0, 40.0, 0.0, 0.0, dict(price=50.0, mortgage=0.0, lfa=10.0, purchase_sale=0.0), 20.0),
    ],
)
def test_final_support_uses_actual_receipts_and_posts_once(
    receipt, sale_proceeds, liquidation, sale_cost, housing, expected_support
):
    """Realised support counts the forced sale net of its cost and never finances a property purchase."""
    hh = Households.__new__(Households)
    hh.uses_feasibility_resolver = True
    hh.post_grant_feasible_plan = SimpleNamespace(
        reserved_liquidation_total=np.array([liquidation]),
        liquidation_sale_cost=np.array([sale_cost]),
        purchase_liquidation_total=np.array([housing["purchase_sale"]]),
    )
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
        price_paid_for_property=np.array([housing["price"]]),
        interest_paid=np.zeros(1),
        debt_installments=np.zeros(1),
        liquid_financial_assets=np.array([housing["lfa"]]),
        received_consumption_loans=np.zeros(1),
        received_mortgages=np.array([housing["mortgage"]]),
        property_sale_proceeds=np.array([sale_proceeds]),
        mortgage_sale_repayment=np.zeros(1),
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
    # Proceeds are financing, not income: income is receipts plus support only,
    # and income together with proceeds covers spending.
    assert hh.compute_income()[0] == pytest.approx(receipt + expected_support)
    assert hh.compute_income()[0] + sale_proceeds + liquidation - sale_cost >= 100
    assert len(hh.ts.dicts["income_social_transfers"]) == 1
