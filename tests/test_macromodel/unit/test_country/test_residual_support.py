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
    country, hh = _country(receipt, sale_proceeds, liquidation, sale_cost, housing)
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


@pytest.mark.parametrize(
    "case,expected_support",
    [
        # Goods of 100 against a floor of 60: support covers the floor only.
        (dict(receipt=50.0, floor=60.0), 10.0),
        (dict(receipt=80.0, floor=60.0), 0.0),
        # Cash rent is covered in full on top of the floor.
        (dict(receipt=50.0, floor=60.0, rent=25.0), 35.0),
        # Realised goods below the floor: support covers what was bought.
        (dict(receipt=20.0, floor=60.0, spent=40.0, target=40.0), 20.0),
        # The only shortfall is investment (spending above the goods target).
        (dict(receipt=100.0, spent=150.0), 0.0),
        # The only shortfall is debt service.
        (dict(receipt=100.0, interest=30.0, instalments=40.0), 0.0),
        # The only shortfall is a property purchase.
        (dict(receipt=100.0, housing=dict(price=50.0, mortgage=30.0, lfa=20.0, purchase_sale=0.0)), 0.0),
        # VAT: the floor is gross-of-VAT cash, realised goods are VAT-exclusive.
        (dict(receipt=50.0, floor=110.0, vat=0.2), 60.0),
    ],
)
def test_final_support_covers_the_goods_floor_and_cash_rent_only(case, expected_support):
    country, hh = _country(**case)
    country.reconcile_realised_household_support()
    assert hh.ts.current("stage5_subsistence_support")[0] == pytest.approx(expected_support)
    assert country.current_stage5_subsistence_support_fiscal_increment() == pytest.approx(expected_support)


def _country(
    receipt,
    sale_proceeds=0.0,
    liquidation=0.0,
    sale_cost=0.0,
    housing=_NO_HOUSE,
    *,
    floor=100.0,
    spent=100.0,
    target=100.0,
    rent=0.0,
    interest=0.0,
    instalments=0.0,
    vat=0.0,
):
    """One household that spends ``spent`` on goods ordered up to ``target``; the floor is gross of VAT."""
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
        nominal_amount_spent_in_lcu=np.array([[spent]]),
        target_consumption=np.array([[target]]),
        rent=np.array([rent]),
        price_paid_for_property=np.array([housing["price"]]),
        interest_paid=np.array([interest]),
        debt_installments=np.array([instalments]),
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
        states={"Value-added Tax": vat, "Household Capital Formation Tax": 0.0, "Household Investment VAT Rate": 0.0}
    )
    country.economy = SimpleNamespace(ts=TimeSeries(subsistence_consumption=np.array([floor])))
    return country, hh
