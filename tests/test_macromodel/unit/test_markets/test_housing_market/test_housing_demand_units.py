"""Housing demand: bid units, affordability in matching, loan-to-value cap, price-growth clamp."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macromodel.agents.households.func.property import (
    PRIVATE_RENTER_TENURE_STATUS,
    DefaultHouseholdDemandForProperty,
)
from macromodel.economy.func.house_price_index import HPIForecastingConstant
from macromodel.markets.credit_market.func.clearing import _mortgage_loan_to_value_cap
from macromodel.markets.housing_market.func.clearing import AutomaticHousingMarketClearer
from macromodel.markets.housing_market.purchase_completion import financeable_price, free_purchase_assets

PRICE_COEFFICIENT, PRICE_EXPONENT = 42.9, 0.789
RENT_COEFFICIENT, RENT_EXPONENT = 17.2166, 0.3464


def _demand_rule() -> DefaultHouseholdDemandForProperty:
    return DefaultHouseholdDemandForProperty(
        probability_stay_in_rented_property=0.0,
        probability_stay_in_owned_property=1.0,
        maximum_price_income_coefficient=PRICE_COEFFICIENT,
        maximum_price_income_exponent=PRICE_EXPONENT,
        maximum_price_noise_mean=0.0,
        maximum_price_noise_variance=0.0,
        maximum_rent_income_coefficient=RENT_COEFFICIENT,
        maximum_rent_income_exponent=RENT_EXPONENT,
        psychological_pressure_of_renting=0.1,
        cost_comparison_temperature=1.0,
        price_initial_markup=0.0,
        price_decrease_probability_monthly=0.0,
        price_mean_percentage_reduction=0.0,
        std_of_price_percentage_reduction=0.0,
        rent_initial_markup=0.0,
        rent_decrease_probability_monthly=0.0,
        rent_mean_percentage_reduction=0.0,
        std_of_rent_percentage_reduction=0.0,
        partial_rent_inflation_indexation=0.0,
        partial_rent_inflation_delay=1,
    )


def _bids(income, *, hpi_growth, **units):
    """Bids of renters who all reconsider; ``hpi_growth`` steers the buy-or-rent choice."""
    np.random.seed(0)
    price, rent, _ = _demand_rule().compute_demand(
        housing_data=pd.DataFrame({"Value": [1.0]}),
        household_residence_tenure_status=np.full(len(income), PRIVATE_RENTER_TENURE_STATUS),
        household_income=np.asarray(income, dtype=float),
        household_financial_wealth=np.zeros(len(income)),
        observed_fraction_value_price=np.array([1.0, 0.0]),
        observed_fraction_rent_value=np.array([0.01, 0.0]),
        expected_hpi_growth=hpi_growth,
        assumed_mortgage_maturity=100,
        rental_income_taxes=0.0,
        **units,
    )
    return price, rent


def test_price_bid_uses_annual_income_of_one_household():
    """Quarterly income of 5000 households, 11,250 each: annual 45,000, bid about 4.5 annual incomes."""
    scale, annual_income = 5000.0, 45_000.0
    price, _ = _bids([scale * annual_income / 4], hpi_growth=1.0, population_scale_factor=scale, period_months=3)

    expected_per_household = PRICE_COEFFICIENT * annual_income**PRICE_EXPONENT
    np.testing.assert_allclose(price, [scale * expected_per_household])
    assert 4.0 < expected_per_household / annual_income < 5.0


def test_rent_bid_is_a_monthly_rent_converted_to_the_model_period():
    scale, annual_income = 5000.0, 45_000.0
    _, rent = _bids([scale * annual_income / 4], hpi_growth=-0.5, population_scale_factor=scale, period_months=3)

    np.testing.assert_allclose(rent, [scale * 3 * RENT_COEFFICIENT * annual_income**RENT_EXPONENT])


def test_bids_scale_with_the_number_of_households_represented():
    """Representing more real households per model household must not change the per-household bid."""
    per_household = np.array([20_000.0, 45_000.0, 90_000.0])
    one, _ = _bids(per_household, hpi_growth=1.0)
    many, _ = _bids(5000.0 * per_household, hpi_growth=1.0, population_scale_factor=5000.0)

    np.testing.assert_allclose(many, 5000.0 * one)


def test_default_units_are_annual_income_of_one_household():
    price, _ = _bids([45_000.0], hpi_growth=1.0)

    np.testing.assert_allclose(price, [PRICE_COEFFICIENT * 45_000.0**PRICE_EXPONENT])


def test_demand_rejects_non_positive_units():
    with pytest.raises(ValueError, match="must be positive"):
        _bids([1.0], hpi_growth=0.0, population_scale_factor=0.0)


def _for_sale(prices) -> pd.DataFrame:
    n = len(prices)
    return pd.DataFrame(
        {
            "Value": np.asarray(prices, dtype=float),
            "Sale Price": np.asarray(prices, dtype=float),
            "Rent": np.ones(n),
            "Temporarily for Sale": np.ones(n, dtype=bool),
            "Up for Rent": np.zeros(n, dtype=bool),
            "Corresponding Owner Household ID": np.full(n, 9),
        },
        index=np.arange(10, 10 + n),
    )


def _match(bids, prices) -> dict[int, float]:
    transactions = AutomaticHousingMarketClearer(random_assignment_shock_variance=0.0).perform_matching(
        housing_data=_for_sale(prices),
        household_main_residence_tenure_status=np.zeros(len(bids)),
        max_willing_to_pay=np.asarray(bids, dtype=float),
        is_rental_market=False,
    )
    return dict(zip(transactions["buyer_id"], transactions["price_or_rent"]))


def test_no_household_is_matched_to_a_property_above_its_bid():
    """Characterisation of the defect: bids of 10 were matched to prices of 100."""
    assert _match([10.0, 10.0], [100.0, 120.0]) == {}


def test_household_takes_the_dearest_property_it_can_afford():
    assert _match([105.0], [80.0, 100.0, 120.0]) == {0: 100.0}


def test_matching_maximises_the_number_of_affordable_pairs():
    """Household 0 could take either property; household 1 only the cheaper one. Both are housed."""
    assert _match([130.0, 105.0], [100.0, 120.0]) == {0: 120.0, 1: 100.0}


def test_unaffordable_property_stays_listed():
    housing_data = _for_sale([100.0, 500.0])
    AutomaticHousingMarketClearer(random_assignment_shock_variance=0.0).perform_matching(
        housing_data=housing_data,
        household_main_residence_tenure_status=np.zeros(2),
        max_willing_to_pay=np.array([110.0, 120.0]),
        is_rental_market=False,
    )

    assert housing_data["Temporarily for Sale"].tolist() == [False, True]


def test_loan_to_value_cap_is_the_ratio_times_the_purchase_price():
    banks = SimpleNamespace(parameters=SimpleNamespace(mortgage_loan_to_value_ratio=0.8))
    prices = np.array([0.0, 200.0, np.nan])
    households = SimpleNamespace(ts=SimpleNamespace(current=lambda name: {"target_house_price": prices}[name]))

    np.testing.assert_allclose(_mortgage_loan_to_value_cap(banks, households), [0.0, 160.0, 0.0])


@pytest.mark.parametrize("value,expected", [(0.01, 0.01), (0.5, 0.2), (-0.5, -0.2)])
def test_house_price_growth_forecast_is_clamped_to_its_bounds(value, expected):
    """The upper bound used to act as a floor, so every forecast was the maximum."""
    forecast = HPIForecastingConstant(value=value).forecast_hpi_growth(
        historic_hpi=np.array([0.0, 0.0, 0.0]), min_hpi_growth=-0.2, max_hpi_growth=0.2
    )

    np.testing.assert_allclose(np.ravel(forecast)[0], expected)


def test_financeable_price_is_own_funds_over_the_down_payment_share():
    np.testing.assert_allclose(financeable_price(np.array([0.0, 20.0, -5.0]), 0.8), [0.0, 100.0, 0.0])
    assert np.all(np.isinf(financeable_price(np.array([0.0, 20.0]), 1.0)))


def test_free_purchase_assets_reserve_the_cash_need_the_planned_sale_and_the_trade_cost():
    free_lfa, free_ifa = free_purchase_assets(
        cash_need=np.array([30.0, 200.0]),
        liquid_assets=np.array([100.0, 100.0]),
        illiquid_assets=np.array([50.0, 50.0]),
        planned_liquidation=np.array([10.0, 60.0]),
        sale_cost=np.array([1.0, 1.0]),
    )

    np.testing.assert_allclose(free_lfa, [70.0, 0.0])
    np.testing.assert_allclose(free_ifa, [39.0, 0.0])


def _demand(rule, tenure, *, hpi_growth, **options):
    np.random.seed(0)
    return rule.compute_demand(
        housing_data=pd.DataFrame({"Value": [1.0]}),
        household_residence_tenure_status=np.asarray(tenure),
        household_income=np.full(len(tenure), 45_000.0),
        household_financial_wealth=np.zeros(len(tenure)),
        observed_fraction_value_price=np.array([1.0, 0.0]),
        observed_fraction_rent_value=np.array([0.01, 0.0]),
        expected_hpi_growth=hpi_growth,
        assumed_mortgage_maturity=100,
        rental_income_taxes=0.0,
        **options,
    )


def test_committed_buyer_bids_to_buy_although_it_would_stay_or_rent():
    """Renters never reconsider and renting is cheaper, yet the committed household 1 bids to buy."""
    rule = _demand_rule()
    rule.probability_stay_in_rented_property = 1.0

    price, rent, _ = _demand(
        rule, [PRIVATE_RENTER_TENURE_STATUS] * 2, hpi_growth=-0.5, committed_buyers=np.array([False, True])
    )

    assert np.isnan(price[0]) and np.isnan(rent[0])
    assert price[1] > 0.0 and np.isnan(rent[1])


def test_moving_owner_gets_a_rent_bid_and_keeps_its_choice_to_buy():
    rule = _demand_rule()
    rule.probability_stay_in_owned_property = 0.0

    price, rent, moving = _demand(rule, [1], hpi_growth=1.0, moving_owners_bid_rent=True)
    _, rent_without, _ = _demand(rule, [1], hpi_growth=1.0)

    assert moving[0] and price[0] > 0.0 and rent[0] > 0.0
    assert np.isnan(rent_without[0])


def _gap_inputs(sales_rows, *, liquid, illiquid=None, cash_need=None):
    from macromodel.markets.housing_market.purchase_completion import PurchaseCompletionInputs

    n = len(liquid)
    sales = pd.DataFrame(sales_rows, columns=["sales_types", "property_id", "buyer_id", "seller_id", "price_or_rent"])
    return PurchaseCompletionInputs(
        current_sales=sales,
        property_owner=pd.Series([0.0, np.nan]),
        property_value=pd.Series([200.0, 300.0]),
        property_inhabitant=pd.Series([0.0, -1.0]),
        cash_need=np.zeros(n) if cash_need is None else np.asarray(cash_need, dtype=float),
        liquid_assets=np.asarray(liquid, dtype=float),
        illiquid_assets=np.zeros(n) if illiquid is None else np.asarray(illiquid, dtype=float),
        planned_liquidation=np.zeros(n),
        sale_cost=np.zeros(n),
    )


def test_mortgage_request_is_the_price_not_covered_by_free_assets_and_net_sale_proceeds():
    """Household 0 sells its home for 200 (repays 50) and buys for 300 with 40 free: gap 300 - 150 - 40.

    Household 1 buys household 0's home for 200 with liquid 100 and cash need 30: gap 200 - 70.
    Household 2 buys nothing.
    """
    from macromodel.markets.housing_market.purchase_completion import mortgage_financing_gap

    inputs = _gap_inputs(
        [("Sell", 0, 1, 0, 200.0), ("Sell", 1, 0, -1, 300.0), ("Rent", 1, 2, -1, 5.0)],
        liquid=[40.0, 100.0, 500.0],
        cash_need=[0.0, 30.0, 0.0],
    )
    inputs.current_sales.loc[1, "seller_id"] = 2  # a household that owns nothing mortgaged
    inputs = inputs.__class__(**{**inputs.__dict__, "property_owner": pd.Series([0.0, 2.0])})

    gap = mortgage_financing_gap(inputs, principal_left=np.array([50.0, 0.0, 0.0]), margin=0.0)

    np.testing.assert_allclose(gap, [110.0, 130.0, 0.0])


def test_mortgage_request_margin_lets_the_requested_mortgage_complete_the_purchase():
    from macromodel.markets.housing_market.purchase_completion import (
        mortgage_financing_gap,
        resolve_purchase_completion,
    )

    inputs = _gap_inputs([("Sell", 1, 0, 1, 300.0)], liquid=[0.1 + 0.2, 0.0], cash_need=[0.1, 0.0])
    inputs = inputs.__class__(**{**inputs.__dict__, "property_owner": pd.Series([np.nan, 1.0])})
    principal_left = np.zeros(2)

    request = mortgage_financing_gap(inputs, principal_left=principal_left)
    completion = resolve_purchase_completion(inputs, granted_mortgage=request, principal_left=principal_left)

    assert request[0] > 300.0 - 0.2
    assert request[0] - (300.0 - 0.2) < 1e-6
    assert completion.kept_sales[0]
    assert request[1] == 0.0


def test_households_request_the_financing_gap_when_it_is_given(test_households):
    n = test_households.ts.current("n_households")
    gap = np.full(n, 70.0)
    sales = pd.DataFrame(
        [("Sell", 0, 1, 0, 100.0)], columns=["sales_types", "property_id", "buyer_id", "seller_id", "price_or_rent"]
    )

    test_households.uses_feasibility_resolver = False
    test_households.compute_target_credit(current_sales=sales, mortgage_financing_gap=gap)

    expected = np.zeros(n)
    expected[1] = 70.0
    np.testing.assert_allclose(test_households.ts.current("target_mortgage"), expected)


def test_purchase_cash_need_adds_deposit_interest_paid_but_not_interest_received():
    """N = [10, 10]. Household 0 pays 2 on its deposits (negative rate); household 1 receives 3."""
    from macromodel.country.country import Country

    series = {
        "interest_rate_on_household_deposits": np.array([0.0]),
        "overdraft_rate_on_household_deposits": np.array([0.0]),
    }
    country = SimpleNamespace(
        households=SimpleNamespace(
            planning_cash_need=lambda service: np.array([10.0, 10.0]) + service,
            compute_interest_paid_on_deposits=lambda **_rates: np.array([2.0, -3.0]),
        ),
        credit_market=SimpleNamespace(preview_opening_household_service=lambda: (np.zeros(2), np.array([1.0, 0.0]))),
        banks=SimpleNamespace(ts=SimpleNamespace(current=series.__getitem__)),
    )

    np.testing.assert_allclose(Country._purchase_cash_need(country), [13.0, 10.0])
