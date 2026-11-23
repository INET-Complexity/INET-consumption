"""Housing shortfall in minimum_income mode (PLAN.md sections 3.3, 4.4, 5.3 steps 5-6; decision D8).

Invariants (PLAN.md section 6): 5 (tenant rent paid = landlord gross rent received; rental tax base = rent paid;
unpaid rent recorded per landlord and in aggregate) and 6 (social rent shortfall written off by government,
recorded separately and in the deficit). A forced-shortfall private renter moves into social housing at the
start of the next housing step, its dwelling is offered for rent, and it can rent privately again later.
Social tenants with unpaid social rent are not moved.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macromodel.agents.households.households import Households
from macromodel.markets.housing_market.housing_market import HousingMarket
from macromodel.markets.housing_market.sales import SOCIAL_HOUSING_TENURE_STATUS
from tests.test_macromodel.unit.test_country.test_minimum_income_entitlement import _production_simulation

# ---------------------------------------------------------------------------
# HousingMarket.vacate_tenancies
# ---------------------------------------------------------------------------


def _market_and_households():
    properties = pd.DataFrame(
        {
            "House ID": [0, 1, 2],
            "Corresponding Owner Household ID": [0, 0, 2],
            "Corresponding Inhabitant Household ID": [0, 1, 2],
            "Is Owner-Occupied": [1, 0, 1],
        }
    ).set_index("House ID", drop=False)
    market = HousingMarket.__new__(HousingMarket)
    market.states = {"properties": properties}
    household_states = {
        "Corresponding Inhabited House ID": np.array([0, 1, 2, -1]),
        "Tenure Status of the Main Residence": np.array([1, 3, 1, SOCIAL_HOUSING_TENURE_STATUS]),
        "corr_renters": [[1], [], [], []],
    }
    return market, household_states


def test_vacate_tenancies_frees_the_dwelling_and_moves_the_tenant_to_social_housing():
    market, states = _market_and_households()

    market.vacate_tenancies(household_states=states, household_ids=np.array([1]))

    properties = market.states["properties"]
    assert properties.loc[1, "Corresponding Inhabitant Household ID"] == -1
    assert properties.loc[1, "Corresponding Owner Household ID"] == 0
    assert properties.loc[1, "Is Owner-Occupied"] == 0
    assert states["Corresponding Inhabited House ID"][1] == -1
    assert states["Tenure Status of the Main Residence"][1] == SOCIAL_HOUSING_TENURE_STATUS
    assert states["corr_renters"][0] == []
    # Nobody else moves.
    np.testing.assert_array_equal(properties["Corresponding Inhabitant Household ID"].to_numpy(), [0, -1, 2])
    np.testing.assert_array_equal(states["Tenure Status of the Main Residence"], [1, -1, 1, -1])


def test_vacate_tenancies_without_households_changes_nothing():
    market, states = _market_and_households()
    before = market.states["properties"].copy()

    market.vacate_tenancies(household_states=states, household_ids=np.array([], dtype=int))

    pd.testing.assert_frame_equal(market.states["properties"], before)
    np.testing.assert_array_equal(states["Corresponding Inhabited House ID"], [0, 1, 2, -1])


@pytest.mark.parametrize("household_id", [0, 3])
def test_vacate_tenancies_rejects_owners_and_households_without_a_dwelling(household_id):
    market, states = _market_and_households()
    with pytest.raises(ValueError):
        market.vacate_tenancies(household_states=states, household_ids=np.array([household_id]))


# ---------------------------------------------------------------------------
# Households.renting_from_landlord and unpaid_rent_by_landlord
# ---------------------------------------------------------------------------


def _households(house_id, tenure):
    households = Households.__new__(Households)
    households.ts = SimpleNamespace(current=lambda name: len(house_id))
    households.states = {
        "Corresponding Inhabited House ID": np.array(house_id),
        "Tenure Status of the Main Residence": np.array(tenure),
    }
    return households


def _dwellings(owner, inhabitant):
    properties = pd.DataFrame(
        {
            "House ID": np.arange(len(owner)),
            "Corresponding Owner Household ID": owner,
            "Corresponding Inhabitant Household ID": inhabitant,
        }
    ).set_index("House ID", drop=False)
    HousingMarket._refresh_owner_occupied_flags(properties)
    return properties


def test_one_landlord_absorbs_the_unpaid_rent_of_all_its_tenants():
    # Household 0 owns and lives in dwelling 0 and lets dwellings 1 and 2 to households 1 and 2.
    households = _households(house_id=[0, 1, 2, -1], tenure=[1, 3, 3, SOCIAL_HOUSING_TENURE_STATUS])
    properties = _dwellings(owner=[0, 0, 0], inhabitant=[0, 1, 2])

    absorbed = households.unpaid_rent_by_landlord(properties, np.array([0.0, 30.0, 12.5, 7.0]))

    # The social tenant's 7.0 is owed to government, not to a landlord.
    np.testing.assert_array_equal(absorbed, [42.5, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(households.renting_from_landlord(properties), [False, True, True, False])


def test_tenant_who_moved_this_period_is_charged_to_its_new_landlord():
    # Household 1 left dwelling 1 (owner 0, now let to household 3) for dwelling 2 (owner 2) in this clearing.
    households = _households(house_id=[0, 2, 3, 1], tenure=[1, 3, 1, 3])
    properties = _dwellings(owner=[0, 0, 2, 2], inhabitant=[0, 3, 1, 2])

    absorbed = households.unpaid_rent_by_landlord(properties, np.array([0.0, 20.0, 0.0, 0.0]))

    np.testing.assert_array_equal(absorbed, [0.0, 0.0, 20.0, 0.0])


def test_tenure_3_household_in_its_own_dwelling_has_no_landlord():
    # Household 1 rented dwelling 1, which it owns: clearing does not exclude buyer == owner.
    households = _households(house_id=[0, 1], tenure=[1, 3])
    properties = _dwellings(owner=[0, 1], inhabitant=[0, 1])

    assert not households.renting_from_landlord(properties).any()
    np.testing.assert_array_equal(households.unpaid_rent_by_landlord(properties, np.array([0.0, 25.0])), [0.0, 0.0])


@pytest.mark.parametrize(
    "unpaid",
    [np.array([0.0, -1.0, 0.0]), np.array([0.0, np.nan, 0.0]), np.array([0.0, 1.0])],
    ids=["negative", "not-finite", "wrong-shape"],
)
def test_unpaid_rent_by_landlord_rejects_invalid_unpaid_rent(unpaid):
    households = _households(house_id=[0, 1, -1], tenure=[1, 3, SOCIAL_HOUSING_TENURE_STATUS])
    with pytest.raises(ValueError):
        households.unpaid_rent_by_landlord(_dwellings(owner=[0, 0], inhabitant=[0, 1]), unpaid)


def test_unpaid_rent_by_landlord_rejects_a_tenant_not_living_in_its_dwelling():
    households = _households(house_id=[0, 1, 1], tenure=[1, 3, 3])
    properties = _dwellings(owner=[0, 0], inhabitant=[0, 1])
    with pytest.raises(ValueError, match="tenant of a rented-out dwelling"):
        households.unpaid_rent_by_landlord(properties, np.array([0.0, 0.0, 5.0]))


def test_unpaid_rent_by_landlord_rejects_a_dwelling_without_household_owner():
    households = _households(house_id=[0, 1], tenure=[1, 3])
    properties = _dwellings(owner=[0, -1], inhabitant=[0, 1])
    with pytest.raises(ValueError, match="household landlord"):
        households.unpaid_rent_by_landlord(properties, np.array([0.0, 5.0]))


# ---------------------------------------------------------------------------
# Simulation: rent paid, landlord loss, social write-off, vacate and exit
# ---------------------------------------------------------------------------


def _force_zero_resources(households, choose):
    """Settle with zero resources for the household ``choose(households)`` picks; records the pick per call."""
    picks = []
    resources = households.current_floor_settlement_resources

    def zeroed():
        values = np.array(resources(), dtype=float)
        household_id = choose(households)
        picks.append(household_id)
        if household_id is not None:
            values[household_id] = 0.0
        return values

    households.current_floor_settlement_resources = zeroed
    return picks


def _private_renter(households):
    house_id = households.states["Corresponding Inhabited House ID"]
    renter = (households.states["Tenure Status of the Main Residence"] == 3) & (house_id != -1)
    return int(np.flatnonzero(renter & (households.current_consumption_cash_rent() > 0.0))[0])


def _snapshot_before_listing(country):
    """Record housing state after the vacate step and before households list dwellings and bid."""
    snapshots = []
    prepare = country.households.prepare_housing_market_clearing

    def spy(**kwargs):
        properties = country.housing_market.states["properties"]
        snapshots.append(
            {
                "house_id": country.households.states["Corresponding Inhabited House ID"].copy(),
                "tenure": country.households.states["Tenure Status of the Main Residence"].copy(),
                "inhabitant": properties["Corresponding Inhabitant Household ID"].copy(),
            }
        )
        result = prepare(**kwargs)
        snapshots[-1]["up_for_rent"] = properties["Up for Rent"].copy()
        return result

    country.households.prepare_housing_market_clearing = spy
    return snapshots


def _assert_rent_identities(country, check_gdp=True):
    """Invariants 5 and 6 for the current period."""
    hh, government, economy = country.households, country.central_government, country.economy
    tax = government.states["Income Tax"]
    unpaid = hh.ts.current("unpaid_rent")
    cash_rent = hh.current_consumption_cash_rent()
    rent_paid = hh.ts.current("rent_paid")
    house_id = hh.states["Corresponding Inhabited House ID"]
    social = house_id == -1
    properties = country.housing_market.states["properties"]
    private = hh.renting_from_landlord(properties)
    np.testing.assert_array_equal(hh.ts.current("social_housing_tenant"), social)

    # Tenant outflow is the rent actually paid.
    np.testing.assert_array_equal(rent_paid, cash_rent - unpaid)
    assert np.all(rent_paid >= 0.0)

    # Invariant 5: what private tenants pay is what landlords receive, gross; the tax base is the rent paid.
    paid_by_private_tenants = rent_paid[private].sum()
    np.testing.assert_allclose(
        economy.ts.current("total_real_rent_rec")[0] + government.ts.current("taxes_rental_income")[0],
        paid_by_private_tenants,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        government.ts.current("taxes_rental_income")[0], tax * paid_by_private_tenants, rtol=1e-12
    )
    np.testing.assert_allclose(hh.ts.current("income_rental").sum(), (1 - tax) * paid_by_private_tenants, rtol=1e-12)
    # Per landlord: rent of its let dwellings less its tenants' unpaid rent.
    let = properties.loc[
        (properties["Is Owner-Occupied"] == 0) & (properties["Corresponding Inhabitant Household ID"] != -1)
    ]
    gross = np.bincount(
        let["Corresponding Owner Household ID"].to_numpy(dtype=int),
        weights=let["Rent"].to_numpy(dtype=float),
        minlength=len(unpaid),
    )
    absorbed = hh.ts.current("rent_unpaid_absorbed")
    np.testing.assert_allclose(hh.ts.current("income_rental"), (1 - tax) * (gross - absorbed), rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(absorbed.sum(), unpaid[private].sum(), rtol=1e-12, atol=1e-9)
    assert hh.ts.current("total_rent_unpaid_absorbed")[0] == absorbed.sum()
    # Rent without counterparty: a tenure-3 household in its own dwelling.
    no_counterparty = hh.ts.current("rent_unpaid_no_counterparty")
    np.testing.assert_array_equal(no_counterparty, np.where(private | social, 0.0, unpaid))

    # Invariant 6: unpaid social rent is written off, and government receives the rest.
    written_off = government.ts.current("social_rent_written_off")[0]
    np.testing.assert_allclose(written_off, unpaid[social].sum(), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(
        government.ts.current("total_rent_received")[0] + written_off,
        hh.states["Rent paid to Government"],
        rtol=1e-12,
    )
    np.testing.assert_allclose(government.ts.current("total_rent_received")[0], rent_paid[social].sum(), rtol=1e-12)
    # The receipt that enters revenue (and hence the deficit) is net of the write-off.
    taxes = sum(
        government.ts.current(name)[0]
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
    np.testing.assert_allclose(
        government.ts.current("revenue")[0], taxes + government.ts.current("total_rent_received")[0], rtol=1e-12
    )

    # GDP keeps rental services at the rent due on both sides.
    if check_gdp:
        np.testing.assert_allclose(
            economy.ts.current("gdp_income")[0] - economy.ts.current("gdp_output")[0],
            0.0,
            atol=1e-9 * economy.ts.current("gdp_output")[0],
        )
    # The household cash ledger still closes.
    assert np.max(np.abs(hh.ts.current("stage5_cash_ledger_residual"))) <= 1e-7


def test_forced_shortfall_renter_moves_to_social_housing_next_period_and_can_rent_again(datawrapper):
    simulation = _production_simulation(datawrapper, {"mode": "minimum_income", "goods_floor": False})
    country = simulation.countries["FRA"]
    hh = country.households
    snapshots = _snapshot_before_listing(country)
    period = {"t": 0}
    picks = _force_zero_resources(hh, lambda households: _private_renter(households) if period["t"] == 0 else None)

    # Period 0: the renter cannot pay; the landlord absorbs the rent.
    simulation.iterate(t=0)
    renter = picks[0]
    dwelling = int(hh.states["Corresponding Inhabited House ID"][renter])
    landlord = int(country.housing_market.states["properties"].loc[dwelling, "Corresponding Owner Household ID"])
    rent_due = hh.current_consumption_cash_rent()[renter]
    assert rent_due > 0.0 and landlord != renter
    assert hh.ts.current("unpaid_rent")[renter] == rent_due
    assert hh.ts.current("rent_paid")[renter] == 0.0
    assert hh.ts.current("rent_unpaid_absorbed")[landlord] >= rent_due
    assert not hh.ts.current("unpaid_rent_vacated").any()
    _assert_rent_identities(country)

    # Period 1: it leaves the dwelling before the listing step; the dwelling is offered for rent. Its bids are
    # withheld so that it stays in social housing for the period (about 10% re-let in the same step).
    clear = country.housing_market.clear
    bid = {"rent": np.nan}

    def set_bid(**kwargs):
        for name, value in (("max_rent_willing_to_pay", bid["rent"]), ("max_price_willing_to_pay", np.nan)):
            bids = np.array(kwargs[name], dtype=float)
            bids[renter] = value
            kwargs[name] = bids
        return clear(**kwargs)

    country.housing_market.clear = set_bid
    period["t"] = 1
    simulation.iterate(t=1)
    vacated = hh.ts.current("unpaid_rent_vacated")
    private_unpaid_before = hh.ts.historic("unpaid_rent")[-2] > 0.0
    assert vacated[renter]
    assert np.all(vacated <= private_unpaid_before)
    state = snapshots[1]
    assert state["house_id"][renter] == -1
    assert state["tenure"][renter] == SOCIAL_HOUSING_TENURE_STATUS
    assert state["inhabitant"].loc[dwelling] == -1
    assert state["up_for_rent"].loc[dwelling]
    assert renter not in hh.states["corr_renters"][landlord]
    # It searches from social housing through the existing rule (it bids to rent or to buy).
    assert np.isfinite(hh.ts.current("max_rent_willing_to_pay")[renter]) or np.isfinite(
        hh.ts.current("max_price_willing_to_pay")[renter]
    )
    # In social housing it pays the social rent, to government.
    assert hh.states["Corresponding Inhabited House ID"][renter] == -1
    assert hh.ts.current("social_housing_tenant")[renter]
    np.testing.assert_allclose(
        hh.ts.current("rent")[renter],
        hh.states["Social Rent per Adult"] * hh.states["Number of Adults"][renter],
        rtol=1e-12,
    )
    assert hh.ts.current("rent")[renter] > 0.0
    _assert_rent_identities(country)

    # Period 2: with a high enough rent bid it leaves social housing through the ordinary market.
    period["t"] = 2
    bid["rent"] = 1e3 * np.nanmax(country.housing_market.states["properties"]["Rent"].to_numpy(dtype=float))
    simulation.iterate(t=2)
    assert hh.states["Corresponding Inhabited House ID"][renter] != -1
    assert hh.states["Tenure Status of the Main Residence"][renter] == 3
    assert not hh.ts.current("social_housing_tenant")[renter]
    _assert_rent_identities(country)


def test_social_tenant_with_unpaid_rent_is_written_off_and_not_moved(datawrapper):
    simulation = _production_simulation(datawrapper, {"mode": "minimum_income", "goods_floor": False})
    country = simulation.countries["FRA"]
    hh, government = country.households, country.central_government
    snapshots = _snapshot_before_listing(country)
    period = {"t": 0}

    def social_tenant(households):
        if period["t"] != 1:
            return None
        social = np.flatnonzero(households.states["Corresponding Inhabited House ID"] == -1)
        return int(social[np.argmax(households.current_consumption_cash_rent()[social])])

    picks = _force_zero_resources(hh, social_tenant)
    simulation.iterate(t=0)
    period["t"] = 1
    simulation.iterate(t=1)
    tenant = picks[1]
    social_rent = hh.current_consumption_cash_rent()[tenant]
    assert social_rent > 0.0
    assert hh.ts.current("unpaid_rent")[tenant] == social_rent
    assert government.ts.current("social_rent_written_off")[0] >= social_rent
    assert hh.ts.current("rent_unpaid_absorbed").sum() == pytest.approx(
        hh.ts.current("unpaid_rent")[hh.states["Corresponding Inhabited House ID"] != -1].sum()
    )
    _assert_rent_identities(country)

    period["t"] = 2
    simulation.iterate(t=2)
    assert not hh.ts.current("unpaid_rent_vacated")[tenant]
    assert snapshots[2]["house_id"][tenant] == -1
    assert snapshots[2]["tenure"][tenant] == SOCIAL_HOUSING_TENURE_STATUS
    _assert_rent_identities(country)


def test_legacy_mode_pays_rent_in_full_and_moves_nobody(datawrapper):
    simulation = _production_simulation(datawrapper, {})
    country = simulation.countries["FRA"]
    hh, government = country.households, country.central_government
    for t in range(2):
        simulation.iterate(t=t)
        np.testing.assert_array_equal(hh.ts.current("unpaid_rent"), 0.0)
        np.testing.assert_array_equal(hh.ts.current("rent_paid"), hh.current_consumption_cash_rent())
        np.testing.assert_array_equal(hh.ts.current("rent_unpaid_absorbed"), 0.0)
        assert government.ts.current("social_rent_written_off")[0] == 0.0
        assert not hh.ts.current("unpaid_rent_vacated").any()
        # Legacy accounting: tax on the rent due by private renters; government receives the social rent due.
        private = hh.states["Tenure Status of the Main Residence"] == 3
        assert government.ts.current("taxes_rental_income")[0] == (
            government.states["Income Tax"] * hh.ts.current("rent")[private].sum()
        )
        assert government.ts.current("total_rent_received")[0] == hh.states["Rent paid to Government"]


def test_tenure_3_household_in_its_own_dwelling_books_unpaid_rent_without_counterparty(datawrapper):
    simulation = _production_simulation(datawrapper, {"mode": "minimum_income", "goods_floor": False})
    country = simulation.countries["FRA"]
    hh = country.households
    snapshots = _snapshot_before_listing(country)
    period = {"t": 0}
    chosen = {}

    # After period-1 clearing, turn one owner-occupier into a tenure-3 household in its own dwelling: the state
    # rental clearing can produce when a household rents a dwelling it owns.
    process = hh.process_housing_market_clearing

    def make_self_tenant(**kwargs):
        if period["t"] == 1:
            house_id = hh.states["Corresponding Inhabited House ID"]
            properties = country.housing_market.states["properties"]
            owner_occupier = (hh.states["Tenure Status of the Main Residence"] == 1) & (house_id != -1)
            candidates = np.flatnonzero(owner_occupier)
            owners = properties.loc[house_id[candidates], "Corresponding Owner Household ID"].to_numpy(dtype=int)
            rents = properties.loc[house_id[candidates], "Rent"].to_numpy(dtype=float)
            household_id = int(candidates[(owners == candidates) & (rents > 0.0)][0])
            hh.states["Tenure Status of the Main Residence"][household_id] = 3
            chosen["id"] = household_id
        return process(**kwargs)

    hh.process_housing_market_clearing = make_self_tenant
    _force_zero_resources(hh, lambda households: chosen.get("id") if period["t"] == 1 else None)

    simulation.iterate(t=0)
    period["t"] = 1
    simulation.iterate(t=1)
    tenant = chosen["id"]
    unpaid = hh.ts.current("unpaid_rent")
    assert unpaid[tenant] > 0.0 and unpaid[tenant] == hh.current_consumption_cash_rent()[tenant]
    assert hh.ts.current("rent_unpaid_no_counterparty")[tenant] == unpaid[tenant]
    assert not hh.renting_from_landlord(country.housing_market.states["properties"])[tenant]
    own_dwelling = int(hh.states["Corresponding Inhabited House ID"][tenant])
    # Its rent is a cash use without a recipient, so GDP income falls short of output by it (pre-existing).
    _assert_rent_identities(country, check_gdp=False)

    period["t"] = 2
    simulation.iterate(t=2)
    assert not hh.ts.current("unpaid_rent_vacated")[tenant]
    assert snapshots[2]["house_id"][tenant] == own_dwelling != -1
