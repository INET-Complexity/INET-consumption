import numpy as np
import pytest

from macromodel.agents.households.func.consumption import CreditAugmentedConsumption


@pytest.mark.parametrize("resolver", [False, True])
@pytest.mark.parametrize("final_rent,final_imputed", [(20.0, 0.0), (80.0, 0.0), (0.0, 60.0), (350.0, 0.0)])
def test_two_pass_cash_needs_and_fixed_credit(test_country, monkeypatch, resolver, final_rent, final_imputed):
    c = test_country
    h = c.households
    n = h.ts.current("n_households")
    c.configuration.households.parameters.uses_feasibility_resolver = resolver
    c.central_government.states["Value-added Tax"] = 0.2
    monkeypatch.setattr(c, "assume_zero_growth", False)
    rule = CreditAugmentedConsumption()
    h.functions["consumption"] = rule

    def evaluate(**kwargs):
        total = 200.0 + kwargs["income"]
        return {"target_consumption_real_budget": total}, total

    monkeypatch.setattr(rule, "_evaluate_target", evaluate)
    monkeypatch.setattr(h, "compute_target_investment", lambda **kw: np.zeros_like(h.ts.current("target_consumption")))
    for key, value in [
        ("expected_income", 100.0),
        ("rent", 40.0),
        ("rent_imputed", 0.0),
        ("liquid_financial_assets", 20.0),
        ("wealth_financial_assets", 20.0),
        ("illiquid_financial_assets", 0.0),
        ("received_mortgages", 0.0),
        ("price_paid_for_property", 0.0),
        ("ficp_exclusion_remaining_periods", 0.0),
    ]:
        h.ts.override_current(key, np.full(n, value))
    monkeypatch.setattr(c.credit_market, "preview_opening_household_service", lambda: (np.zeros(n), np.zeros(n)))
    for name in (
        "compute_scheduled_consumption_loan_payments_by_household",
        "compute_scheduled_mortgage_payments_by_household",
        "compute_opening_scheduled_consumption_payments_by_household",
        "compute_opening_scheduled_mortgage_payments_by_household",
    ):
        monkeypatch.setattr(c.credit_market, name, lambda: np.zeros(n))
    h.consumption_weights = np.full(len(h.consumption_weights), 1.0 / len(h.consumption_weights))
    c._set_household_target_demand(replace_current=False)
    np.testing.assert_allclose(h.ts.current("target_consumption").sum(axis=1) * 1.2, 260.0)
    # Credit request must keep planning rent even though housing preparation changed it.
    h.ts.override_current("rent", np.full(n, 900.0))
    h.compute_target_credit(current_sales=None)
    if not resolver:
        np.testing.assert_allclose(h.ts.current("target_consumption_loans"), 180.0)
    else:
        np.testing.assert_allclose(h.ts.current("target_consumption_loans"), h.current_live_credit_requested())
        h.populate_post_grant_feasible_plan_from_granted_credit(credit_granted=np.full(n, 10.0))
        h.reserve_post_grant_executable_liquidation(available_pre_stage4_ifa=np.zeros(n))
        settled = h.post_grant_feasible_plan
    lengths = {
        key: len(h.ts.dicts[key])
        for key in ("target_consumption", "cacf_real_consumption_budget", "target_consumption_total_mpc")
    }
    h.ts.override_current("rent", np.full(n, final_rent))
    h.ts.override_current("rent_imputed", np.full(n, final_imputed))
    c._set_household_target_demand(replace_current=True)
    goods = max(0.0, 300.0 - final_rent - final_imputed)
    np.testing.assert_allclose(h.ts.current("target_consumption").sum(axis=1) * 1.2, goods)
    np.testing.assert_allclose(h.ts.current("target_consumption_total_mpc"), 1.0)
    np.testing.assert_allclose(h.ts.current("formula_implied_mpc"), 1.0 / 1.2)
    assert lengths == {key: len(h.ts.dicts[key]) for key in lengths}
    if resolver:

        def fail_rebooking(**kw):
            pytest.fail("Post-labour refresh must not book loans")

        monkeypatch.setattr(c.credit_market, "settle_granted_consumption_loans", fail_rebooking)
        h.refresh_post_labour_feasibility(np.zeros(n))
        np.testing.assert_allclose(h.post_grant_feasible_plan.credit_granted, settled.credit_granted)
        np.testing.assert_allclose(h.current_post_grant_residual_shortfall(), max(goods + final_rent - 130.0, 0.0))
