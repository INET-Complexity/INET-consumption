"""Audit existing FRA output; no simulations or model mutations.

Run from repository root: .venv/bin/python docs/audits/2026-11-12-macro-finance/replay.py
Only aggregate diagnostics are exported. Monetary output is in EUR billions.
"""

import ast
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from macromodel.agents.central_bank.func.policy_rate import SmoothTaylorRule
from macromodel.agents.government_entities.func.consumption import ExpectedGrowthGovernmentConsumptionSetter
from macromodel.utils.prehooks.irf_shocks import (
    ShockSpec,
    _GovernmentConsumptionShockProxy,
    _PolicyRateShockProxy,
)

ROOT = Path(__file__).resolve().parents[3]
DEST = Path(__file__).resolve().parent
SOURCE = ROOT / "run_model/data/output_data"
df = pd.read_pickle(SOURCE / "FRA_df_benchmark.pkl")
manifest = json.loads((SOURCE / "run_manifest_FRA_seed40_t50.json").read_text())
rows = []
summary = {}

with h5py.File(SOURCE / "FRA_benchmark.h5", "r") as h5:
    configuration_text = h5.attrs["configuration"]
    config = ast.literal_eval(configuration_text)
    country = config["country_configurations"]["FRA"]
    firms = h5["FRA/firms"]
    banks = h5["FRA/banks"]
    gov = h5["FRA/central_government"]
    horizon, n_firms = firms["deposits"].shape
    arrays = {
        key: np.asarray(value)
        for key, value in firms.items()
        if isinstance(value, h5py.Dataset) and value.ndim == 2 and value.shape[0] == horizon
    }
    # The exported flattened firm/good arrays must be reshaped before summing.
    purchases = arrays["nominal_amount_spent_in_lcu"].reshape(horizon, n_firms, -1)
    cash_base = (
        arrays["nominal_amount_sold_in_lcu"]
        - arrays["total_wage"]
        - purchases.sum(axis=2)
        - arrays["taxes_paid_on_production"]
        - arrays["corporate_taxes_paid"]
    )
    capital_purchases = np.asarray(firms["total_capital_inputs_bought_costs"]).reshape(horizon, -1).sum(axis=1)
    rate = country["central_government"]["tax_overrides"]["firm_capital_formation_rate"]
    capital_tax = rate * capital_purchases
    np.testing.assert_allclose(capital_tax, np.asarray(gov["taxes_cf"]).ravel())
    np.testing.assert_allclose(df["deficit"].iloc[1:], np.asarray(gov["deficit"]).ravel()[1:])
    debt = np.asarray(gov["debt"]).ravel()
    deficit = np.asarray(gov["deficit"]).ravel()
    np.testing.assert_allclose(np.diff(debt), deficit[1:], rtol=1e-10, atol=0.001)
    bp = country["banks"]["parameters"]
    cfads_errors = []
    for t in range(1, horizon):
        start = max(0, t - bp["firm_loans_cfads_window"])
        cfads = bp["firm_loans_cfads_haircut"] * np.maximum(cash_base[start:t], 0).mean(axis=0)
        saved_cfads = arrays["credit_market_firm_cfads"][t]
        cfads_errors.append(float(np.max(np.abs(cfads - saved_cfads))))
        # Diagnostic alternative: recognize paid direct-TFP expense, preserving
        # the existing capex treatment. This is a frozen-history sensitivity.
        cfads_with_tfp = bp["firm_loans_cfads_haircut"] * np.maximum(
            cash_base[start:t] - arrays["direct_tfp_investment_cash_expense"][start:t], 0
        ).mean(axis=0)
        demand = arrays["target_long_term_credit"][t]
        received = arrays["received_long_term_credit"][t]
        unmet = np.maximum(demand - received, 0)
        reason = arrays["credit_market_firm_lt_binding_reason"][t]
        needy = demand > 1.0
        period_rate = float(np.asarray(banks["interest_rates_on_long_term_firm_loans"])[t].max())
        maturity = bp["long_term_firm_loan_maturity"]
        factor = period_rate / (1 - (1 + period_rate) ** (-maturity)) if period_rate > 0 else 1 / maturity
        # Opening scheduled service comes from the previous closing schedule.
        service = arrays["scheduled_debt_service"][t - 1]
        cap = np.maximum(cfads / bp["firm_loans_min_dscr"] - service, 0) / factor
        cap_tfp = np.maximum(cfads_with_tfp / bp["firm_loans_min_dscr"] - service, 0) / factor
        deposits = arrays["deposits"][t]
        np.testing.assert_allclose(df["firm_credit_demand_long_term"].iloc[t], demand.sum())
        row = {
            "quarter": t,
            "gdp_bn": df["gdp"].iloc[t] / 1e9,
            "deficit_pct_gdp": 100 * df["deficit_to_gdp"].iloc[t],
            "primary_deficit_pct_gdp": 100 * (deficit[t] - df["interest_payments_on_debt"].iloc[t]) / df["gdp"].iloc[t],
            "capital_tax_without_identified_payer_bn": capital_tax[t] / 1e9,
            "firm_deposits_bn": deposits.sum() / 1e9,
            "lt_demand_bn": demand.sum() / 1e9,
            "lt_received_bn": received.sum() / 1e9,
            "lt_unmet_bn": unmet.sum() / 1e9,
            "lt_unmet_dscr_bn": unmet[reason == 3].sum() / 1e9,
            "lt_unmet_other_bn": unmet[reason != 3].sum() / 1e9,
            "lt_borrowers": int(needy.sum()),
            "borrowers_closing_deposits_bn": deposits[needy].sum() / 1e9,
            "nonborrowers_closing_deposits_bn": deposits[~needy].sum() / 1e9,
            "cfads_bn": cfads.sum() / 1e9,
            "cfads_including_tfp_expense_bn": cfads_with_tfp.sum() / 1e9,
            "lt_min_demand_dscr_capacity_replay_bn": np.minimum(demand, cap).sum() / 1e9,
            "lt_min_demand_dscr_capacity_with_tfp_bn": np.minimum(demand, cap_tfp).sum() / 1e9,
            "bank_firm_supply_cap_bn": df["bank_credit_supply_cap_firms"].iloc[t] / 1e9,
            "cpi_yoy_pct": df["cpi_fixed_basket_yoy_change"].iloc[t] * 100,
            "policy_rate_period_pct": df["central_bank_policy_rate"].iloc[t] * 100,
        }
        rows.append(row)
    panel = pd.DataFrame(rows).set_index("quarter")
    weights = arrays["production"][0]
    price_changes = {}
    for key in [
        "price",
        "pricing_mc_smooth",
        "pricing_labour_mc",
        "pricing_material_mc",
        "pricing_markup_mu",
        "tfp_multiplier",
        "wage_offer_level",
    ]:
        series = np.average(arrays[key], axis=1, weights=weights)
        price_changes[key] = {
            "quarter_1": float(series[1]),
            "quarter_50": float(series[-1]),
            "change_pct": float(100 * (series[-1] / series[1] - 1)),
        }
    cash_identity = (
        arrays["deposits"][:-1]
        + cash_base[1:]
        - arrays["direct_tfp_investment_cash_expense"][1:]
        - arrays["interest_paid"][1:]
        + arrays["received_credit"][1:]
        + arrays["operating_revolving_current_draw"][1:]
        - arrays["received_operating_refinance_credit"][1:]
        - arrays["operating_revolving_repayment"][1:]
        - arrays["debt_installments"][1:]
        - arrays["dividend_fund_settlement_debit"][1:]
    )
    summary = {
        "source": str(SOURCE / "FRA_benchmark.h5"),
        "source_bytes": (SOURCE / "FRA_benchmark.h5").stat().st_size,
        "configuration_sha256": hashlib.sha256(configuration_text.encode()).hexdigest(),
        "manifest_git": manifest["git"],
        "hdf5_phi": country["firms"]["functions"]["target_capital_inputs"]["parameters"][
            "target_capital_inputs_fraction"
        ],
        "manifest_scenario": manifest["scenario"],
        "quarters": horizon - 1,
        "firms": n_firms,
        "cfads_replay_max_abs_error_eur": max(cfads_errors),
        "cash_identity_max_abs_error_eur": float(np.max(np.abs(cash_identity - arrays["deposits"][1:]))),
        "capital_tax_cumulative_bn": float(capital_tax[1:].sum() / 1e9),
        "capital_tax_mean_pct_gdp": float(np.mean(100 * capital_tax[1:] / df["gdp"].to_numpy()[1:])),
        "firm_deposit_increase_bn": float((arrays["deposits"][-1].sum() - arrays["deposits"][0].sum()) / 1e9),
        "cumulative_deficit_bn": float(deficit[1:].sum() / 1e9),
        "bank_reserve_increase_bn": float(
            np.asarray(banks["deposits"])[-1].sum() / 1e9 - np.asarray(banks["deposits"])[0].sum() / 1e9
        ),
        "zero_policy_rate_quarters": int((panel["policy_rate_period_pct"] == 0).sum()),
        "mean_deficit_pct_gdp": float(panel["deficit_pct_gdp"].mean()),
        "mean_primary_deficit_pct_gdp": float(panel["primary_deficit_pct_gdp"].mean()),
        "mean_cpi_yoy_pct_q5_to_50": float(panel.loc[5:, "cpi_yoy_pct"].mean()),
        "total_unmet_lt_bn": float(panel["lt_unmet_bn"].sum()),
        "total_unmet_lt_dscr_bn": float(panel["lt_unmet_dscr_bn"].sum()),
        "credit_exposure_history_shape": list(firms["total_credit_exposure"].shape),
        "price_diagnostics_fixed_initial_output_weights": price_changes,
        "cash_flow_bridge_bn": {
            "sales_minus_wages_goods_production_and_corporate_taxes": float(cash_base[1:].sum() / 1e9),
            **{
                key: float(arrays[key][1:].sum() / 1e9)
                for key in [
                    "received_credit",
                    "debt_installments",
                    "interest_paid",
                    "direct_tfp_investment_cash_expense",
                    "dividend_fund_settlement_debit",
                ]
            },
        },
    }
    np.testing.assert_allclose(cfads_errors, 0, atol=0.001)
    np.testing.assert_allclose(cash_identity, arrays["deposits"][1:], rtol=1e-10, atol=0.001)

government_rule = ExpectedGrowthGovernmentConsumptionSetter(consistency=1.0, sectoral_weights="initial_fixed")
government_proxy = _GovernmentConsumptionShockProxy(
    government_rule,
    ShockSpec(name="one_quarter", kind="government_consumption", period=0, magnitude=0.1, mode="multiplicative"),
)
previous = np.array([100.0])
government_path = []
for active in [True, False, False]:
    government_proxy.active = active
    previous = government_proxy.compute_target_consumption(
        previous_desired_government_consumption=previous,
        model=None,
        historic_total_consumption=None,
        initial_good_prices=np.ones(1),
        current_good_prices=np.ones(1),
        expected_growth=0.0,
        expected_inflation=0.0,
        current_time=1,
        exogenous_total_consumption=None,
        forecasting_window=1,
    )
    government_path.append(float(previous.sum()))
np.testing.assert_allclose(government_path, [110.0, 110.0, 110.0])
policy_rule = SmoothTaylorRule(lower_bound=0.0)
policy_proxy = _PolicyRateShockProxy(policy_rule)
policy_proxy.shock = -0.01 / 4
policy_kwargs = dict(
    prev_rate=0.0,
    inflation=0.0,
    growth=0.0,
    central_bank_states=dict(rho=0.9, r_star=0.0, targeted_inflation_rate=0.02, phi_pi=1.9, phi_q=0.4),
    cpi_yoy_inflation=0.0,
    output_gap=0.0,
    time_unit=3,
)
summary["deterministic_shock_probes"] = {
    "government_one_quarter_10pct_shock_targets_zero_growth": government_path,
    "policy_floor_zero_baseline": policy_rule.compute_rate(**policy_kwargs),
    "policy_floor_zero_proxy_negative_100bp": policy_proxy.compute_rate(**policy_kwargs),
    "policy_floor_zero_native_negative_100bp": policy_rule.compute_rate(**policy_kwargs, shock=-0.01),
}
# The saved baseline predates the repair; the live shock probe verifies fix 8.
assert summary["deterministic_shock_probes"]["policy_floor_zero_proxy_negative_100bp"] == 0.0
panel.to_csv(DEST / "quarterly_evidence.csv")
(DEST / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
print(json.dumps(summary, indent=2, allow_nan=False))
