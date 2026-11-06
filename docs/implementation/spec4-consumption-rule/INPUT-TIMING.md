# Consumption inputs and planning dates

Source trace: 4 November 2026, originally at INET `fff0e869`; current implementation baseline `b1db808e` includes the committed floor-unit repair. Final consistency review completed on 4 November. Active FRA uses CreditAugmentedConsumption, aligned behavioural income, income learning and the feasibility resolver. The income/state adapter is the first implementation increment after this preparation review and is still pending. Approved contracts describe requirements; source and historical evidence are distinguished below.

## Definitions and units

Let H be households, C_s the selected consumer-price index, C_0 its initial value, d_s=C_s/C_0, and a=12/time_unit (a=4 quarterly). Monetary arrays are H-vectors in existing model monetary units. Keep stocks and income on the same population scale; do not divide gamma by a or independently descale wealth ratios. Nominal/real conversion uses the same d for numerator and denominator. Training references and ratio bounds stay fixed.

The selected price series comes from Economy.consumer_price_level_series_name(), not a hardcoded CPI. Before goods clearing its current row is the latest available planning price; it is not the not-yet-observed end-t CPI. Income history has a newly appended t row; current code fills an exactly one-row-short price history with this planning price. Both planning passes revise that same income row, not two different quarters.

Historical normalization must be consistent with current conversion: normalize the supplied price-index history by C_0 before using it with base-period real b. Current _geometric_average_income divides by raw supplied index levels, whereas _evaluate_target divides by C/C_0. C_0=1 hides this difference. Add a nonunit-C_0 acceptance case; normalize once, not twice.

## Input/timing table

| Input | Current source / construction | Economic date and both passes | Required conversion / adapter |
|---|---|---|---|
| Quarterly behavioural income B | HH expected_behavioural_income; Country._record_behavioural_income -> reconcile_behavioural_income | t expectation appended before first target, overwritten after labour before second; t0 seeded at construction | y_s=B_s/d_s. Use the history of this expected series, not realised cash income or financing_income. |
| Behavioural income components | N_raw=expected_income_employee+expected_income_social_transfers; P=income_rental+expected dividend/financial cash receipts+positive deposit receipts; borrower interest I; imputed rent R | Pass-specific income component snapshot. Opening loan interest preview is reused on second pass; deposit receipts/rent may be refreshed | B=max(sqrt(max(N_raw,0)*max(N_raw+P,0))-I+R,S). Component and final-floor flags already exist. The floor creates no cash receipt. |
| Quarterly subsistence floor S | Economy subsistence_consumption from half net monthly SMIC * time_unit * household consumption units, indexed by C/C_0 | t0 initialized; append alongside first expected-income row, replace on second pass | s_s=S_s/d_s. Pass the full period-aligned floor history, not today's floor broadcast backward. Current target API passes only today's vector. The committed floor is already in model monetary units; preserve it without another expansion and verify scale compatibility in adapter fixtures. |
| Smoothed quarterly base b | Currently _geometric_average_income(expected-income history) | Window through the current pass's t observation; previous entries unchanged by replanning | Required b=exp(mean(log(max(B_s,S_s)/d_s))) over latest min(4,available) quarters. Reject nonfinite operands; preserve every available quarter. Quarterly Y_annual=4*b; generic a*b. |
| Monetary current income M | current_income_belief_learning_inputs -> monetary_income_moments -> income_bridge_current | Reconstructed from that pass's same component snapshot | M_nom=max(max(N_raw,0)+P,S); M_real=M_nom/d_t. Excludes imputed rent and separate borrower-interest subtraction. This is NOT B. |
| Existing PI signal q | permanent_income_log_ratio = bridge mean + common forecast once | Same-date planning signal; posterior comes from completed-income learning state, common forecast may be refreshed | q=E[log(M_projected/M)]+q_common. Preserve the bridge's implied log-income level by p_new=q+log(M_real)-log(b). Do not substitute B as the old denominator. |
| LFA | HH liquid_financial_assets | Active resolver: current row is closing t-1 at first and second target; next wealth row appended later. Existing evaluator receives prev instead | Copy opening nominal H-vector on first pass before financing planning; reuse both passes. Deflate by current planning d_t for ratios. |
| IFA | HH illiquid_financial_assets | Same stock-row timing as LFA; current update_wealth appends after realised consumption | Copy opening current row; ratio=(IFA_open/d_t)/(a*b). No extra annualization of stock. |
| Main-residence HA | HH wealth_main_residence; helper currently defaults lagged housing to prev | Wealth series appends at update_wealth, after both plans; property ownership/tenure and rent can change between passes | Copy the opening main-residence value directly. Do not reselect using post-transaction ownership for the second target. Exclude other-property value from HA but preserve its rental income. |
| Mortgage debt for NLA | HH mortgage_debt | Resolver leaves opening HH row through second target; authoritative settlement appends later. Legacy path may append earlier | Snapshot opening HH debt and reuse; includes mortgage interest arrears in current ledger getter. Do not use mutable post-grant/settlement debt. |
| Consumer debt for NLA/cashflow | HH consumption_loan_debt | Same resolver timing as mortgage debt; includes consumer interest arrears in current getter | Snapshot opening ledger debt; NLA=LFA_open-mortgage_open-consumer_open before regressor bounds. Cashflow retains its existing ledger debt coverage with the approved new date/base. |
| Borrowing-rate principal weights | CreditMarket.household_contractual_debt_rate_components; first pass current loan states, second use_opening_schedule=True | Separate contractual consumer/mortgage rates and principal, not the HH arrears-inclusive totals | Preserve current principal-only weighting and rate definitions, capture/verify opening exposures. Annual real rate=[(1+i_period)/(1+pi_expected_period)]^a-1. Divide nominal weighted principal by d_t*a*b. Do not silently add interest arrears to rate weights. |
| Nominal consumer repricing change | CreditMarket.consumer_debt_rate_delta_for_cacf | First-pass value cached in Country, reused on second | Existing period delta multiplied by a (linear annualization). Cashflow index=annual_delta*(consumer_ledger_debt_open/d_t)/(a*b), coefficient -.003. |
| Uncertainty change | current_income_belief_learning_inputs: monetary bridge variance + common variance | Period keyed by len(HH income history); first-period delta zero; both passes retain the same previous-period anchor | Rebase of log-income denominator adds a deterministic constant and does not change conditional variance. Preserve uncertainty policy/coefficient -.005. |
| Previous desired real budget | HH cacf_real_consumption_budget | First target uses current(t-1), appends provisional t; second reads prev(t-1), replaces provisional t | Already real: no new deflation. Preserve this deliberate pass-aware accessor; it differs from stock accessors. |
| Fixed residual | CreditAugmentedConsumption._initial_state_epsilon, _household_epsilon | Backed out on first unshocked evaluation, cached; not redrawn or recalculated on replanning | Recompute the initial anchor under the new deterministic inputs once on a fresh run; keep within-run residual fixed. Fitted sigma does not impose its dispersion. |

## Permanent-income bridge: resolved denominator

Write X for the existing Gaussian log non-property-income belief. The source defines

```
M_nom = max(max(N_raw,0) + P, S)
M_projected_nom(X) = max(max(N_raw,0)*exp(X) + P, S)
q = E[log(M_projected_nom(X)/M_nom)] + q_common
log_yp_proxy_real = log(M_nom/d_t) + q
p_new = log_yp_proxy_real - log(b)
```

This preserves the current monetary bridge's log-level proxy (including its existing common-forecast approximation), without re-estimating beliefs or adding a lognormal mean correction. exp(log_yp_proxy_real) is a geometric/log-level proxy, not E[M_projected]. Working in logs avoids unnecessary overflow. Property receipts remain fixed under the individual projection. The common term is added once; it is a macro proxy, not an exactly matched household monetary forecast.

Earlier generic advice p_new=q+log(current_behavioural_income/b) is incorrect for the active monetary bridge. Its actual old denominator is M. Using B would shift the preserved PI level by log(B/M). Retain q, its individual/common components and the new denominator adjustment as separate diagnostics.

Example using the actual source helpers: N=100, P=40, borrower interest=5, imputed rent=15, floor=20 gives M=140 but B=sqrt(100*140)-5+15=128.321596. These are different income concepts deliberately already present in runtime.

## Runtime sequence and snapshot lifetime

1. Country.update_pre_credit_household_finance_metrics updates benefits and appends expected household components. _set_household_target_demand(False) appends S and B, builds the monetary PI signal and evaluates the first target. Capture opening consumption stocks and principal exposures at entry to this first target, before its financing/drawdown planning.
2. Simulation.iterate clears housing and credit, processes housing outcomes and the credit path, and clears labour. Rent, tenure, loan market states and feasibility information may change. Do not infer economic dates from a global current/prev rule.
3. Country.update_post_labour_planning_metrics replaces expected income components after labour, refreshes known receipts, then calls _set_household_target_demand(True). Replace S/B/PI/diagnostics for t. Reuse opening consumption stocks. Recompute b, rebased p, alpha/gamma and target using revised flow information. Previous desired budget and uncertainty lag remain t-1.
4. On the active feasibility-resolver path, settle_authoritative_household_payments runs AFTER the second target and appends household debt/interest rows. On the legacy path process_credit_market_clearing services/appends debt earlier. Support both without changing the approved opening-stock convention.
5. Goods clear; update_realised_metrics records realised income, updates learning, consumption/investment, then update_wealth appends LFA/IFA/housing. Reset opening snapshot on the next first planning pass and on run reset; do not retain it across periods/runs. Full-graph snapshots for paired validation must also copy this state and RNG/residual state.

Current active-path financial/housing prev values can therefore be t-2 at BOTH plans, while current HH stocks remain t-1 until later updates. HH debt and market principal timing/coverage differ. TimeSeries.current/prev are positional, and prev aliases current when only one row exists: startup-only tests cannot expose an extra lag.

## Implementation changes and checks

- Expose aligned expected B, historical S and selected price paths with a single documented d=C/C_0 convention. The aligned FRA income path already floors every recorded B and rejects nonfinite components; applying the same per-date floor during smoothing is idempotent there. Preserve raw-income and floor-hit diagnostics so smoothing does not conceal upstream binding.
- Keep the repaired canonical subsistence source; do not introduce another economic floor or multiply the stored model-unit base again. INSEE expansion occurs once in _select_net_smic_base, before the floor helper. The resolved-state check and 809-test repair verification are complete in b1db808e. Adapter currency/population-scale tests still scale income, S, PI components and stocks together to verify the new construction.
- Pass M (or log_yp_proxy) explicitly into the adapter so it can form the correct p. Do not reinterpret the monetary q as a behavioural-income ratio. Preserve learner/common forecast, uncertainty variance and financing receipts.
- Build real rate weights and cashflow indices after b is known, or pass their unnormalized opening numerators. Dividing an already income-normalized index by another income would double-normalize it. Keep principal rate exposures distinct from arrears-inclusive NLA debt.
- Capture opening H-vector copies, not references or post-housing tenure filters. Preserve main-residence-only HA and rental accounting. Retain existing pass-aware ECM budget persistence.
- Declare whether standalone overrides supply a complete dated history or an explicit conditional probe. Current income_override changes the level but not historic_income; do not use that route to claim ordinary smoothed-income transition parity.
- Required fixtures: distinct t-2/t-1/t stock values; different first/second-pass income; changed tenure/rent and granted/serviced debt; nonunit C_0 and varying CPI; quarter-specific floors; startup and nonfinite/missing rows; M different from B; nonzero individual/common PI; nonzero interest arrears; compatible/incompatible population scales; reset and paired-state copying. Source tracing is complete for the named paths; these executable acceptance checks remain pending.

## Source ledger

Paths are relative to INET. Line anchors refer to source at fff0e869, not this documentation revision.

- macromodel/simulation.py:384,418,436 — iterate and both planning passes.
- macromodel/country/country.py:862,1018,1040 — initial S and per-pass SMIC/CU floor.
- macromodel/country/country.py:1357,1363,1442 — expected-flow passes and post-target settlement.
- macromodel/country/country.py:1604,1678,1801,1848 — B, monetary components, rates and caller arguments.
- macromodel/country/country.py:2152,2326,3726,4236 — debt settlement, wealth update and common PI.
- macromodel/agents/households/households.py:1809,1826,1961,1971,1990,2039 — housing selection, input/ECM/history/stock accessors.
- macromodel/agents/households/households.py:2516,2598,2799,3016,3817 — PI/uncertainty, CU, rent and wealth appends.
- macromodel/agents/households/func/consumption.py:1172,1189,1390,1420,1462,1571,1746 — residual, denominator, units, ratios and target/probe.
- macromodel/util/behavioural_income.py:6; macromodel/util/permanent_monetary_income.py:38 — distinct income definitions.
- macromodel/markets/credit_market/credit_market.py:1888,2476,2484 — principal rate weights versus arrears-inclusive debt.
- macromodel/timeseries.py:80,101 — current/prev semantics.
- macro_data/util/rates.py:43; macro_data/processing/synthetic_population/hfcs_household_tools.py:189,259 — rate conversion and synthetic monetary scaling.

GitNexus context/query and its named process resource confirmed Country -> current_income_belief_learning_inputs -> monetary_income_moments -> _log_income; current source supplies the definitions above. A direct numeric check using the actual pure bridge/reconciliation helpers passed PI log-level preservation and common currency-scale invariance, and demonstrated M != B. The source-tracing step itself performed no new fit or runtime implementation. The subsequent floor repair is committed as recorded below; no registered macro experiment was run and the adapter has not been tested or implemented.

## Historical floor finding before b1db808e

Before the repair, the resolved-state scale check found the following. Initializing the existing
2014 FRA cache with active resolved configuration gives scale=5000, H=5760,
monthly SMIC=1128.70 and a quarterly floor per CU=1693.05. Income/assets are in
expanded monetary units; that same floor must be 8465250 per CU in those units.
At that baseline there was no expansion before the canonical floor helper.
This established the unit mismatch that was subsequently fixed in b1db808e.

The same raw initial behavioural-income rows bind for 65 households at the old
threshold and 218 at the expanded threshold. This is a mechanical comparison,
not a corrected-model simulation. GitNexus marks the canonical-floor change HIGH
risk, with construction and per-pass refresh as direct callers. The shared floor
also affects PI, support and goods shortfall. The canonical repair was completed
before the pending adapter; this paragraph records the historical diagnosis.
Evidence/reproduction: vault outputs/implementation/2026-11-04-floor-units/.

## Floor repair implemented and verified

The floor-unit increment committed as b1db808e passes the population scale to
_select_net_smic_base at construction. It expands only the external INSEE monthly
source; net_smic_base thereafter carries model monetary units, so floor refreshes
reuse it without another multiplication. Already-expanded benefit fallbacks keep
their legacy proxy/frequency treatment. This supersedes the unexpanded-floor
current-state descriptions in the pre-repair trace above.

809 selected country/household/credit tests pass, including15 new scale/frequency/
CPI/fallback/planning/reset regressions and the two-period accounting integration.
Production-config initialized FRA now has model-unit quarterly floor8465250 per CU
and ordinary-unit floor1693.05. No adapter implementation or registered macro run.
Evidence: vault outputs/implementation/2026-11-04-floor-unit-repair/.
Experiment sizes/accounts/seeds/horizon/criteria and the residual comparison remain
deferred as specified in PLAN. This document does not authorize the next increment.
