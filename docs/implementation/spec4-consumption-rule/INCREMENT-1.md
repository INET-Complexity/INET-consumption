# Increment 1 — income/state adapter

Status: implemented locally on 4 November 2026; review and approval pending.
The three confirmed audit clarifications are implemented below.
Implementation baseline: INET cef2bff0 on feature/spec4-consumption-rule, following
the floor-unit prerequisite b1db808e. See the implementation record at the end.

## Authority and review boundary

[PLAN.md](PLAN.md) governs the economic choices, full sequence and review gates.
[INPUT-TIMING.md](INPUT-TIMING.md) supplies the traced sources, economic dates and
unit conventions. This document specifies only increment 1. If it conflicts with
PLAN, reconcile the discrepancy before coding; do not select another policy.

Deliver a separately testable adapter and its input/output contract. Active
consumption routing, coefficient evaluation, target/ECM changes and persisted
household diagnostics belong to increment 2. Do not add another active selector,
configuration block or policy toggle. Do not start a simulation experiment,
recalibrate, commit or push without the applicable user instruction. Review this
increment before detailing/implementing the next one.

## Confirmed technical clarifications — 4 November 2026

1. Supply current_consumer_price explicitly. Complete only the declared one-row-
   short history; a complete history must end at the same current planning price.
2. Capture opening stocks once per period. Reject duplicate capture, backward
   periods and replacement without a matching snapshot. Reset clears the state.
3. Supply M and q together in a copied same-period/same-pass PI payload. The
   collector neither reads an old target diagnostic nor reruns the PI producer.

These settle the three increment1 audit findings without changing the economics.
The exact fields, transitions and acceptance cases below implement this contract.

## Files and source boundaries

| File / symbol | Increment 1 work |
|---|---|
| `macromodel/agents/households/consumption_inputs.py` (new) | Typed input/result records, pure history/PI arithmetic, copied opening snapshot and an explicitly invoked collector/lifecycle API. Implemented interfaces are recorded at the end. |
| `tests/test_macromodel/unit/test_agents/test_households/test_consumption_inputs.py` (new) | Independent arithmetic, input-error, snapshot-lifetime and source-collection regressions. |
| `docs/implementation/spec4-consumption-rule/INCREMENT-1.md` | Record implemented interfaces, verification commands and deviations after review. |
| `Country._set_household_target_demand`, `Country.reset`, `Households.compute_target_consumption` | Source/integration seams to inspect and exercise explicitly in fixtures; do not hook the adapter into these active methods in increment 1. Hook and reset integration are increment 2 work. |
| `CreditAugmentedConsumption._evaluate_target`, `_geometric_average_income`, PI learner/bridge, TimeSeries and YAML | Read as references; keep active behavior unchanged in this increment. Do not replace the existing denominator helper yet. |

Use GitNexus context/impact before any existing-symbol edit. If implementing the
adapter requires active-method changes beyond this boundary, report the dependency
and agree the scope before expanding it.

## Inputs and outputs

Use H for households, K for available history rows, a=12/months_per_period and
current planning d=C/C0. The production case is quarterly with four history rows;
monthly/annual cases below test units, not a different window/dynamic policy.

| Field | Shape / units / meaning |
|---|---|
| `period_id`, `pass_kind`, `household_layout_key` | Explicit nonnegative integer planning-period identity, first/replacement pass and owner-supplied run/layout identity. Runtime period tags follow the completed-income row count used by the existing uncertainty producer; startup may use explicit period0. A layout key identifies run, positional household order and H, not invented persistent household IDs. |
| `behavioural_income_nominal_history` | K x H, expected behavioural B through the current provisional/revised row; existing model monetary units. |
| `subsistence_nominal_history` | K x H, period-matched canonical S; already population-expanded by b1db808e. |
| `consumer_price_history`, `price_history_layout` | Positive finite raw selected index levels: K rows with layout=aligned, or K-1 rows with layout=missing_current. No other layout is accepted. |
| `current_consumer_price`, `initial_consumer_price` | Explicit positive finite scalars C_current and C0. Define d_current=C_current/C0, including when the price history is already complete. |
| `pi_payload` | Copied period_id/pass_kind/household_layout_key plus H-vectors monetary_income_nominal M and pi_log_ratio_original q from the same ready learning_inputs result. M positive; q already includes the common term once. Tags must match this preparation call. |
| `opening_stocks` | Copied nominal H-vectors for LFA, IFA, main-residence HA, mortgage/consumer ledger debt and separate mortgage/consumer principal exposures. Tagged with period and household layout. |
| `months_per_period`, `history_window` | Positive period length; explicit history window=4 for quarterly runtime. Use available rows during startup. |

Return one `ConsumptionInputs` record containing H-vectors `income_base_real`,
`annual_income_base_real`, `pi_log_ratio_rebased`, `pi_log_denominator_adjustment`,
`log_permanent_income_proxy_real`, the opening snapshot and real opening values
at the current planning d. Include real NLA formed from the ledger debt components.
Keep original q and M inspectable. Returned diagnostics include window row count,
per-row/per-household floor-hit mask and floor-hit counts. Also retain the existing
upstream behavioural raw-income/floor flags when supplied: B is already floored
on the aligned FRA path, so a second-stage mask alone can be zero despite upstream
binding. Do not reconstruct unavailable historical raw-income flags.

Do not compute alpha, gamma, wealth caps/tail, C_star, ECM, goods budgets, .14,
realized spending or a final normalized rate/cashflow index here. Return the
separate opening principal and ledger exposures needed to form those terms once
b is available in increment 2. Preserve source rates/repricing inputs separately
if the collector carries them; no new rate definition or forecast is introduced.

All records own their arrays. A frozen dataclass alone does not prevent numpy
mutation: copy and make stored arrays read-only, or return defensive copies with
an explicit ownership contract. Never mutate source histories or TimeSeries.

## History alignment and arithmetic

The collector reads expected_behavioural_income, the canonical subsistence series
and the selected Economy consumer-price series. Flow sources must already be
ready for the requested pass. M and q are supplied through the explicit PI payload
below, not independently reread from histories. The collector does not invoke
producers, advance learning, book cash or append/replace histories itself.

1. Check row/household correspondence before taking the latest window. Require B
   and S histories to have the same K rows. Do not shorten mismatched arrays until
   they appear aligned. Take the latest min(4,K) quarterly observations.
2. For price_history_layout=missing_current require exactly K-1 raw rows and
   append current_consumer_price once. For layout=aligned require exactly K rows;
   do not append. After either path, require the final price to match the explicit
   current_consumer_price using rtol=1e-12/atol=0. A full-path disagreement is an
   error, not a reason to overwrite the row or use two different current deflators.
   Reject any other shape/layout combination. Do not guess dates or broadcast
   today's floor backward. With explicit period labels, check matching labels.
3. Normalize selected prices exactly once: d_s=C_s/C0. With all consumed income,
   floor and price operands finite and S_s,d_s positive, compute

```
z_s = max(B_s, S_s) / d_s
log_b = mean_s(log(z_s))
b = exp(log_b)
annual_b = a * b
d_current = current_consumer_price / initial_consumer_price
M_real = M_nominal / d_current
log_yp_proxy_real = log(M_real) + q
pi_adjustment = log(M_real) - log_b
p_new = q + pi_adjustment
NLA_open_nominal = LFA_open - mortgage_ledger_open - consumer_ledger_open
stock_open_real = stock_open_nominal / d_current
```

The PI preservation check is log_b+p_new=log(M_real)+q. Use log arithmetic rather
than exp(q). Do not use B as the old PI denominator, re-add the common forecast,
add sigma-squared/2 or re-estimate beliefs. Zero/negative finite B is floored and
included. Nonfinite values are errors. Compute b in logs; reject nonfinite or
nonpositive results rather than silently replacing them.

## Same-pass PI payload and collection point

Provide an explicit collector interface equivalent to
collect_inputs(sources, opening_snapshot, pi_payload, period_id, pass_kind,
            household_layout_key, current_consumer_price, price_history_layout).
The pure prepare API consumes the validated arrays extracted from that call.

The owner creates a ready PI payload from one learning_inputs result:
M=learning_inputs["income_bridge_current"] and
q=learning_inputs["permanent_income_log_ratio"], with matching period/pass/layout
metadata. Copy both arrays together. The payload is required on the aligned
monetary-income path. Reject missing or mismatched tags, dimensions or values;
do not fall back to an earnings-only interpretation or previous target diagnostic.

Current source has q local in Country._set_household_target_demand before target
execution. Therefore the future hook sequence is: capture opening stocks before
intervening transactions; let the existing flow/PI producers run once for that
pass; form the ready payload from their result; collect/prepare inputs; then
(in increment2) evaluate the target. Do not reconstruct q from
last_target_consumption_components or household target diagnostics. Do not call
current_income_belief_learning_inputs from the collector: that producer updates
the uncertainty runtime state. Replanning has its own replacement-pass payload.

Increment1 tests exercise this sequence explicitly without installing live hooks.
Increment2 passes the payload at the traced call site. No new persistent PI
series, another common-forecast addition or learner update is required here.

## Opening snapshot and two-pass lifetime

The adapter has an explicit `capture_opening`, `prepare` and `reset`
lifecycle. Keep it owned by one run; no module-global cache or RNG. Runtime hooks
will be added only in increment 2.

| Event | Adapter behavior exercised explicitly in tests |
|---|---|
| First pass for t | Copy HH current LFA/IFA/main-residence HA and ledger debts before financing/drawdown planning; collect pre-credit contractual principal exposures from CreditMarket. Supplied initial balances serve as opening state at startup. |
| Replacement pass for t | Require the same period/layout snapshot; recompute b and PI from revised current flows, reuse the opening nominal arrays. Do not recapture post-housing tenure, loan or asset states. |
| First capture for a later period | Require period_id greater than the captured period; replace with copied latest completed balances. Live runtime advances one period normally; do not infer a skipped period's observations. |
| Reset/new run | Clear snapshot and period/layout metadata; subsequent first pass captures fresh state. Do not carry residuals or RNG state in this adapter. |
| Duplicate capture for the same period, even with pass_kind=first | Raise ValueError before reading/replacing source balances. Leave the stored opening snapshot unchanged, including after intervening source mutations. |
| Backward capture; replacement without capture or for another period; changed H/layout | Raise ValueError rather than infer balances or reuse arrays from another population/run. Reset is required to start a fresh run. |

capture_opening is the only operation that captures/replaces opening exposures;
prepare and collection must never implicitly recapture them. Repeating pure
preparation from a valid snapshot does not authorize a second capture. A first-pass
collector entry that would capture again is a duplicate-capture error. The owner
supplies run/layout identity explicitly; H alone cannot detect same-size reorder.
Reset clears period and layout metadata so a new run may start at its initial tag.

Keep signed LFA/NLA representable; no new asset-domain restriction or regressor
clip is introduced. Preserve existing producer validation of principal/debt and
other stock domains. Do not infer income-scale compatibility from numeric size;
require consistent source units/metadata and demonstrate invariance in fixtures.
The selected main-residence value comes from the opening value series, not a new
post-transaction owner mask. Rate weights remain principal-only; NLA and cashflow
retain their arrears-inclusive ledger-debt coverage.

## Work sequence within this increment

1. Define typed records and field/unit validation; document ownership semantics.
2. Implement the pure aligned-history and M-based PI transformation.
3. Implement explicit snapshot lifecycle and the read-only source collector.
4. Exercise the collector on named production-source fixtures and the two-pass
   sequence; verify it does not call or change the active target calculation.
5. Run the checks below, review the diff/source mapping and record evidence.
   Stop for increment review; do not activate the adapter or advance to increment 2.

## Tests and acceptance gate

| Case | Independent expected result / failure |
|---|---|
| Known positive four-row history; one/two-row startup; five-row history | Hand-calculated geometric mean of the selected rows; fifth case excludes only the oldest row. |
| Zero/negative B; dated floors; already floored B | Every available quarter contributes after its own floor; no positive-only omission and no second population expansion. Check upstream versus smoothing floor diagnostics separately. |
| Varying CPI and C0 != 1 | Nominal income/floor divided by each row's C/C0 once; expected b from hand real values. |
| Explicit missing_current and aligned price layouts | Append only for K-1 missing_current rows; accept K aligned rows only when the final price matches the supplied current price at rtol=1e-12/atol=0. Test mismatched full-path price, wrong mode/count, missing floor rows, labels, empty history and household dimensions, including nonunit C0. |
| Nonfinite B/S/M/q/prices; invalid positive quantities | ValueError identifying the input; reject nonpositive S, M, C0 and price levels. No nan filtering or silent fallback. |
| M != B, nonzero individual/common q, extreme finite q | Preserve log PI level without exp(q); changing B alone changes the denominator, not the reconstructed M-based numerator. |
| Common monetary scaling at 1 and 5000 | b and opening real amounts scale equally; p_new and stock/annual_b ratios are invariant. Explicitly inconsistent unit metadata is rejected; no magnitude heuristic. |
| Monthly/quarterly/annual flows | Annualization gives the same matched annual-income base and stock ratios. Do not reinterpret the quarterly window as a new fitted policy. |
| Distinct t-2/t-1/t values, changed tenure and loan principals | Capture opening t-1; preserve ledger/principal coverage and main-residence-only HA across replacement. |
| Revised second-pass income, mutated source arrays, next period/reset | Recompute b/p_new, retain copied opening exposures, refresh only on the next first pass, clear on reset; no source/result aliasing. |
| Duplicate first capture after source mutation; backward capture; same-H layout change | Each raises before replacing the snapshot. Stored opening values stay unchanged. Replacement without matching capture also fails; reset permits a fresh run/initial period. |
| Same-pass PI payload versus stale diagnostics | Give first/replacement payloads different M/q and leave target diagnostics deliberately stale. Use only the matching payload, reject cross-period/pass/layout payloads, and assert collector never invokes the producer or changes uncertainty/learning state. |
| Collector and active-state isolation | Source series lengths/values, active target outputs, residual/learning state and RNG are unchanged by explicitly collecting inputs. |

Use synthetic deterministic cases as primary arithmetic oracles and existing
country/household fixtures for source mapping. No new fit or full macro run is
needed. Relevant existing floor-unit and income-bridge regressions must pass.
Record exact tolerances for floating calculations, with rtol=1e-10/atol=1e-12 as
the default for dimensionless/algebraic cases; use declared scale-aware absolute
tolerances for monetary amounts. Exact counts/period identities must match exactly.

Before delivery run the focused adapter tests, existing floor-unit and monetary-
income bridge regressions, relevant household/country tests, Ruff lint/format and
git diff --check. Refresh GitNexus where needed and inspect expected changed files.
Do not claim the earlier 809-test result verifies this new adapter. No acceptance
claim for target MPCs, aggregate .14 or realized spending belongs to increment 1.

## Review deliverable

Provide the input/output API, file diff, source mapping, independent arithmetic
examples, test commands/results, error/lifetime behavior and any remaining limits.
Review against PLAN and INPUT-TIMING. Resolve substantive ambiguities before code
that depends on them. Later-increment plans are expanded one at a time after the
preceding review; this file does not authorize their implementation.

## Implementation record — 4 November 2026

Implemented locally against cef2bff0. Result: PASS for the increment's implementation
and validation gate; user review/approval remains pending. No policy deviations.
PLAN and INPUT-TIMING remain the economic/source authorities; their preparation-era
pending-status statements are historical. Active target and reset wiring remain
increment 2 work.

### Implemented interfaces

- `MonetaryUnits(currency, population_scale)` declares common source units. The
  owner supplies matching metadata for income, floors, stocks, principal and PI;
  the adapter never infers compatibility from numeric size.
- `OpeningStockValues` holds separate H-vectors for liquid/illiquid assets,
  main-residence wealth, mortgage/consumer ledger debt and principal. Its
  `net_liquid_assets` property subtracts the ledger debts, retaining signed NLA.
- `OpeningSnapshot(period_id, household_layout_key, units, nominal)` owns the
  immutable opening stock record.
- `PermanentIncomePayload.from_learning_inputs(result, period_id=..., pass_kind=...,
  household_layout_key=..., units=...)` copies M/q from the one ready producer result.
- `IncomeHistories` validates complete B/S/price shapes and optional matching row
  labels before window selection. The quarterly window must be four. Optional raw
  and floor-flag histories remain available under `upstream_*_history` fields.
- `prepare_inputs(histories, opening_snapshot, pi_payload, *, period_id, pass_kind,
  household_layout_key) -> ConsumptionInputs` performs only smoothing, annualization,
  M-based log rebasing and current-price opening-stock deflation. Original M/q remain
  in `pi_payload`; floors/counts in `floor_hit_mask`/`floor_hit_counts` cover the selected
  window, with counts per household. Full supplied upstream diagnostics stay in
  `histories`, without reconstructing unavailable observations.
- `ConsumptionSources(households, economy, credit_market, income_units,
  subsistence_units, stock_units, principal_units)` declares production objects.
- `ConsumptionInputAdapter.capture_opening(sources, *, period_id,
  household_layout_key, pass_kind="first")` captures explicitly. `prepare(...)`
  uses the stored snapshot; `collect_inputs(sources, pi_payload, *, period_id,
  pass_kind, household_layout_key, current_consumer_price, price_history_layout,
  history_window=4)` reads ready histories and delegates to preparation.
  `opening_snapshot` is inspectable; `reset()` clears it. Neither preparation nor
  collection implicitly captures or replaces opening stocks.

Produced records are frozen; arrays are copied onto immutable byte buffers. They
cannot be mutated or made writable. Deep-copying an adapter preserves independent
lifecycle state while safely sharing immutable input records.

### Verified source mapping

Opening household `ts.current`: `liquid_financial_assets`,
`illiquid_financial_assets`, `wealth_main_residence`, `mortgage_debt`,
`consumption_loan_debt`. Principal comes from tuple positions 1/3 of
`CreditMarket.household_contractual_debt_rate_components(use_opening_schedule=False)`.
No other-property wealth or post-transaction tenure mask is used.

Flow histories: household `expected_behavioural_income`, Economy
`subsistence_consumption`, and the price series selected by
`consumer_price_level_series_name()`. C0 uses `initial_consumer_price_level()`;
period length uses `time_unit`. Scalar price rows `(K,1)` are validated/converted to
`(K,)`; multicolumn rows are rejected. Optional upstream raw/floor/component flags
use the existing `expected_behavioural_income_*` series. Same-pass PI is supplied
explicitly, never reconstructed from stale target diagnostics or a producer rerun.

### Verification

With `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1`,
`MPLCONFIGDIR=/private/tmp/inet-inputs-mpl` and
`NUMBA_CACHE_DIR=/private/tmp/inet-inputs-numba`:

```sh
.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_macromodel/unit/test_country \
  tests/test_macromodel/unit/test_agents/test_households \
  tests/test_macromodel/unit/test_markets/test_credit_market \
  tests/test_macromodel/unit/test_household_repairs_integration.py
```

881 passed (72 new adapter cases), 106 warnings, 126.63 seconds. This includes the
floor-unit, monetary-income bridge, behavioural-interest and two-period accounting
regressions. A focused adapter/floor/bridge/interest run also passed all 120 cases.
Warnings come from existing pandas concatenation and legacy CES NumPy calls.
Ruff `check --no-cache`, `format --check --no-cache` on both new files and whitespace
checks pass. New-file whitespace was also inspected with `git diff --no-index --check`.

GitNexus refreshed with `analyze --index-only`; adapter impact is LOW, one importer
(the new tests), zero affected execution flows. `detect_changes(scope="all")` cannot
report these untracked new files, so its empty result was supplemented by indexed
context/impact and direct file/status review. INET changes are exactly the new
adapter, new tests and this implementation record. No commit, push, branch creation,
PR, experiment or recalibration. No live-target/MPC/.14/spending acceptance claimed.

Next action: review and approval of increment 1. Do not advance to increment 2.
