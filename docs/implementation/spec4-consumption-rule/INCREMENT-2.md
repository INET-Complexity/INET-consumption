# Increment 2 — target and dynamic integration

Status: implemented and locally validated on 4 November 2026; technically reviewed
and explicitly approved by the user on 5 November 2026. Implementation baseline: cef2bff0 on feature/spec4-consumption-rule,
with the reviewed Increment 1 adapter/tests/document uncommitted. See the
implementation record below. No Increment 3 work has started.

## Authority, objective and boundary

[PLAN.md](PLAN.md) governs the economic choices, sequence and acceptance gates.
[INPUT-TIMING.md](INPUT-TIMING.md) governs source definitions and economic dates.
[INCREMENT-1.md](INCREMENT-1.md) defines the implemented adapter API and lifetime.
Reconcile material conflicts before coding; do not change a policy to simplify
integration. Names proposed below are local interfaces, not existing symbols.

Wire the reviewed adapter into the existing aligned CreditAugmentedConsumption
path, using its common smoothed base and opening stocks in target/rate/cashflow
terms. Preserve approved maps, residual anchoring, dynamics, guards and accounting.
Deliver separately named conditional probes and executable equation/accounting
evidence. Keep other rules and existing non-aligned contracts working.

No new rule selector, calibration block, policy toggle, fitted values, training
normalization, ratio bounds, floor definition, learner, credit policy or VAT basis.
Do not run a new simulation experiment, refit, alter notebooks or proceed to
Increment 3. Existing deterministic unit/integration tests and saved-fit replays
are validation within this increment, not new macro experiments. No commit,
branch creation, push, PR or merge without user instruction.

## Files and integration seams

| File / symbol | Intended work |
|---|---|
| macromodel/country/country.py: Country construction, _set_household_target_demand, reset | Own adapter and layout/unit identity per run; capture opening stocks, form same-pass payload after the existing producer, collect, normalize rate weights and pass prepared inputs. Clear adapter/run identity on reset. |
| macromodel/agents/households/households.py: compute_target_consumption and target diagnostic persistence | Pass the explicit prepared record through to the existing consumption object; retain current/prev ECM budget handling, housing carve-outs and append/replace semantics. |
| macromodel/agents/households/func/consumption.py: compute_target_consumption, _evaluate_target, _initial_state_epsilon | Consume prepared inputs coherently, preserve wrappers and legacy calls, add explicit conditional operators and expose decomposition/guard diagnostics. |
| macromodel/agents/households/consumption_inputs.py | Reuse the reviewed API. Only narrowly required integration fixes, with new regressions and disclosure; do not redesign its economic contract. |
| macromodel/agents/households/households_ts.py | Only if required to initialize/reset newly persisted diagnostic series; verify actual existing series factory and conventions before editing. |
| tests/test_macromodel/unit/test_country and tests/test_macromodel/unit/test_agents/test_households | Two-pass/reset/source wiring, target/probe/dynamic identities and error regressions; retain Increment 1's tests. |
| tests/test_macromodel/unit/test_household_repairs_integration.py | Extend existing accounting checks only where needed for integrated inputs. |
| docs/implementation/spec4-consumption-rule/INCREMENT-2.md | Record final interfaces, exact verification commands/results, replay provenance and remaining limits. |

Refresh GitNexus context/impact before existing-symbol edits and report HIGH/CRITICAL
impact before proceeding. Keep changes in the named seams; report any need to
expand the scope. TimeSeries core, PI producer and credit-market rate definitions
are references, not planned behavior changes.

## Runtime owner and two-pass sequence

1. Construct one ConsumptionInputAdapter per Country/run, without global state.
   Use a stable owner-supplied layout key identifying run, household ordering and H;
   a same-size population reorder must invalidate the key. Declare source currency
   and population scale from the actual initialized sources, not mixed test-fixture
   defaults or a numeric-size guess. No new user-facing configuration is required.
2. At entry to the first ordinary aligned target pass, before financing/drawdown
   planning, capture current opening household assets/ledger debts and pre-credit
   principal exposures. Use period_id=len(households.ts.historic("income")), matching
   the uncertainty producer. Explicit startup-only fixtures may use period 0.
   Duplicate/backward capture errors remain errors; do not silently recapture.
3. Let existing S/B recording and the existing PI/uncertainty producer run once for
   this pass. From that one ready learning_inputs result, copy the tagged M/q payload.
   Never rerun the producer or reconstruct q from target diagnostics.
4. Collect the complete ready B/S/selected-price paths with explicit current price.
   Declare aligned for exactly K prices or missing_current for exactly K-1 prices
   at the traced source seam; reject other counts. Do not repair a full-history
   current-price disagreement or discard rows to manufacture alignment.
5. Compute borrowing-rate weights from prepared annual_b and opening principal,
   retaining existing per-pass contractual-rate selection and inflation conversion.
   Pass one explicit consumption_inputs record through Country -> Households ->
   CreditAugmentedConsumption. Reject absent/mismatched records on the integrated
   aligned monetary-income path rather than silently reverting to old inputs.
6. On the replacement pass, use the same period/layout opening snapshot. Replace
   current S/B/PI/diagnostics, recompute b/p/maps/rate weights and target from revised
   flows, and retain the existing previous-budget and uncertainty anchors. Do not
   recapture post-housing tenure, newly granted loans or mutable assets.
7. A later first pass captures the latest completed stocks. Country.reset clears
   snapshot and identity metadata as well as the rule's existing residual/runtime
   caches through the established reset mechanism. Deep-copied runs have independent
   adapter lifecycles, intended fixed residuals and RNG state. Reinitialization must
   not encounter stale snapshots or silently extend the residual vector to a new H.

Gate integration through the existing rule/input capabilities, not country names
or a new selector. The dispatch contract below is confirmed by the user on
4 November 2026. It applies to the existing CreditAugmentedConsumption rule;
other rules retain their existing contracts.

| Aligned behavioural income | Income-belief learning | Ordinary target behavior |
|---|---|---|
| Enabled | Enabled | Integrate the adapter using M/q from the existing ready producer result. Missing/stale/mismatched payloads are errors, never a reason to fall back. |
| Enabled | Disabled | Preserve the existing consumption behavior and input contract. Do not activate the adapter, invoke learning to manufacture inputs or invent M/q. |
| Disabled | Enabled | Preserve the existing non-aligned learning and consumption contract; do not reinterpret its PI signal as an aligned monetary payload. |
| Disabled | Disabled | Preserve the existing non-aligned consumption contract without adapter activation. |

For assume_zero_growth, preserve the existing fixed-budget bypass and its existing
producer schedule; do not add a learner call solely to manufacture adapter inputs.
Do not install an adapter snapshot for a bypassed target. Test bypass/ordinary
transitions and reset so later ordinary calls cannot consume stale snapshots.
Standalone callers on the integrated aligned-and-learning path supply a validated
prepared record. Other standalone and legacy numerical helper calls retain their
explicit prior contracts; they are not fallbacks for missing integrated inputs.

## One input contract throughout the long-run target

Use the ConsumptionInputs values directly; do not deflate them again or run the
old positive-only geometric-average helper on the integrated path. For quarterly
runtime a=4; general unit fixtures use a=12/months_per_period.

```
b = consumption_inputs.income_base_real
annual_b = consumption_inputs.annual_income_base_real
p = consumption_inputs.pi_log_ratio_rebased
LFA, IFA, HA = opening_real asset values (HA is main residence only)
NLA = opening_real.net_liquid_assets
raw_ratios = [NLA/annual_b, IFA/annual_b, HA/annual_b, LFA/annual_b]
```

Apply the existing frozen bounds before BOTH coefficient maps and wealth terms.
Alpha uses NLA/IFA/HA; gamma uses the separate LFA map. Recompute both on every
ordinary evaluation using beginning-of-t stocks (closing t-1), including revised
second-pass b. Both passes reuse these stocks; revised b can change alpha and
gamma1. Do not recompute training
quantiles, remove the retained LFA support cap or freeze maps across runtime dates.

Keep principal-only borrowing exposures separate from arrears-inclusive debts:

```
r_cons = ((1+i_cons_period)/(1+expected_inflation_period))**a - 1
r_mort = ((1+i_mort_period)/(1+expected_inflation_period))**a - 1
r = (r_cons*opening_real.consumer_principal
     + r_mort*opening_real.mortgage_principal) / annual_b
cashflow_index = annualized_contractual_consumer_repricing_delta
                 * opening_real.consumer_ledger_debt / annual_b
W = gamma*NLA_ratio_used + .022*IFA_ratio_used + .013*HA_ratio_used
log_C_star = log(b) + alpha0 - .72*r + alpha*p + F(W,1-alpha) + epsilon
```

F is the existing logarithmic wealth-tail transformation at tau=.95. Reuse the
actual bounded implementation. The normalized r is formed once; passing it to the
evaluator must not trigger another income division. Existing contractual-rate
selection (including replacement's opening schedule) and cached repricing timing
stay intact. Its existing period repricing delta is annualized linearly by a once,
not converted again using Fisher compounding.

The active level multiplier becomes b; all wealth, rate and cashflow denominators
use annual_b. Raw current B remains inspectable and is not overwritten in source
histories. Original q/individual/common components remain distinct from the new
denominator adjustment. Check log(b)+p = log(M_nominal/d_current)+q.

## Residuals, dynamics, guards and accounting

Back out epsilon once on the unshocked initial evaluation using the SAME prepared
deterministic target as ordinary evaluation:

```
epsilon_i = log(C_desired_previous_real_i) - log(C_star_deterministic_i)
delta_log_C = .56*(log(C_star_real)-log(C_desired_previous_real))
              - .003*cashflow_index - .005*delta_uncertainty
guarded_delta_log_C = clip(delta_log_C, -.5, .5)
```

Preserve existing invalid-anchor/numerical-floor handling and disclose those rows;
do not turn them into new random draws. Keep epsilon fixed across both passes and
future periods; do not re-anchor after a shock. Fitted sigma does not enforce the
backed-out dispersion and changing the intercept offsets at the initial anchor.
No lognormal mean correction or once-drawn-residual comparison belongs here.

Retain beta1=0 and preserve the existing nominalizer, numerical target/log guards,
positive desired-budget floor, rate-change and uncertainty units/timing. Previous
desired budget is already real: use current(t-1) on first pass and prev(t-1) on
replacement. Do not substitute realised consumption or add another deflation.
Initial uncertainty change stays zero; denominator rebasing does not change its
variance. Diagnose the raw long-run target separately from the guarded desired
budget; do not call _evaluate_target's existing dynamic return value raw C_star.

Preserve existing gross total-target -> cash/imputed-rent carve-outs -> gross
market target -> VAT-exclusive goods demand -> financing/settlement boundaries.
Keep nonnegative market/goods floors and the current nominal conversion, including
expected inflation where already applied. Main-residence HA selection must not
remove other-property rental receipts or duplicate a housing carve-out. No floor
flag books a cash receipt; realised spending may differ from desired spending.

Persist inspectable b/annual_b, original q/M, rebased p/adjustment, window size and
both upstream/smoothing floor diagnostics, opening exposure values, raw/used wealth
ratios, cap flags, alpha/gamma, W/tail contribution, r/rate contribution, epsilon,
raw log target/numerical guard, ECM/cashflow/uncertainty terms, pre/post growth guard
and its flag. Follow existing diagnostic schema/naming conventions. First pass
appends and replacement overwrites; initialize/reset series coherently. Do not
persist another PI learning history or double-count counts by appending both passes.

## Distinct response operators

Ordinary runtime evaluations update maps and consume history-derived b. Conditional
probes are explicit counterfactual evaluations; they do not edit source histories,
rerun learning, update residual caches or stand in for a runtime date transition.
Implement locally named probe helpers in the existing consumption module; coefficient
overrides are probe-only arguments, not new active policy controls.

| Operator | Fixed quantities and perturbation |
|---|---|
| Notebook conditional income | Hold log Yp, alpha, gamma, opening stocks, epsilon and training references fixed. Perturb the declared period-income base b by T; annual base becomes a*(b+T), p changes by -log(1+T/b). Re-evaluate ratio clipping. Hold contractual rate levels fixed and recompute their debt/income weights where this channel is enabled; set extra channels to zero for static notebook parity. |
| Conditional asset | Hold b, Yp, gamma, epsilon and references fixed. Change the declared opening stock/account component, rebuild NLA, reclip ratios and update alpha. An LFA addition raises NLA one-for-one when debts are held fixed. Recompute any affected debt-rate weight if the perturbed component is debt; do not silently neutralize alpha. |
| Historical formula diagnostic | Keep its fixed denominator/fixed supplied-p conditioning explicitly labelled, evaluated at the new baseline b. Perturb the level operand using an explicit probe context so prepared inputs do not accidentally overwrite it and produce a zero diagnostic. Preserve its VAT-exclusive dynamic-budget interpretation and held rate input; it is not the notebook fixed-Yp response. |
| Runtime transition | Change the stated current B/S/M/q inputs through the ordinary two-pass/history contract; recompute b/p/maps/weights, reuse opening stocks. Label raw-target and desired-budget responses separately. Financing/realised responses require subsequent accounting or registered experiments. |

For each probe return the perturbed levels, step/units, conditioning and numerator/
denominator definition. Distinguish dC_star/dT, desired-budget MPC and historical
VAT-exclusive formula diagnostic; do not relabel them all total MPC. Guard/cap
binding must be reported. For conditional income, freeze alpha and gamma1 from
the baseline evaluation. For conditional assets and the .14 stock replay, freeze
only baseline gamma1; recompute alpha from perturbed, reclipped ratios using the
fixed map parameters. Ordinary runtime evaluations update both alpha and gamma1
from the shared beginning-of-t stocks and that pass's b. No probe reruns a producer
or re-estimates a map. Existing diagnostic names
may remain compatibility aliases only with their exact response object documented.

## Frozen-fit translation replay

Use the original 9,679 retained households, weights, input hashes and frozen
settings recorded in the vault's outputs/implementation/2026-11-04-spec4-audit/
calibration-audit-record.json and the saved results-a8932a0a7eeb artifact (notebook
commit 4783467; manifest SHA256
93af6de592444d11270895fae48b3bf68fd96302ad8ec6adc5c033695bb5dae2).
Resolve paths from the available artifact; record actual hashes in delivery. Do
not regenerate a fit or silently substitute another sample if evidence is missing.

Reproduce uncapped notebook local and finite income/stock responses with zero
runtime-only channels and matched epsilon/consumption anchor. Use rtol=1e-5 and
atol=1e-7 as PLAN requires. Use recorded notebook transfer sizes; recover and pin
them from saved settings/source before executing the replay rather than choosing
new experimental shocks.

Reproduce .14 as the observed-consumption-weighted derivative of log C with respect
to e where LFA_i += e*Y_annual_i, with gamma fixed and alpha updating. It is not a
euro MPC or a quarterly-income injection. Pin survey/observed-consumption weights
and their normalization. Validate against the saved analytical operator and direct
finite differences at e=1e-5,1e-6,1e-7; use the finest stable result and disclose
threshold effects. Target tolerance is atol=1e-7 for the .14 replay; never loosen it
silently. Cap kinks use the appropriate one-sided response rather than a smooth
uncapped derivative. Decompose any discrepancy before claiming translation passes.
No coefficient, intercept or endpoint adjustment enforces .14 in simulation.

## Implementation stages and acceptance matrix

Complete these internal stages autonomously once implementation is authorized:
owner/lifecycle wiring; evaluator input contract; residual/dynamic integration;
diagnostic persistence; explicit probes; frozen replay; accounting/regression review.
These are not separate user approval gates. Stop at the increment-level review.

| Case | Acceptance evidence |
|---|---|
| Startup, positive/zero/negative B, dated floors and varying CPI/C0 | Live caller/evaluator uses independently calculated b and rebased PI in level, wealth/rate/cashflow terms; preserves floor/source units and available-row semantics. |
| Invalid or missing inputs | Missing/stale payload/snapshot, mismatched histories/price/layout/H/units and nonfinite inputs raise clearly; no active fallback to old denominator or stale diagnostics. |
| Two passes, distinct t-2/t-1/t stocks | Both plans use copied t-1 LFA/IFA/main-residence HA and separate debts/principals despite intervening housing/credit changes; revised b updates both maps and normalized weights. Unchanged inputs give matching plans with the same lag budget/epsilon. |
| Dispatch combinations | Exercise all four aligned/learning combinations above and the zero-growth bypass. Only the aligned-and-learning ordinary path activates the adapter; other cases retain their prior behavior and producer schedule. Missing payloads on the integrated path still fail. |
| Period advance, reset, copied run and bypass | Fresh next-period opening values; duplicate/backward capture rejected; no stale layout/period/residual on reset or fresh run; independent adapter lifecycle after deepcopy; zero-growth/non-aligned behavior preserved. |
| Debt coverage, rates and frequency/scale | Nonzero interest arrears affect NLA/cashflow, not principal weights. Hand-computed Fisher rates, single annualization/normalization, current deflator, scales 1/5000 and monthly/quarterly/annual unit fixtures agree. |
| Residual and ECM | Initial anchor equation matches deterministic target; backed-out mean/dispersion and intercept/sigma effects reported; no reanchoring on replacement or controlled input changes; no RNG draw. Correct pass-aware previous real budget and .56 adjustment. |
| Short-run channels and guards | Separate -.003 repricing/-.005 uncertainty contributions, first-period uncertainty zero, zero beta1, unchanged repricing lag; raw/pre/post guard values reconstruct the budget and binding flags agree. Test binding/nonbinding numerical and growth guards. |
| Ordinary maps versus conditional probes | Ordinary b changes update alpha/gamma1 using shared beginning-of-t stocks. Income probes fix both; asset probes fix gamma1/update alpha. Include a stock fixture with nonzero alpha sensitivity so accidental freezing fails; no state/history mutation. Formula, raw target and desired-budget response objects remain distinguishable. |
| Ratio bounds/tail | Interior, below/above, exact-bound and cap-entry/exit states for each ratio; negative NLA with positive IFA/HA; direct bounded-rule perturbations agree, including one-sided kinks and tail entry. No extra MPC/sign clip. |
| Frozen replay | All 9,679 raw targets/maps and matched local/finite operators reproduced within declared tolerances; original weighted .14 operator reproduced and not imposed on live states. Report retained endpoint-profile uncertainty; no new fit. |
| Diagnostics | First append/second replacement, correct shapes/identities/units, no extra learning or common forecast, reset clears new histories, upstream and smoothing floor flags not conflated. |
| Accounting | Initial/later and both-pass total/rent/imputed-rent/VAT identities, goods floors, financing and existing two-period settlement/accounting regressions pass. Rental income preserved; no new transfer/cash booking from adapter floors. |

Run focused new integration/probe tests plus the entire existing adapter, country,
household, credit-market and household accounting regression selection used for
Increment 1. Include relevant configuration and diagnostic-series regressions.
Record exact commands and new counts; earlier 881/120 results do not validate this
increment. Run Ruff check/format, git diff --check, source/impact and complete diff
review. For algebra use rtol=1e-10/atol=1e-12; declare scale-aware monetary tolerances
and retain stricter existing accounting tolerances. Exact identities/counts match
exactly. No tests or replay have been run merely by writing this plan.

## Review deliverable and stop

Provide final APIs/source map, diff, independent numeric examples, live two-pass/
reset evidence, full residual/channel/guard/accounting decomposition, probe
definitions, pinned replay hashes/steps/tolerances/results, test commands/results,
scope analysis and limitations. Report PASS or BLOCKED against the matrix; do not
declare user approval. Then stop for review. Only after integration review may
Increment 3's experiment specification be detailed/registered on user direction.

## Implementation record — 4 November 2026

Result: PASS for implementation/validation; neither increment is marked approved.
Runtime changes are confined to consumption.py, Country ownership/planning/reset,
Households forwarding/reset/diagnostics and household diagnostic-series creation.
Added test_consumption_prepared.py (41 cases) and
test_consumption_inputs_integration.py (11 cases); updated the existing two-period
accounting regression to assert p = individual_q + common_q + denominator_adjustment.
The reviewed Increment 1 adapter/API was reused without modification.

### Implemented interfaces and lifetime

- `consumption_inputs: ConsumptionInputs | None` is passed through Country,
  Households.compute_target_consumption and CreditAugmentedConsumption. It is
  mandatory only on ordinary aligned-and-learning calls. Other dispatch cases and
  zero-growth bypass retain their existing behavior and producer schedules.
- Country owns `_consumption_input_adapter` and a run epoch. Its layout key hashes
  ordered household `corr_individuals` membership plus run/H identity. Units are
  declared from actual household/floor/principal source metadata. The capture,
  same-pass payload, collect, rate weighting and target sequence follows this plan.
- The integrated evaluator consumes b, annual_b, p and current-price opening real
  stocks directly. Existing raw/used ratios, alpha/gamma1, tail, rate, cashflow,
  uncertainty, ECM and guards remain in the same evaluator. The old positive-only
  helper is bypassed only for prepared inputs. Missing records, invalid channels,
  inconsistent annual bases and stale period/pass/layout records fail clearly.
- `evaluate_prepared_target(inputs, *, lagged_real_consumption_budget, epsilon=None,
  real_borrowing_rate=0, consumer_debt_rate_delta=0, uncertainty_delta=0,
  expected_inflation=0)` returns components and the nominal desired budget without
  cache initialization, RNG draws, learning or history mutation. The rate argument
  is already weighted/normalized. `target_consumption_raw_log_target` and
  `target_consumption_raw_target_real` identify the raw target and its numerical
  guard separately from the dynamic return value.
- `probe_prepared_target(inputs, *, kind, step_real, stock_field=...,
  consumption_anchor_real=None, lagged_real_consumption_budget=None, epsilon=None,
  real_consumer_rate=0, real_mortgage_rate=0, consumer_debt_rate_delta=0,
  uncertainty_delta=0)` returns baseline/perturbed components, step/conditioning,
  raw-target MPC, real desired-budget MPC and raw log response. Rate arguments here
  are annual real contractual levels; probe weights are formed once. Kinds are
  conditional_income, conditional_asset and historical_formula. A supplied anchor
  affects only the reported static MPC; it never changes epsilon or runtime sigma.
  The live historical formula diagnostic retains its VAT-exclusive wrapper.
- New per-household diagnostics persist the bases, PI rebase, floor counts/raw
  input, opening exposures, epsilon, raw log target and pre-guard growth alongside
  existing decomposition. First pass appends, replacement overwrites, reset clears.
  Standalone cached residuals also reject a different same-size run/layout.

Required lifecycle checks justified three narrow owner/reset compatibility fixes:
Country preserves original belief priors in the initial household snapshot only
for the integrated capability combination; Households.reset reconstructs that rule
through the existing force_reset mechanism; Country.__deepcopy__ uses standard memo
copying with explicitly copied directly owned TimeSeries containers, preserving
aliases and independent arrays/residuals/rule RNG. TimeSeries core and PI producer
math are unchanged. Full Country copy and subsequent independent reset were tested.

### Executed validation and evidence

With PYTHONDONTWRITEBYTECODE=1, OPENBLAS_NUM_THREADS=1,
NUMBA_CACHE_DIR=/private/tmp/inet-inputs-numba and
MPLCONFIGDIR=/private/tmp/inet-inputs-mpl:

```sh
.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_macromodel/unit/test_country \
  tests/test_macromodel/unit/test_agents/test_households \
  tests/test_macromodel/unit/test_markets/test_credit_market \
  tests/test_macromodel/unit/test_household_repairs_integration.py \
  tests/test_macromodel/unit/test_configurations \
  tests/test_credit_augmented_ratio_bounds.py
```

964 passed, 117 existing warnings, 139.03s. This includes 52 new cases and existing
adapter, floor, bridge, household/configuration/credit and two-period accounting
regressions. Focused new evaluator/integration/accounting selection passed53 in
17.81s. After narrowing the final constructor prior-preservation eligibility, all11
integration cases passed again in7.84s. A final evaluator docstring correction was
nonfunctional; frozen replay was rerun to pin the final source hash. Existing
warnings are pandas concatenation and legacy CES NumPy deprecations.

Saved-fit replay: 35 checks pass over all9679 households, monthly/quarterly/annual
raw targets/maps, both income anchors and local/finite income/stock operators.
Protected manifest and16 input/source hashes match. No fitting or artifact change.
The .14 results at e=1e-5/1e-6/1e-7 are .13999999991198778,
.13999999999751445 and .14000000006141156 (absolute tolerance1e-7).
Maximum check error is4.47720508063576e-8 under response rtol1e-5/atol1e-7.

Full commands/hashes/source mapping/numeric examples and acceptance evidence are
in the vault outputs/implementation/2026-11-04-increment-2/report.md,
frozen-replay.json, replay_runtime.py and residual-anchor-check.json. The replay
pins evaluator SHA256
1bced8f1c7d5f254c02d3bbb255d5f68a18c50af3eb90be4b509098c4bcfeb41.
The synthetic residual check shows sigma does not impose backed-out dispersion,
intercept offsets at the initial anchor, and zero initial raw-target gap.

Ruff lint/format and whitespace checks pass. GitNexus refreshed index-only and
change scope reviewed: HIGH impact is confined to expected planning/reset/target
flows. Untracked adapter/tests are additionally inspected directly. No YAML,
notebook, PI-producer, credit-policy, TimeSeries-core, calibration or floor-policy
change. No commits, branches, push, PR, new simulation experiment or Increment 3.
Next action at implementation close: review and approval of Increment 2 (subsequently completed on 5 November 2026).

## Approval record — 5 November 2026

User explicitly approved Increment 2 after the technical review found no actionable
findings. Review evidence: vault outputs/implementation/2026-11-05-increment-2-review/report.md;
964 broader tests, 53 focused checks, 35 frozen replay checks and 750 randomized
comparisons passed, as did lint/format/whitespace/source-hash checks. This approval
is specific to Increment 2; it does not infer an unrecorded Increment 1 approval.

User requested fleshing out Increment 3 in [INCREMENT-3.md](INCREMENT-3.md).
That is a detailed proposal for review, not authority to run experiments. No INET
commit, push, PR, macro experiment or refit was performed in recording approval.
