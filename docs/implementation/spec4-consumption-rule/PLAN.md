# Implement the retained consumption rule in INET

Branch: `feature/spec4-consumption-rule`, created from main `a3d48e5c`.
Final preparation/consistency review: 4 November 2026, against INET b1db808e.
That preparation review changed documentation only. Increment 1 has since been
implemented locally and reviewed; the current planning status is recorded below.

Update the existing `CreditAugmentedConsumption` and
`run_model/config/consumption_paper_parameters.yaml:desired_consumption.credit_augmented`
directly. The earlier separate calibration-block/alternate-selector proposal is
superseded. Work remains on this branch, one reviewed step at a time. No automatic
refitting, push, merge or changes to policy assumptions.

## Current status and Increment 3 follow-up

Canonical floor repair is committed in b1db808e. In country.py, from_pickled_country
passes the population scale to _select_net_smic_base, which expands the external
INSEE monthly level once. net_smic_base is stored in model monetary units;
initialization and both planning-pass refreshes reuse it without another scale.
Already-expanded benefit fallbacks retain their existing proxy/frequency behavior.
No household debt, benefit-payment formula, consumption coefficient or YAML edit.

809 country/household/credit-market tests pass, including 15 new scale/frequency/
CPI/fallback/two-pass/reset regressions and the existing two-period production
household accounting integration. Ruff lint/format and whitespace pass. An
initialized production-config FRA check yields model monthly base 5643500 and
quarterly floor 8465250 per CU, or 1693.05 in ordinary currency units, with 218 initial
raw behavioural-income observations below the corrected threshold. No registered
macro experiment or new fit. User-requested local commits: INET b1db808e and vault
17e835f6; no push or merge. GitNexus change scope was reviewed before committing.
Evidence: vault outputs/implementation/2026-11-04-floor-unit-repair/.

**Increment 1 is implemented locally and technically reviewed.** The adapter,
72 new test cases and implementation record remain uncommitted; cef2bff0
was the implementation baseline. Current re-entry HEAD is 96c897e5.
881 required broader tests passed during implementation; the review reran120
focused tests plus500 independent numeric cases and found no changes required.
Details and evidence are in [INCREMENT-1.md](INCREMENT-1.md).

**Increment 2 was technically reviewed and approved by the user on 5 November 2026.** Its scope,
interfaces, live wiring, probe contracts and passing validation evidence are in
[INCREMENT-2.md](INCREMENT-2.md). The technical review reran 964 broader tests,
53 focused checks, 35 frozen replay checks and 750 randomized comparisons, all
passing, with no actionable findings. Approval does not commit or publish the work.

**Increment 3 implementation is authorized, one task at a time.**
Detailed scope: [INCREMENT-3.md](INCREMENT-3.md). The user confirms exogenous external
funding through existing receipt/LFA links. Task 1 source/account mapping is
complete in [INCREMENT-3-ACCOUNTS.md](INCREMENT-3-ACCOUNTS.md); no new institution
or domestic financing policy is introduced.
It specifies paired-state validation, shocks/accounts, run matrix, measurement,
budgets and gates. The seed-12 smoke and the approved six-case full campaign are
complete. All six cases passed controller completion, source/config/cache and
environment pins, cohort/path completeness, primary-outcome finiteness, and
transfer/LFA booking and account-link checks. The campaign used the approved
178-package environment; all 60 case-artifact hashes match. Results and gate
review are recorded in
knowledge-vault/outputs/implementation/2026-11-05-increment-3-campaign-review/report.md.
**Updated 6 November 2026:** the approved diagnostic recapture supplies the
previously missing main operands. Both revisions' 144 target passes reproduce
within declared tolerances, conditional on captured inputs/residuals; this is
not an independent replay of upstream history formation or initialization.
Branch's independently computed guard contribution passes 288 checks across
96 comparisons. Its earlier zero telescope used a balancing final block and
is superseded by this independent check. Main's separate unguarded/guard split
fails 123/480 checks from cancellation; its additional guarded nine-block
method passes, but includes guard interactions inside each block.

The branch seed-13 LFA sign reversal is traced to funding-floor adjustment:
h8 desired goods cash spending falls EUR 2.246bn, cuts shrink EUR 2.600bn and
top-ups rise EUR 0.044bn, leaving post-floor goods cash spending EUR 0.398bn
higher. Interior-cut households identify most of the shortfall improvement;
full funding mediation lacks saved financing-income/property/investment/service
operands. Actual first-pass post-FICP credit demand was already saved as
household_state.live_credit_requested. The distinct missing
funding_plan.credit_requested reads an absent post-grant carrier attribute;
do not substitute actual requests into that field.

Recoverable reporting is complete: 6,480 response cells, 2,160 seed summaries,
7,200 channel contributions, level/funding and diagnostic transition tables,
and seven inspected plots. Case manifests still say provenance_verified=false
despite controller pin checks. Frozen main also has the identified missing
x5000 SMIC population expansion; the multi-change comparison does not isolate
the quantitative effect of that repair. Economic acceptance remains
inconclusive; no production promotion or coefficient retuning is approved.

Current evidence in the vault:
- outputs/implementation/2026-11-06-increment-3-acceptance/report.md
- outputs/implementation/2026-11-06-increment-3-main-reconstruction/report.md
- outputs/implementation/2026-11-06-increment-3-funding-shortfalls/report.md
- outputs/implementation/2026-11-06-increment-3-guard-audit/report.md
- outputs/implementation/2026-11-06-increment-3-reporting/report.md

The separately registered and explicitly approved residual smoke/full comparison
is complete. All12 full cases pass engineering checks; levels/support and longer-
horizon responses do not support promoting once-drawn residuals. Retain the live
initial_state policy. Final implementation review and acceptance were completed
on6 November2026; see [FINAL-REVIEW.md](FINAL-REVIEW.md). Scientific limitations
remain explicit; no fit, new run, commit, push or merge is authorized by this record.

The two plan-audit contracts are recorded in INCREMENT-2: ordinary evaluations
update alpha/gamma1 from beginning-of-t stocks and pass-specific b; probe freezing
is operator-specific. Aligned income without learning retains its existing
behavior; adapter integration applies to the aligned monetary learning path.
No new policy discussion is required absent substantive implementation ambiguity.
PLAN remains the governing guide; later increments are detailed one at a time
after review.
Statements below describing pre-integration runtime are historical source traces;
Increment 2 supplies the approved integration, as recorded in its implementation record.

## 1. Existing calibration wiring and its verification limits

The active block supplies the retained notebook fit: 9679 French households,
notebook commit 4783467, result results-a8932a0a7eeb.

- Fitted intercept, consumption residual SD, accessibility weights/midpoint and
  separate LFA gamma-map endpoints/midpoint.
- Alpha [.1,.9], fixed IFA=.022 and main-residence housing=.013, transition
  width .36, logarithmic wealth tail tau=.95 and beta1=0.
- Fixed training normalization references; gamma uses LFA/Y, not accessibility.
- Main-residence selection in the household caller.
- Explicit ratio winsorization before both coefficient maps and wealth terms.
  The old lagged-consumption-dependent wealth clip is bypassed when the
  logarithmic tail is configured.
- Persisted raw/used ratios, cap-hit indicators, tail entry and the actual
  nonlinear wealth contribution. Actual balance-sheet assets/debts are unchanged.

| Ratio | Lower | Upper | Source |
|---|---:|---:|---|
| NLA/Y | -7.1160503836326745 | 3.9198632665924547 | Survey-weighted p1/p99, valid pre-trim 2014 pool |
| IFA/Y | 0 | 8.178091110827788 | Same |
| Main-residence HA/Y | 0 | 16.139471789828157 | Same |
| LFA/Y | 0 | 4.515874225190635 | Additional runtime cap at retained-sample support; NOT a percentile-trim threshold |

The first three bounds are the actual trimming thresholds, not quantiles
recomputed after trimming or over live model households. Hard bounds precede
the percentile trim and are documented in YAML. Calibration drops households
crossing any trim threshold; the user-requested runtime translation clips the
regressors instead. The LFA cap remains as currently coded following the user's
"never mind"; that does not make it an empirically estimated threshold or prove
that a cap is necessary for the bounded logistic map. No LFA policy change is
made by this re-audit.

**Earlier calibration-wiring evidence (fff0e869):** 529 household/configuration
tests and Ruff passed. The floor-unit increment's 809-test result is recorded
separately above; neither result establishes pending adapter equivalence.
With identical annual-income states, supplied PI log ratios and epsilon=0,
raw alpha/gamma/log propensity match the notebook for all9679 retained
households (maximum errors about7e-16). No retained household is clipped in
that replay. Monthly/quarterly/annual synthetic scaling and boundary-equivalence
tests also pass.

**Limit:** this verifies the equation on explicit inputs. It does not verify
live income definitions, planning-date stock timing, the runtime formula-MPC
probe, residual initialization, ECM dynamics or realised macro spending. The
replay directly calls `_evaluate_target` with an explicit annual/current-income
state; it does not exercise the active rolling-denominator pipeline.

## 2. Current runtime versus agreed requirements

The table describes code at b1db808e. It is not a list of undecided policies.
Sections 3 and 4 contain the agreed requirements still to implement or verify;
the static fit does not estimate the temporal/dynamic policies.

| Mechanism | Current behavior | What must be established |
|---|---|---|
| Income base | Current income in level/rate terms; positive-only four-period geometric history for wealth ratios | Implement the agreed common smoothed base, historical floors and consistent deflation |
| Runtime formula-MPC probe | Reuses the baseline ratio denominator AND supplied PI log ratio in the perturbed evaluation | A notebook fixed-Yp experiment instead changes p=log(Yp/Y) when Y changes; define and label both response objects |
| Coefficient timing | `_evaluate_target` recomputes alpha/gamma from supplied ratios | Retain ordinary per-evaluation updates; implement the explicitly conditioned probe operators separately |
| Financial/housing stock timing | Traced `prev` wealth inputs can read t-2 at both target passes | Implement shared copies of the opening t-1 stocks; preserve debt coverage and main-residence selection |
| Residuals | `idiosyncratic_persistence: initial_state` backs out epsilon from the initial budget and deterministic target | The configured sigma does NOT enforce the initial residual dispersion; the intercept is offset at the initial anchor |
| Dynamics | ECM .56, real-rate level -.72, cashflow -.003, uncertainty -.005, beta1=0, growth guard ±.5 | Retain these agreed policies; integrate common-base denominators and verify/report each contribution |
| Accounting | Gross total target, existing real-budget state, housing carve-outs, goods VAT conversion, financing/settlement | Verify unchanged identities with the newly wired target |

HFCS `disp_geom_avg_income` is a same-year component blend, not evidence for a
four-quarter temporal smoothing window. The existing rolling denominator is
not a lifetime-fixed initial anchor. Initial-state residuals are not iid draws
from the fitted normal distribution. A changed YAML sigma/intercept alone
therefore does not establish changed initial consumption levels/dispersion.

Source anchors: `consumption.py` `_evaluate_target`, `compute_target_consumption`,
`_initial_state_epsilon` and `_geometric_average_income`; `households.py`
`compute_target_consumption`. Refresh impact analysis before changing symbols.

## 3. Agreed income/state contract and implementation sequence

Source tracing is recorded in [INPUT-TIMING.md](INPUT-TIMING.md), including the
source ledger, both planning passes, income definitions, units, stock dates and
required adapter checks. It distinguishes current code from the approved contract.
The trace resolves the PI denominator as monetary income M, not behavioural B:

```
q = existing permanent_income_log_ratio
M_nom = income_bridge_current
p_new = q + log(M_nom / (C_t/C_0)) - log(b)
```

Preserve the implied monetary log-income level and add the common forecast once;
do not use q+log(B_real/b), re-estimate beliefs or add a lognormal correction.
Current behavioural income is already floored at subsistence and rejects
nonfinite components; same-date floor application during smoothing is idempotent
on this active path. Historical floors still need explicit period alignment.
Use C/C_0 consistently for current and historical deflation; test C_0 != 1.

Opening HH debt includes interest arrears, while contractual borrowing-rate
weights use principal only. Preserve these existing coverage definitions and
align their dates; do not silently force them to the same numeric exposure.

The floor-unit prerequisite is complete in b1db808e: external INSEE SMIC is
expanded once at selection and stored in model monetary units; initialization
and refresh reuse that base. Already-expanded benefit fallbacks retain their
existing proxy/frequency treatment. Scale, frequency, CPI, fallback, both-pass,
reset and affected accounting checks passed. Historical floor-array alignment
is still part of the adapter; do not rescale the repaired base again.

### Agreed opening-stock contract — 4 November 2026

Use the latest completed, opening-of-period t-1 balance sheet for LFA, mortgage
and consumer debt, IFA and main-residence housing in the raw consumption target.
Both planning passes must use the same economic-date stock snapshot, before
current-period consumption, borrowing and portfolio transactions:

```
NLA_open = LFA_open - mortgage_debt_open - consumer_debt_open
C_star_t = f(b_t, yp_t, LFA_open, NLA_open, IFA_open, HA_main_residence_open, r_t)
```

Construct NLA before applying the configured regressor bounds. Use opening debt
exposures consistently in the borrowing-rate weighting and cashflow index;
rate/repricing inputs retain their separately declared timing. This fixes stock
exposures, not all current-period income, belief or rate inputs between passes.

Follow the completed append/replace trace in INPUT-TIMING.md and verify it with
fixtures when changing accessors. Do not replace every prev with current. Capture
or retrieve the opening snapshot so intervening current-period balance updates
cannot contaminate the authoritative pass. At startup identify the supplied
initial balance sheet as the opening state. Main-residence selection must remain
explicit and preserve other-property rental-income accounting.

Acceptance requires distinguishable t-2, t-1 and t stocks in fixtures; both passes
must use t-1. Verify opening snapshots are not aliased to mutable current arrays,
NLA uses the intended debt components, and between-pass transactions do not alter
the consumption stock inputs. This is an approved implementation requirement.
The b1db808e positional-prev baseline has since been replaced by the reviewed
Increment 2 opening-stock wiring; see INCREMENT-2.md for implementation details.

### Agreed runtime income contract — 4 November 2026

User selects a consistent smoothed quarterly income base throughout the raw
long-run target. Gradual transmission through rolling income AND the core ECM
is intended; no additional current-income channel is introduced by this decision.
These are approved implementation requirements, not claims about current code.

Let y_(i,s) be real quarterly behavioural income and s_(i,s)>0 the corresponding
real quarterly subsistence-income floor. For the latest n<=4 available quarters:

```
b_(i,t) = exp(mean_s(log(max(y_(i,s), s_(i,s)))))
Y_annual_(i,t) = 4 * b_(i,t)
p_(i,t) = log(yp_quarterly_(i,t) / b_(i,t))
```

- Include every available quarter, including zero/negative-income quarters after
  applying its subsistence floor. Do not discard nonpositive observations.
- At startup use available quarters up to four. Nonfinite income is an input
  error, not an observation to omit. Require aligned historical deflators and
  period-matched real subsistence floors; record their series/date contract.
- Use b for the quarterly consumption-level multiplier and 4*b for all wealth
  ratios and the real-borrowing-rate debt/income weights. Keep fixed training
  references and the existing regressor bounds.
- Rebase the supplied permanent-income log ratio using the resolved monetary
  M-based formula above. Preserve its implied real quarterly log-income proxy,
  with the common forecast added once. Do not re-estimate beliefs.
- Unemployment benefits should normally keep income positive. The floor is a
  fallback; persist/report its binding frequency rather than assume it never binds.

The intended quarterly raw target, before existing numerical guards, is below.
All stocks/debts in these equations are opening nominal values divided by the
current planning deflator d_t=C_t/C_0. NLA uses arrears-inclusive ledger debts;
D_principal uses the separate contractual-principal exposures for rate weights.
Apply configured wealth-ratio bounds before maps and W.

```
W = gamma * NLA/(4*b) + .022 * IFA/(4*b) + .013 * HA/(4*b)
r = (real_rate_consumer * D_consumer_principal
     + real_rate_mortgage * D_mortgage_principal) / (4*b)
log(C_star) = log(b) + alpha0 - .72*r + alpha*p
              + F(W, 1-alpha) + epsilon
```

Apply the configured ratio winsorization before maps and W. Preserve the real
borrowing-rate term in the long-run target, including annualized real-rate units;
it is distinct from the short-run rate-change cashflow term. Use the traced
opening dates and coverage above. CM supports debt/income weighting; annual-income
normalization and household contractual rates are runtime conventions, and exact
coefficient-scale equivalence to CM has not been established. Using smoothed
income in those weights is an explicit adaptation.

Keep the existing core ECM adjustment speed .56. The fitted beta1 is zero.
Residual anchoring and cashflow/uncertainty additions are retained as specified
in section 4; their decomposition and validation remain required.
At baseline b1db808e, runtime used current income for the level and rate weights,
positive-only history for the rolling denominator, and its supplied PI ratio.
Increment 2 implemented the approved smoothed-income integration and passed its
technical review. The campaign and follow-up target replay have run; main separate-guard
numerical verification and economic acceptance remain open under Increment 3.

### Agreed coefficient-update timing — 4 November 2026

Recalculate alpha and gamma whenever ordinary runtime consumption is evaluated,
using the shared opening balance sheet, that evaluation's smoothed income base
and frozen training references (with configured ratio winsorization). Alpha uses
NLA/IFA/main-residence ratios; gamma uses the separate LFA ratio. If the second
planning pass revises income and hence b, both maps may change despite unchanged
opening stocks. Do not freeze either map for a household's lifetime.

The freezing conventions below apply only to the explicitly conditioned
notebook-parity probes: income probes freeze alpha and gamma; conditional asset
probes freeze gamma and update alpha. Full runtime paired experiments let maps
evolve according to the ordinary rule and report that distinct response object.
Test unchanged-input pass equivalence and map updating when the income base is
revised, plus independent verification of both conditioned probe conventions.

Specify the following experiments separately:

1. **Notebook conditional income response:** hold Yp, alpha, gamma, stocks and
   training references fixed. Thus p changes by -log(1+T/Y); stock ratios change
   with the declared income denominator. Use the notebook consumption anchor
   and transfer size when testing notebook parity.
2. **Conditional asset response:** hold income/Yp/gamma/references fixed, update
   the relevant balance sheet and alpha, and apply the explicit ratio bounds.
3. **Runtime date transition:** follow the input/timing table: append current
   expected income before the first target, replace it after labour for the
   second; recompute b and maps, preserve opening stocks and update learned
   posterior state from completed income on the existing schedule. This is
   distinct from either conditional experiment; no lifetime-fixed gamma.

**Winsorization must be part of the response calculation.** Re-evaluate the
clipping operation after each perturbation; do not reuse an unclipped derivative
when a ratio is capped. For each ratio test interior, below/above bounds, exact
bounds and cap-entry/exit cases, including mixed positive IFA/housing and negative
NLA. Use one-sided checks at kinks. The original common-scaling income-elasticity
formula and its nonnegativity argument cannot simply be assumed for every capped
state. A sign difference must be decomposed, not hidden by an extra MPC clip.

**Target-integration acceptance gate (increment 2):**

The adapter-specific gate is defined in the increment list below. These response
and live-target checks follow adapter review, before experiment registration.

- The input/timing table and conditioning decisions are explicit.
- The agreed smoothed-base contract is exercised through both planning passes,
  including startup, positive/zero/negative income, binding/nonbinding floors,
  nonfinite-income rejection and missing/misaligned history errors. Check level,
  wealth ratios, borrowing-rate weights and PI rebasing against one common base.
  Test current-to-smoothed transitions and period-matched floors/deflators.
- Matched uncapped states reproduce notebook local and finite responses at
  rtol1e-5/atol1e-7; numerical steps are declared and refined near thresholds
  without silently changing tolerances or economic assumptions.
- Frozen-sample runtime response replay reproduces the calibration's aggregate
  .14 restriction using the original 9679 households, weights, observed-consumption
  anchor and annual-income LFA injection, fixed pre-shock gamma and updating alpha.
  Pin the reference artifact, perturbation and numerical tolerance. This verifies
  translation only. In live simulation .14 is NOT enforced: report the endogenous
  response without coefficient adjustment or endpoint retuning.
- Capped-state response operators agree with direct perturbations of the actual
  bounded rule; discrepancies from the uncapped notebook are identified.
- Formula, raw-target and dynamic-budget MPCs are separately named. The current
  VAT-exclusive formula-MPC diagnostic is not relabelled as notebook total MPC.
- Monthly/quarterly/annual ratios and levels scale consistently; a matched
  annual-income injection is not confused with a period-income injection.
- Existing household/configuration regressions pass. No claim of macro
  equivalence and no auto-advance to the next step.

### Reviewed implementation increments

The canonical floor-unit prerequisite is completed and committed as b1db808e;
it is not a pending increment. The remaining order is:

1. **Income/state adapter — implemented locally and technically reviewed.**
   Detailed scope, interfaces, lifetime and tests: [INCREMENT-1.md](INCREMENT-1.md).
   Construct an inspectable input object and diagnostics from aligned behavioural
   income B, historical nominal floors S, selected CPI/C_0 histories, the M-based
   PI rebase and copied opening-stock/principal exposures. Preserve reset and
   both-pass lifetimes. Deliver the adapter, source/caller mapping and tests.
   **Acceptance:** independent arithmetic checks for b and PI log-level
   preservation; matched history/floor/price dates; nonunit C_0; startup and
   invalid/missing input cases; distinct t-2/t-1/t stock fixtures; revised
   second-pass income; unchanged opening arrays despite intervening mutations;
   debt coverage, currency-scale and reset checks. Existing relevant regressions
   pass. Adapter construction does not itself establish active target parity;
   activation of its inputs in all target/rate terms belongs to increment 2.
2. **Target/dynamic integration — detailed plan in [INCREMENT-2.md](INCREMENT-2.md).**
   Implemented locally, technically reviewed and user-approved on 5 November 2026. Uses the approved
   adapter outputs into the existing CreditAugmentedConsumption; use b in the
   level and a*b in wealth/rate/cashflow ratios, with a=12/time_unit. Preserve the
   agreed map/residual/ECM/extra-channel/guard and accounting policies. Deliver
   active caller/evaluator wiring and separately named conditional probe operators.
   **Acceptance:** the response gate above, residual/channel decomposition in
   section 4 and equation/accounting checks in section 5 pass. Reproduce the
   frozen-sample calibration-only .14 operator, never enforce it in simulation.
   Run relevant household/configuration regressions before review.
3. **Paired validation review — detailed contract in [INCREMENT-3.md](INCREMENT-3.md).**
   Smoke, six-case campaign, approved diagnostic recapture and recoverable reporting
   are complete. Pin/cohort/path/primary-finiteness/booking checks pass. Both native
   target endpoint replays pass; branch independent guard checks pass. Main's
   separate guard split fails numerically; its guarded alternative is a distinct
   verified method. Branch seed-13 LFA reversal is mechanically explained, with
   exact funding mediation still partial. Full Increment 3 acceptance remains
   inconclusive. The dated evidence above supersedes older missing-main-operands
   statements in INCREMENT-3.md; historical run records remain unchanged.

Review each increment before advancing. The original source tracing and final
consistency review were preparation; subsequent implementation and campaign
results are recorded above. The once-drawn-residual sensitivity remains a separate
queued initialization diagnostic; the review is documented with an inconclusive
verdict, not silently accepted. Prepare its own design before authorization or execution.
Refresh GitNexus impact analysis before
symbol edits and run change analysis before any directed commit. For monthly/
annual unit fixtures use a=12/time_unit; this is a units check, not a decision to
re-estimate quarterly dynamics or change the four-quarter runtime window.

## 4. Retained residual and dynamic policies: integration checks

Deliver the exact initialization equation and a decomposition of raw target,
residual, real-rate term, ECM gap, cashflow/uncertainty additions and growth guard.
User decision, 4 November 2026: retain initial_state residual anchoring for this
runtime integration. Compute epsilon once from the unshocked initial budget and
deterministic target under the agreed income contract, then hold it fixed per
household. Treat this as an initialization convention, not evidence that all
cross-sectional residual variation is a persistent household characteristic.
Do not infer persistence from sigma or add a residual draw or sigma-squared/2
mean correction to this backed-out residual. Do not reopen the settled
purchaser-price/VAT basis to retune ECM.

User decision, 4 November 2026: retain both existing short-run additions outside
the core ECM: consumer-debt repricing cashflow coefficient -.003 and uncertainty
change coefficient -.005. Retain ECM speed .56 and the fitted income-growth
coefficient zero. These additions are existing runtime policies, not coefficients
estimated by the retained static calibration.

```
cashflow_index = delta_nominal_consumer_debt_rate * D_consumer_ledger_real_open / (4*b)
delta_log_C_desired = .56 * (log(C_star) - log(C_desired_previous))
                      - .003 * cashflow_index - .005 * delta_uncertainty
```

Use the agreed annualized smoothed-income denominator in the cashflow index.
Preserve existing rate-change units, contractual repricing definition, debt-date
contract and uncertainty-change timing (including first-period treatment).
Keep the borrowing-rate level term -.72*r inside C_star distinct from this
short-run nominal-rate-change term. Report ECM, cashflow and uncertainty
contributions separately and test unchanged channel behavior apart from the
explicitly approved income-denominator adaptation. Existing growth/numerical
guards still apply; the equation above is before those guards. The denominator
adaptation is implemented under Increment 2. Its paired campaign and target follow-ups have run. Main
separate-guard numerical verification and overall economic interpretation remain
open under Increment 3; the dated ledger distinguishes passes from limitations.

User decision, 4 November 2026: retain the existing quarterly desired-real-
consumption growth guard, clipping the combined ECM/cashflow/uncertainty log
change to [-.5, .5]. Report its binding frequency in baseline and shock runs;
it is a guard on desired consumption growth, not a bound on realised spending.
Keep its diagnostics separate from wealth-ratio caps and logarithmic-tail entry.

The once-drawn fixed-residual sensitivity was separately registered and explicitly
approved, then completed on6 November2026: two-policy smoke plus3 A/9 D full cases
across model seeds12–14 and draw seeds12001–12003. All runtime and saved-data
engineering gates pass. The completed design, lifecycle, measurement ledger and
results are in [RESIDUAL-SENSITIVITY.md](RESIDUAL-SENSITIVITY.md); vault evidence is
outputs/implementation/2026-11-06-residual-full/report.md. The comparison retains
coefficients/floors/maps/dynamics, controlled preconstruction RNG/state and native
funding inputs; no quarterly innovations, coefficient refit or mean correction.

Once-drawn q20 mean goods consumption is21.69% lower and support higher; longer-
horizon response signs remain seed-sensitive. Retain initial_state. The residual
diagnostic is complete; this does not erase historical Increment3 limitations or
establish empirical macro-response acceptance. Final implementation acceptance,
with explicit limitation dispositions, is recorded in FINAL-REVIEW.md.

**Integration acceptance gate:** residual mean/dispersion and the effect of
changing the intercept/sigma are shown under the chosen policy. Baseline/shock evaluations
reuse the intended residual/state. Pure target changes are distinguished from
new dynamic-policy changes. Existing policies remain until explicitly revised.

The retained epsilon is not a random draw or quarterly innovation; fitted sigma
does not impose its cross-sectional dispersion. A later initialization comparison
may inform a revision; it is not a prerequisite for adapter/integration work.

## 5. Accounting checks and registered paired validation

The approved seed-12 smoke and six-case full campaign have run on pinned main
`a3d48e5c` and branch revisions with the common cohort and frozen interventions.
The review confirms complete paired paths, finite primary outcomes, one-time
transfer/LFA booking and payer/bank/household account links within registered
tolerances. The campaign report records measured row-20 GDP, response paths, resource use
and flags. Dated follow-ups now provide both revisions' conditional native target
replay, independent branch guard checks, branch funding-floor tracing and the
completed recoverable reporting package. Main separate-guard checks still fail
numerically; missing native funding operands prevent complete mediation. Keep
these limitations distinct from successful intervention booking/account links.

The six-case campaign and diagnostic recapture remain frozen. Preserve raw target,
desired budget, requested goods, realized outcomes and financing as separate
objects. Actual requested credit is available under its native household time
series; no absent carrier field or financed-consumption series is inferred.
Report guard/cap/tail frequencies and transitions; do not discard rows or reject
conditional stock responses by sign alone. Resolving remaining limitations or
conducting a separately authorized residual sensitivity does not change the
registered campaign's historical results or imply production acceptance.

### Agreed purpose and initialization protocol — 4 November 2026

The transfer/stock perturbation is a deliberately controlled validation input,
not a new behavioural shock term or a calibration run. Trace changed input -> raw
long-run target -> desired budget -> realised spending, separating ECM delay,
financing/rationing effects and wiring/initialization errors. Specify the shock's
size, units, date, affected accounts and accounting counterpart before running;
the approved campaign described above subsequently implemented this protocol.

For each code revision, initialize an unshocked state first, including any
initial-state residual calculation, then clone into control and shock runs.
Preserve starting residuals, income history, beliefs, lagged consumption budget
and random-generator state as intended. Apply the declared perturbation only to
the shock copy and specify timing relative to both planning passes. Do not back
out a new residual after applying the shock. With another chosen residual policy,
use its agreed paired-state/randomness protocol rather than silently fixing iid
innovations forever.

Subsequent states may evolve differently according to the model. Compare
shock-minus-control within each revision before comparing those responses across
revisions. Each revision has its own unshocked initialization; cross-revision
residual equality is not required. Initial level comparisons are separate.
No .14 runtime target or automatic recalibration is introduced by this experiment.

**Acceptance gate:** accounting identities and numerical tests pass; substantive
macro criteria are specified before runs and assessed honestly. Static target
MPC is not required to equal first-quarter realised MPC. No automatic refit.

## 6. Final review and merge decision

**Completed6 November2026: implementation ACCEPTED with recorded limitations.**
[FINAL-REVIEW.md](FINAL-REVIEW.md) reviews the full effective configuration/diff
against a3d48e5c, includes dirty/untracked implementation, and records1106 fresh
passing regressions/35 frozen replay checks plus all candidate engineering evidence.
No blocking correctness findings. Retain initial_state and current settings;
macroeconomic robustness, historical comparison and calibration/provenance limits
remain qualified. No runtime/YAML edit, commit, push or merge in this review.
The text below defines the review contract and separate merge authority.

This is the later production/merge review, not the completed preparation review.
Review the complete effective credit_augmented YAML and code diff against pinned
main a3d48e5c, including live input definitions, all active guards and initial-state policies.
Report exactly what changed, unchanged mechanisms, reproduction commands,
failures and limitations. Keep the previous main commit as the rollback reference;
no extra active parameter block is needed. Commit/push/merge only as directed.

The aggregate .14 condition concerns the calibration sample and its declared
observed-consumption-weighted annual-income LFA injection. It is not a quarterly
endpoint-adjustment algorithm. Four endpoint-profile searches remain unresolved;
that caveat and absent sampling intervals remain explicit. They do not authorize
a bound change or block every implementation check by default.


### Consolidated status — 6 November 2026

- Floor repair: committed locally; Increment 1 adapter: implemented and technically
  reviewed; Increment 2 integration: implemented, reviewed and user-approved.
  Current re-entry branch HEAD is 96c897e5 with implementation/docs still uncommitted.
- Increment 3 execution, conditional target replays and reporting are complete.
  Branch independent guard verification passes; main's separate guard split retains
  123 failed checks. Exact branch funding mediation and economic acceptance remain
  incomplete. Neither a reporting completion nor this plan update grants approval.
- Original campaign specification SHA256 remains
  bd6196283e740085285f5518da4d3b51ff707ffd7ed243fb64992a6d9e973676;
  amendment, execution approval and diagnostic recapture are recorded separately.
- Residual sensitivity under section 4 is now implemented, separately registered,
  explicitly user-approved and fully executed/reported: 3 initial_state plus 9
  crossed fixed_effect cases, 528 quarters/1056 target passes; all engineering and
  independent saved-data checks pass. See RESIDUAL-SENSITIVITY.md and vault
  outputs/implementation/2026-11-06-residual-full/report.md. Once-drawn Q20 goods
  consumption is 21.69% lower on the crossed-grid descriptive mean, support is
  higher, and longer-horizon response sign sensitivity remains. Recommendation:
  retain initial_state; the live default is unchanged. The residual diagnostic is
  complete, not a grant of economic acceptance or permission for another run/fit.
- Final full-code/config review is complete: implementation accepted with explicit
  limitation dispositions in FINAL-REVIEW.md. No further implementation work is
  required by that review. Commit/push/merge only on separate user direction.
