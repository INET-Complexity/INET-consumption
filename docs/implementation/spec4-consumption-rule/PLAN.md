# Implement the retained consumption rule in INET

Branch: `feature/spec4-consumption-rule`, created from main `a3d48e5c`.
Final preparation/consistency review: 4 November 2026, against INET b1db808e.
This review changes documentation only; the adapter has not been implemented.

Update the existing `CreditAugmentedConsumption` and
`run_model/config/consumption_paper_parameters.yaml:desired_consumption.credit_augmented`
directly. The earlier separate calibration-block/alternate-selector proposal is
superseded. Work remains on this branch, one reviewed step at a time. No automatic
refitting, push, merge or changes to policy assumptions.

## Current status and next implementation increment

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

**The income/state adapter is the first implementation increment afterward.**
Here, afterward means after this final plan consistency review. The floor-unit prerequisite is complete
and committed. The adapter is pending and is distinct from target integration.
See the deliverables and acceptance gates under section 3. Detailed planning
for this increment is in [INCREMENT-1.md](INCREMENT-1.md). Its three audit
clarifications are now confirmed and recorded: explicit current-price matching,
once-per-period capture with duplicate/stale rejection, and same-pass M/q payload.
Implementation remains pending.
PLAN remains the governing guide; expand later increments one at a time after
the preceding review. No adapter implementation is started by creating its plan.

Source tracing and the input/timing table are complete. No further core policy
discussion is required before adapter work; return to the user if implementation
reveals a substantive ambiguity. Experiment registration and the later residual
comparison remain deferred. This documentation review does not start the adapter.

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
the consumption stock inputs. This is an approved implementation requirement;
current positional prev wiring has not yet been changed.

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
The current runtime still uses current income for the level and rate weights,
positive-only history for the rolling denominator, and its supplied PI ratio.
Implementation and validation of this approved contract remain pending.

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

1. **Income/state adapter — first implementation increment after this review.**
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
2. **Target/dynamic integration — after adapter review.** Wire the approved
   adapter outputs into the existing CreditAugmentedConsumption; use b in the
   level and a*b in wealth/rate/cashflow ratios, with a=12/time_unit. Preserve the
   agreed map/residual/ECM/extra-channel/guard and accounting policies. Deliver
   active caller/evaluator wiring and separately named conditional probe operators.
   **Acceptance:** the response gate above, residual/channel decomposition in
   section 4 and equation/accounting checks in section 5 pass. Reproduce the
   frozen-sample calibration-only .14 operator, never enforce it in simulation.
   Run relevant household/configuration regressions before review.
3. **Register and run paired validation — after integration review.** Record
   transfer/stock sizes, affected households, units, accounts/counterparts, shock
   timing, seeds, horizon, resource budget, abort conditions and numerical/economic
   assessment criteria before any run. These details remain deliberately unset
   until equation and accounting checks pass. Deliver the registered specification,
   smoke results, then the agreed full-budget evidence.
   **Acceptance:** section 5's paired-state protocol, accounting and declared
   numerical/economic criteria; report caps/floors/growth-guard binding and
   target/desired/financed/realised paths with all failures disclosed.

Review each increment before advancing. Source tracing and this final consistency
review are preparation; they do not implement the adapter or start experiments.
The queued once-drawn-residual sensitivity is outside these increments and waits
until income/transmission validation. Refresh GitNexus impact analysis before
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
guards still apply; the equation above is before those guards. Implementation
of the denominator adaptation and its verification remain pending.

User decision, 4 November 2026: retain the existing quarterly desired-real-
consumption growth guard, clipping the combined ECM/cashflow/uncertainty log
change to [-.5, .5]. Report its binding frequency in baseline and shock runs;
it is a guard on desired consumption growth, not a bound on realised spending.
Keep its diagnostics separate from wealth-ratio caps and logarithmic-tail entry.

After validating the income contract and runtime transmission, compare the
retained initial_state residual policy with a once-drawn fixed residual,
epsilon_i ~ N(0, fitted_sigma^2), as a separate registered initialization sensitivity. Assess initial target gaps,
transition paths, consumption dispersion and financing effects with controlled
random seeds. This later comparison is queued, not authorized to run now; no
quarterly iid residual is selected. The static fit does not identify persistence.

**Integration acceptance gate:** residual mean/dispersion and the effect of
changing the intercept/sigma are shown under the chosen policy. Baseline/shock evaluations
reuse the intended residual/state. Pure target changes are distinguished from
new dynamic-policy changes. Existing policies remain until explicitly revised.

The retained epsilon is not a random draw or quarterly innovation; fitted sigma
does not impose its cross-sectional dispersion. A later initialization comparison
may inform a revision; it is not a prerequisite for adapter/integration work.

## 5. Accounting checks, then registered paired runs

First verify the existing total-consumption/rent/imputed-rent/VAT/credit identities
on the changed function, including initial and later periods and the two planning
passes. Main-residence wealth selection must not silently remove other-property
rental income or duplicate an existing housing carve-out.

Only after these equation/accounting checks pass, register a small paired
baseline/transfer/stock experiment: pinned main baseline `a3d48e5c` versus the
branch, identical inputs/seeds, stated shock,
horizon, budgets and predeclared plausibility criteria. Run a short smoke before
the agreed full budget. Compare raw target, desired budget, requested goods,
realised goods/total/cash spending, financing and subsequent paths separately.
Report guard-hit rates and each ratio's cap mass/response effects; no discarded
outliers or unreported failures. Conditional negative stock responses are not
rejected by sign alone.

### Agreed purpose and initialization protocol — 4 November 2026

The transfer/stock perturbation is a deliberately controlled validation input,
not a new behavioural shock term or a calibration run. Trace changed input -> raw
long-run target -> desired budget -> realised spending, separating ECM delay,
financing/rationing effects and wiring/initialization errors. Specify the shock's
size, units, date, affected accounts and accounting counterpart before running;
no such experiment has been run by this planning discussion.

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
