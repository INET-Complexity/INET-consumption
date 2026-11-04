# Implement the retained consumption rule in INET

Branch: `feature/spec4-consumption-rule`, created from main `a3d48e5c`.
Plan re-audited against the working code on 4 November 2026.

Update the existing `CreditAugmentedConsumption` and
`run_model/config/consumption_paper_parameters.yaml:desired_consumption.credit_augmented`
directly. The earlier separate calibration-block/alternate-selector proposal is
superseded. Work remains on this branch, one reviewed step at a time. No automatic
refitting, push, merge or changes to policy assumptions.

## 1. Implemented and verified

The active block supplies the retained notebook fit:9679 French households,
notebook commit4783467, result results-a8932a0a7eeb.

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

**Evidence already obtained:**529 household/configuration tests and Ruff pass.
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

## 2. Existing runtime choices still in force

These settings/mechanisms have not been estimated by the static fit:

| Mechanism | Current behavior | What must be established |
|---|---|---|
| Income ratios | Annualized four-quarter geometric runtime income | Match economic definitions and distinguish ordinary transitions from conditional experiments |
| Runtime formula-MPC probe | Reuses the baseline ratio denominator AND supplied PI log ratio in the perturbed evaluation | A notebook fixed-Yp experiment instead changes p=log(Yp/Y) when Y changes; define and label both response objects |
| Coefficient timing | `_evaluate_target` recomputes alpha/gamma from the ratios it receives | Specify which maps are held fixed inside probes and which update between simulation dates |
| Financial/housing stock timing | Caller passes `prev` stocks; planning-pass comments identify financial wealth terms as t-2 | Trace actual series positions; do not assume that every `prev` is the desired economic t-1 |
| Residuals | `idiosyncratic_persistence: initial_state` backs out epsilon from the initial budget and deterministic target | The configured sigma does NOT enforce the initial residual dispersion; the intercept is offset at the initial anchor |
| Dynamics | ECM speed .56, real-rate coefficient -.72, cashflow -.003, uncertainty -.005, growth bound ±.5 log units | Keep these additions distinct from the calibrated static target; decide explicitly whether they remain |
| Accounting | Gross total target, existing real-budget state, housing carve-outs, goods VAT conversion, financing/settlement | Verify unchanged identities with the newly wired target |

HFCS `disp_geom_avg_income` is a same-year component blend, not evidence for a
four-quarter temporal smoothing window. The existing rolling denominator is
not a lifetime-fixed initial anchor. Initial-state residuals are not iid draws
from the fitted normal distribution. A changed YAML sigma/intercept alone
therefore does not establish changed initial consumption levels/dispersion.

Source anchors: `consumption.py` `_evaluate_target`, `compute_target_consumption`,
`_initial_state_epsilon` and `_geometric_average_income`; `households.py`
`compute_target_consumption`. Refresh impact analysis before changing symbols.

## 3. Next step — agree and test the income/state contract

Deliver a short input/timing table before changing code. For Y, Yp or its supplied
log ratio, LFA, debt, IFA and main-residence housing, state the account series,
economic date, deflator, population scaling, flow frequency and ratio denominator.
Trace both planning passes, including append/replace behavior. Confirm NLA equals
LFA minus the intended debt components before regressor clipping.

Specify the following experiments separately:

1. **Notebook conditional income response:** hold Yp, alpha, gamma, stocks and
   training references fixed. Thus p changes by -log(1+T/Y); stock ratios change
   with the declared income denominator. Use the notebook consumption anchor
   and transfer size when testing notebook parity.
2. **Conditional asset response:** hold income/Yp/gamma/references fixed, update
   the relevant balance sheet and alpha, and apply the explicit ratio bounds.
3. **Runtime date transition:** choose when an income innovation enters income
   history, the learned PI signal, ratio denominators and map updates. This is
   not automatically either conditional experiment. Do not infer lifetime-fixed
   gamma or remove existing rolling income without a decision.

**Winsorization must be part of the response calculation.** Re-evaluate the
clipping operation after each perturbation; do not reuse an unclipped derivative
when a ratio is capped. For each ratio test interior, below/above bounds, exact
bounds and cap-entry/exit cases, including mixed positive IFA/housing and negative
NLA. Use one-sided checks at kinks. The original common-scaling income-elasticity
formula and its nonnegativity argument cannot simply be assumed for every capped
state. A sign difference must be decomposed, not hidden by an extra MPC clip.

**Acceptance gate:**

- The input/timing table and conditioning decisions are explicit.
- Matched uncapped states reproduce notebook local and finite responses at
  rtol1e-5/atol1e-7; numerical steps are declared and refined near thresholds
  without silently changing tolerances or economic assumptions.
- Capped-state response operators agree with direct perturbations of the actual
  bounded rule; discrepancies from the uncapped notebook are identified.
- Formula, raw-target and dynamic-budget MPCs are separately named. The current
  VAT-exclusive formula-MPC diagnostic is not relabelled as notebook total MPC.
- Monthly/quarterly/annual ratios and levels scale consistently; a matched
  annual-income injection is not confused with a period-income injection.
- Existing household/configuration regressions pass. No claim of macro
  equivalence and no auto-advance to the next step.

## 4. Then review residuals and the dynamic wrapper

Deliver the exact initialization equation and a decomposition of raw target,
residual, real-rate term, ECM gap, cashflow/uncertainty additions and growth guard.
Compare the current initial-state anchoring with the intended treatment of the
fitted residual distribution before choosing a policy. Do not infer persistence
from sigma or accidentally add both a residual draw and a sigma-squared/2 mean
correction. Do not reopen the settled purchaser-price/VAT basis to retune ECM.

**Acceptance gate:** residual mean/dispersion and the effect of changing the
intercept/sigma are shown under the chosen policy. Baseline/shock evaluations
reuse the intended residual/state. Pure target changes are distinguished from
new dynamic-policy changes. Existing policies remain until explicitly revised.

## 5. Accounting checks and registered paired runs

First verify the existing total-consumption/rent/imputed-rent/VAT/credit identities
on the changed function, including initial and later periods and the two planning
passes. Main-residence wealth selection must not silently remove other-property
rental income or duplicate an existing housing carve-out.

Then register a small paired baseline/transfer/stock experiment: pinned main
baseline `a3d48e5c` versus the branch, identical inputs/seeds, stated shock,
horizon, budgets and predeclared plausibility criteria. Run a short smoke before
the agreed full budget. Compare raw target, desired budget, requested goods,
realised goods/total/cash spending, financing and subsequent paths separately.
Report guard-hit rates and each ratio's cap mass/response effects; no discarded
outliers or unreported failures. Conditional negative stock responses are not
rejected by sign alone.

**Acceptance gate:** accounting identities and numerical tests pass; substantive
macro criteria are specified before runs and assessed honestly. Static target
MPC is not required to equal first-quarter realised MPC. No automatic refit.

## 6. Final review and merge decision

Review the complete effective credit_augmented YAML and code diff against main,
including live input definitions, all active guards and initial-state policies.
Report exactly what changed, unchanged mechanisms, reproduction commands,
failures and limitations. Keep the previous main commit as the rollback reference;
no extra active parameter block is needed. Commit/push/merge only as directed.

The aggregate .14 condition concerns the calibration sample and its declared
observed-consumption-weighted annual-income LFA injection. It is not a quarterly
endpoint-adjustment algorithm. Four endpoint-profile searches remain unresolved;
that caveat and absent sampling intervals remain explicit. They do not authorize
a bound change or block every implementation check by default.
