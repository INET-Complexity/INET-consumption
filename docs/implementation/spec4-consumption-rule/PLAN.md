# Implement the retained consumption rule in INET

Branch: `feature/spec4-consumption-rule`, created from main `a3d48e5c`.
Plan re-audited against the working code on 4 November 2026.

Update the existing `CreditAugmentedConsumption` and
`run_model/config/consumption_paper_parameters.yaml:desired_consumption.credit_augmented`
directly. The earlier separate calibration-block/alternate-selector proposal is
superseded. Work remains on this branch, one reviewed step at a time. No automatic
refitting, push, merge or changes to policy assumptions.

## Floor-unit increment completed — 4 November 2026

Canonical floor repair implemented locally in country.py: from_pickled_country
passes the population scale to _select_net_smic_base, which expands the external
INSEE monthly level once. net_smic_base is stored in model monetary units;
initialization and both planning-pass refreshes reuse it without another scale.
Already-expanded benefit fallbacks retain their existing proxy/frequency behavior.
No household debt, benefit-payment formula, consumption coefficient or YAML edit.

809 country/household/credit-market tests pass, including 15 new scale/frequency/
CPI/fallback/two-pass/reset regressions and the existing two-period production
household accounting integration. Ruff lint/format and whitespace pass. An
initialized production-config FRA check yields model monthly base5643500 and
quarterly floor8465250 per CU, or1693.05 in ordinary currency units, with218 initial
raw behavioural-income observations below the corrected threshold. No registered
macro experiment, fit, commit/push/merge. GitNexus refreshed/change scope reviewed.
Evidence: vault outputs/implementation/2026-11-04-floor-unit-repair/.

Review this increment before starting the income/state adapter. The agreed
smoothed-base, PI-rebase and opening-snapshot implementation remains pending.

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

Resolved-state floor check completed on 4 November: with the existing 2014 FRA
cache and active resolved configuration, scale=5000 and H=5760. The canonical
quarterly floor is 1693.05 per CU, unexpanded; income/assets are expanded. Its
consistent value in model monetary units is 8465250 per CU. This is a confirmed
unit mismatch, not a new subsistence policy. Evidence and reproduction:
`vault-starter-kit/knowledge-vault/outputs/implementation/2026-11-04-floor-units/`.

User approved this repair in the plan on 4 November 2026, after the independent
double-check. Precede adapter work with a separate reviewed canonical-floor unit
repair. This approval records planned scope; runtime implementation followed in the separately verified floor-unit
increment recorded above.
Apply population expansion exactly once at initialization and per-pass refresh;
preserve the .5 SMIC/CU/time-frequency policy. The shared floor also feeds the
monetary bridge, financing support and goods subsistence checks, so a local
consumption-only rescaling would leave inconsistent definitions. GitNexus reports
HIGH upstream impact (2 direct callers, 5 affected symbols at depth 3). Required
checks include scale=1/5000, both planning passes, initial income/PI, support/fiscal
identities and no duplicate scaling. Verify monthly/quarterly/annual conversion,
nonunit initial CPI, and reset behavior. Check other-country/fixture fallback
SMIC units explicitly so already-expanded fallback values are not scaled twice.
Keep the physical floor .5 * monthly_net_SMIC * months_per_period * CU unchanged;
expand its monetary representation once. Review the repair and its regression
results before beginning the income/state adapter. This repair is now implemented and verified as recorded above; adapter work
remains pending.

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

Trace actual append/replace behavior and document each source series/date before
changing accessors. Do not mechanically replace every prev with current. Capture
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
- Rebase the supplied permanent-income log ratio to b while preserving the
  intended permanent-income level. Verify that level's frequency, income concept
  and bridge to the supplied ratio; this decision does not re-estimate beliefs.
- Unemployment benefits should normally keep income positive. The floor is a
  fallback; persist/report its binding frequency rather than assume it never binds.

The intended quarterly raw target, before existing numerical guards, is:

```
W = gamma * NLA/(4*b) + .022 * IFA/(4*b) + .013 * HA/(4*b)
r = (real_rate_consumer * debt_consumer
     + real_rate_mortgage * debt_mortgage) / (4*b)
log(C_star) = log(b) + alpha0 - .72*r + alpha*p
              + F(W, 1-alpha) + epsilon
```

Apply the configured ratio winsorization before maps and W. Preserve the real
borrowing-rate term in the long-run target, including annualized real-rate units;
it is distinct from the short-run rate-change cashflow term. Debt dates still
require the input/timing trace. CM supports debt/income weighting; annual-income
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

**Acceptance gate:**

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

0. **Canonical floor units — implemented, review pending:** expand the external
   INSEE base once; preserve existing model-unit fallback conventions. 809 tests,
   initialized FRA check and Ruff pass. Review before adapter work.
1. **Income/state adapter:** implement the traced B/S/price history alignment,
   M-based PI rebase and copied opening-stock/principal exposures. Verify
   corrected subsistence units, initialization/reset and both passes.
   Deliver the explicit input object/diagnostics and meaningful timing/unit tests.
2. **Target/dynamic integration:** use b consistently in level, wealth and rate
   terms; preserve agreed map updates, residuals, ECM/extra channels, guards and
   accounting. Add conditional probe operators distinct from ordinary runtime
   updates; reproduce frozen-sample responses and calibration-only .14. Verify
   existing configurations and accounting regressions.
3. **Registered paired validation:** after reviewing increments 1 and 2, specify
   shocks/accounts/seeds/horizon/budget and numerical/economic criteria, smoke,
   then the agreed run. Compare within-revision responses before revisions.
   Report caps, floors, growth-guard binding and financing/realized paths.

Review each increment before advancing. Later once-drawn-residual sensitivity
remains deferred; no runtime implementation or simulation is started by the
source-tracing/documentation step. Refresh GitNexus impact analysis before symbol
edits and run the required change analysis before any directed commit.

## 4. Then review residuals and the dynamic wrapper

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
cashflow_index = delta_nominal_consumer_debt_rate * lagged_consumer_debt / (4*b)
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

After validating the income contract and runtime transmission, compare this
policy with a once-drawn fixed residual, epsilon_i ~ N(0, fitted_sigma^2), as a
separate registered initialization sensitivity. Assess initial target gaps,
transition paths, consumption dispersion and financing effects with controlled
random seeds. This later comparison is queued, not authorized to run now; no
quarterly iid residual is selected. The static fit does not identify persistence.

**Acceptance gate:** residual mean/dispersion and the effect of changing the
intercept/sigma are shown under the chosen policy. Baseline/shock evaluations
reuse the intended residual/state. Pure target changes are distinguished from
new dynamic-policy changes. Existing policies remain until explicitly revised.

The active initial_state policy computes epsilon once from the initial budget
and deterministic target, then holds it fixed for that household. It is not a
random draw or a quarterly innovation; fitted sigma does not impose its
cross-sectional dispersion. This is the retained policy for the present
integration; a later initialization comparison will inform any revision.

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
