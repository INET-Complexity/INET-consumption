# Increment 3 — registered paired transmission validation

Current implementation: tasks 1–17 are recorded. The approved six-case campaign
completed for main/branch seeds 12–14 with control, transfer and LFA arms over
eight post-shock quarters. All case artifact hashes, pins, cohort/path, finite
primary-outcome, and booking/account-link checks passed. The frozen branch
evaluator reproduces all 144 target passes at saved precision, and all 96
registered 10-block treatment/control paths telescope with zero saved-precision
residual. Main captures lack eight native operands; the main target gate remains
unavailable. Branch seed dispersion is traced through native target blocks, but
main's large late-horizon target-budget differences and a branch LFA
target-to-realized sign reversal leave economic acceptance inconclusive. The exact
campaign approval is recorded at
`knowledge-vault/outputs/implementation/2026-11-05-increment-3-campaign-environment-amendment/campaign-approval.json`.
Campaign and follow-up reviews:
`knowledge-vault/outputs/implementation/2026-11-05-increment-3-campaign-review/report.md`;
`knowledge-vault/outputs/implementation/2026-11-05-increment-3-target-review/report.md`.
Do not refit, infer missing fields, or promote from this run alone.

Status: implementation authorized on 5 November 2026, one independently reviewable
task at a time. Increment 2 is user-approved. The user accepts exogenous external
funding through existing receipt/LFA links; no new fund behaviour is required.
Task 1 source/account mapping is complete in [INCREMENT-3-ACCOUNTS.md](INCREMENT-3-ACCOUNTS.md).
Smoke registration/execution and full-campaign specification registration are
complete. The amended campaign hashes have explicit user approval; the
controller-compatible record is saved at
`knowledge-vault/outputs/implementation/2026-11-05-increment-3-campaign-environment-amendment/campaign-approval.json`.
The six approved cases have run. The branch numerical target review is
complete; main target reconstruction and overall economic acceptance remain
unavailable/inconclusive as summarized in Tasks 16–17 below.

## Authority and objective

[PLAN.md](PLAN.md) governs this increment. [INPUT-TIMING.md](INPUT-TIMING.md) and
[INCREMENT-2.md](INCREMENT-2.md) define the integrated timing and target contract.
Follow the vault wiki/processes/experiment-workflow.md: spec -> smoke -> run ->
report. Register the final design in the vault's wiki/experiments/specs/ before
simulation. Keep household records outside the vault, under /private/tmp.

Question: does a declared household income/stock intervention propagate through
inputs, raw target, desired budget, financing and realised spending as the
implemented equations/accounting imply? Compare within each revision first,
then compare responses between pinned main a3d48e5c and the approved branch.
This is transmission validation, not estimation or evidence of empirical MPC fit.

Do not change calibration, ratio caps, floors, training references, PI learning,
residual policy, ECM/channels, credit policy or VAT conventions. Do not enforce
.14 in live states. Once-drawn residual sensitivity remains a separate later task.
No automatic commit, push, PR, production promotion or merge.

## Frozen comparison and proposed run matrix

| Item | Proposed specification |
|---|---|
| Revisions | Main a3d48e5c; exact approved branch source/config snapshot based on cef2bff0, including uncommitted Increment 1/2 files. Pin SHA256 manifest and diff, not just HEAD. |
| Country/data | FRA, quarterly, 2014 single-wave production input. Reuse one verified real-data cache/sample in both revisions; hash cache, source metadata, ordering, population scale and initial accounts. No fixture substitution, regeneration or new HFCS wave. |
| Config | Each revision's native resolved FRA configuration, frozen in isolated experiment directories; report all differences. This compares the complete implementation against pinned main, not the isolated causal effect of Increment 2. |
| Full seeds | 12, 13, 14; serial, one worker, BLAS threads 1. No seed substitution. |
| Timing | 20 unshocked iterations, then intervene before iteration 20's first planning pass; initial HDF5 row 0, pre-shock row 20, impact row 21. |
| Horizon | Eight response quarters: iterations 20..27, realised rows 21..28; t_max=28. Report impact, four-quarter and eight-quarter cumulative responses. |
| Arms per revision/seed | Unshocked control; one-time flow transfer; one-time LFA endowment. A zero-intervention clone is an additional smoke/determinism control. |
| Full budget | Six unshocked warm-ups and 18 eight-quarter continuations, at most 264 simulated iteration-equivalents. Shared controls are reused only within the same revision/seed. |
| Smoke | Seed 12, both revisions; same 20-quarter warm-up and intervention timing, two-quarter continuation for control/zero/transfer/LFA. This checks operations, not eight-quarter economic acceptance. |
| Host/resources | Local machine and existing INET venv; one process at a time; 16 GiB process RSS ceiling, 20 GiB output ceiling. Smoke wall cap 45 minutes; full campaign cap 6 hours, including smoke. |

The pre-run budget projection used the seed-12 smoke. The approved campaign
then completed in 222.2 active seconds; combined smoke-plus-campaign output was
3,990,007,799 bytes. The run stayed within the registered six-hour, 20-GiB output
and 16-GiB RSS ceilings; no controller budget abort occurred. These measured
figures supersede the pre-run projection. The registered matrix and horizon were
not shortened, and no seeds were added or replaced after inspecting responses.

### Common shock amounts and affected households

Freeze one common reference vector from the shared initial cache: quarterly
nominal cash income before revision-specific target initialization. Let m be its
unweighted median over positive finite observations. Record the exact source
field, m, H, population scale and household ordering in registration. The same
sample and amounts apply across revisions and seeds; failure to obtain that
common source is a preflight blocker, not permission to switch definitions.

- Flow arm: T_i = .01*m, for every initial household; one quarterly cash receipt.
- Stock arm: A_i = .01*(4*m), for every initial household; an opening nominal LFA
  addition. Annual-income scaling is explicit: this amount is four times T_i.
- Control: zero intervention. No selective exclusion by response, employment,
  income sign, debt or cap status. Initial households stay in the primary cohort.
- Nominal amounts carry the model population expansion once. Report per-model-
  household and descaled ordinary-currency amounts; never multiply by survey
  weights again when booking transactions.

These are proposed finite validation shocks, not the notebook's conditional
income probe or its e*annual-income .14 stock operator. Changing the size after
registration requires a separately labelled specification amendment.

## Accounts, counterpart and timing

Confirmed funding convention (user decision, 5 November 2026): assume an exogenous
external fund for the controlled experiment and reuse existing household receipt
and bank-household LFA links. Record its outflow; do not build a new institution.
Report the aggregate injection relative to baseline quarterly GDP and household
cash income. Large interventions need a separate endogenous financing design;
no numerical size threshold has been selected or inferred.

The exact source/account map is in [INCREMENT-3-ACCOUNTS.md](INCREMENT-3-ACCOUNTS.md).
Use the existing staged external-income receipt for the flow arm, with one normal
cash settlement. Use an opening LFA addition for the stock arm. Preserve native
bank aggregation and its signed balancing deposit position; do not add a second
reserve asset or count the receipt twice. Government fiscal formulas remain
unchanged; endogenous support/tax/credit responses remain active.

Accounting implementation tests must verify the mapped receipt, intervention
snapshot and banking recurrences. The source map resolves the funding-design
question; it does not claim the full harness has been implemented or tested.

Flow staging must be visible in B and M exactly once per pass and once in settled
cash receipts; it must not enter again via a second social-transfer edit. Preserve
raw source components and report floor/support crowd-out separately. If endogenous
support falls after the transfer, show gross T, support change and net disposable
resource increment rather than redefining the shock denominator.

The LFA addition occurs after the unshocked warm-up and before opening-stock
capture. It must not be recaptured between passes. Do not apply a new residual
back-out after either intervention. Do not treat a stock injection as current
income or stage it through stage_household_income_shock.

## Paired state and randomness protocol

1. For each revision/seed initialize and run its own unshocked 20-quarter warm-up.
   Verify residual initialization completed without any intervention. Save the
   pre-shock household states, histories, priors/posteriors, lag budgets, adapter
   state, bank/market state and all RNG states.
2. Clone that complete simulation checkpoint into all arms. Country deepcopy is
   tested in Increment 2; whole Simulation cloning and the old-main copy seam still
   need verification. A harness compatibility shim may copy containers while
   preserving aliases; it must not backport the new consumption rule to main.
3. Use independent processes or restore process-global NumPy/Python RNG state at
   each arm's start, in addition to object-owned generator state. Country copying
   does not create independent global random streams. Include other used RNGs in
   the checkpoint inventory; never claim matching seeds alone proves pairing.
4. Before intervention, require equality of household IDs/order, all balance
   sheets, beliefs, histories, residuals, lag budgets and RNG states within each
   revision. Do not require cross-revision residual equality.
5. Run the zero arm and control with identical state/RNG restoration. They must
   agree before interpreting nonzero-shock paths. Then apply only the stated
   perturbation to each intervention arm and run the ordinary model sequence.
6. Subsequent prices, rates, jobs and random call paths may diverge endogenously.
   Shared initial RNG state is not a claim that every later draw is matched after
   control flow diverges. Record this limit and expose macro input differences.

No baseline macro-path replay or strict labour-state filter is imposed on the
primary response: it is full endogenous transmission. Optional stable-labour
summaries are descriptive sensitivities with counts, never replacement results.

## Response measures and predeclared analysis

For X measured in nominal currency, save path-specific real values X^r=X/(C/C0),
using the same selected consumption-price definition as the active rule. Report
nominal differences too. Do not deflate both paths by the control's CPI.

For a one-time nominal intervention Z_i, primary real impact response is
sum_i[X^r_shock(i,21)-X^r_control(i,21)] / sum_i[Z_i/d_control(20)].
Cumulative responses sum that numerator over rows 21..24 and 21..28 with the same
one-time denominator. Report the convention explicitly: this measures real
spending per pre-shock real unit received, not a derivative holding prices fixed.
Stock responses are spending per unit of added LFA, not income MPCs. Report the
small-shock log-target elasticity separately; do not label it the .14 operator.

Primary aggregates sum all model households once. Also show household response
median, p10/p90, negative share and deterministic pre-shock income/LFA/debt/tenure
bins. Freeze group cutpoints using the common initial input, report group counts
and amounts; do not let shock outcomes redefine groups. Empty groups are NA.
Monte Carlo summaries: each seed, arithmetic seed mean, SD, min/max; three seeds
are a small diagnostic sample, not survey sampling intervals or causal inference.

Save and analyse both target passes and each later quarter:

- B/S/M/original q/rebased p/b/annual_b, opening ledger debts and principal,
  alpha/gamma1, raw/used ratios, each cap flag, tail flag and residual.
- Raw log/level target; rate-level term; ECM gap; repricing and uncertainty
  additions; desired budget before/after the growth guard.
- Gross total target, rent/imputed rent carve-outs, VAT-exclusive goods demand,
  requested/granted credit, liquid drawdown/liquidation, funded budget and realised
  goods/total/cash consumption. Use actual native fields; do not invent a financed
  consumption series where only resource envelopes exist.
- Support, transfers and debt/deposit changes; account residuals and payer ledger;
  CPI, policy/contractual rates, employment, output and household income paths.

Produce predeclared tables for impact/4q/8q responses and baseline/arm levels,
channel/accounting reconciliation, guard/floor/cap/tail frequencies and entry/exit
shares. Plot control/shock levels and differences, cumulative spending responses,
and the channel contributions. Distinguish four-quarter rolling-income persistence
from ECM adjustment and subsequent endogenous feedback. Use equation reconstruction
and explicitly labelled sequential counterfactual decomposition; nonlinear channel
contributions depend on decomposition order, which must be fixed in registration.
Old main may lack diagnostics: use read-only observers on its native operands,
record unavailable fields as NA, and never substitute branch equations for main.

## Numerical and economic assessment gates

Register the exact tolerances before execution. Proposed gates:

| Criterion | PASS | FAIL / INCONCLUSIVE |
|---|---|---|
| Inputs/provenance | Same verified real cache, order/H and shock vector; complete revision/config/environment manifest. | Missing/stale cache, mock data, mixed scale or missing manifest: abort. |
| Pairing/zero control | RNG/identity metadata exact; numeric states and zero paths within rtol=1e-10, atol=1e-12 in descaled ordinary currency units. | Any unexplained divergence: fail; do not run the full campaign. |
| Target/dynamics | Independent reconstruction at rtol=1e-10/atol=1e-12 for logs/ratios; monetary tolerance atol=1e-7 ordinary currency, rtol=1e-10. | Unexplained discrepancies: fail, with household/period/pass diagnostics. |
| Booking/accounting | Exactly one intervention; gross credit equals payer debit; deposit and settlement counterpart identities; existing household/credit/VAT identities at their stricter existing tolerances. | Double booking, unrepresented counterpart or identity breach: fail. |
| Complete paths | All 3 seeds, 2 revisions, 3 arms, 8 quarters; finite required metrics and complete cohort. | Missing/failed arm or silently dropped rows: inconclusive campaign, retain all failures. |
| Economic transmission | Any response sign/magnitude is reconciled to inputs, nonlinear target, dynamics and funding/market outcomes; no unexplained residual above numeric tolerances. | Unexplained channel or accounting residual: fail. Mechanically correct but economically unclear response: inconclusive, not automatic promotion. |

Proposed economic review flags, to be assessed without coefficient retuning:
negative aggregate transfer impact/cumulative spending, transfer spending response
above 1, large disagreement in signs across seeds, baseline growth-guard mass >1%,
or intervention-minus-control growth-guard mass >1 percentage point. These are
review triggers, not empirical rejection thresholds. Show sign/magnitude and
financing/price/support explanations; do not clip MPCs or discard affected rows.
Negative conditional stock responses alone are not a failure. Caps/floors/tails
can legitimately bind; report them, never suppress them to pass a gate.

There is no predeclared empirical target for full-model MPCs, no requirement that
raw/desired/realised responses coincide and no required improvement over main.
The falsifiable engineering claim is exact declared input -> equation -> dynamic
budget -> accounted spending transmission within the stated numeric tolerances.
Economic adequacy and production acceptance require the final review in PLAN §6.

## Work sequence and deliverables

1. Funding assumption confirmed and source/account mapping completed (task 1).
   Checkpoint/clone primitive and targeted independence/RNG tests completed
   (task 2), awaiting review. Subsequent tasks separately cover
   intervention booking, observers and measurement. Keep meaningful lifecycle,
   clone, zero-shock, one-time booking and measurement tests. Reuse production
   APIs; impact analysis precedes existing-symbol edits. Source changes outside
   the harness require a concrete scope finding, not an unnoticed baseline patch.
2. Pin both effective source trees, resolved configurations and real input cache;
   finish exact account/source-field mapping; write the dated experiment spec and
   full CLI/manifest schema. No production simulation precedes registration.
3. Execute smoke on the actual local host. Check clean data/logs, exit 0, finite
   first and last outputs, two complete response rows, exact shock totals,
   counterpart reconciliation and zero-control equality. Preserve every failure.
4. If smoke and budget projection pass, execute only the registered full matrix.
   Checkpoint each completed seed/revision/arm; fail fast on NaN/Inf, missing
   household IDs, stale adapter state, identity errors or resource ceilings.
5. Deliver per-criterion PASS/FAIL/INCONCLUSIVE, all predeclared tables/plots,
   aggregate artifacts, hashes, exact reproduction commands, logs and limitations.
   Keep household panels/checkpoints in /private/tmp and aggregate evidence in the
   vault's outputs/implementation/<date>-increment-3/. Update tasks/log/context.
6. Review Increment 3 before any final production/merge decision. A good numerical
   result does not authorize refitting, a policy change or automatic promotion.

Planning checks performed on 5 November: governing links and documented source
seams checked; no harness implemented and no smoke/full run launched. The native
MPC runner's separately initialized paths and unassigned income-hook counterpart
are recorded implementation gaps for this design, not completed capabilities.

## Task 1 completion — 5 November 2026

Confirmed the user-selected exogenous-funding convention and traced native income,
wealth, bank and government seams in INCREMENT-3-ACCOUNTS.md. No runtime source
edits, harness, registration or experiment. Next task is checkpoint/clone plus
RNG/independence tests; user approval required before advancing beyond task 1.

## Task 2 completion — checkpoint/clone primitive, 5 November 2026

User approved advancing from task 1. Added the isolated experimental helper
run_model/src/experiments/paired_checkpoint.py and targeted tests in
tests/test_macromodel/unit/test_paired_checkpoint.py. Production Simulation,
Country, TimeSeries, consumption rules and the existing MPC runner are unchanged.

API: PairedCheckpoint(model) captures a private in-memory model snapshot and RNG
state; checkpoint.clone() returns an independent arm without activating global
RNGs; with checkpoint.activate_rng(): restores checkpoint streams around a whole
serial continuation and restores ambient streams on exit, including exceptions.
Capture before installing hooks; existing hooks are rejected because closures
cannot be rebound safely by deepcopy. Use only on the capture process/thread.

A local memo pre-seeds reachable TimeSeries containers, preserving aliases/cycles
without changing their class. Python/NumPy globals, both model-used Numba legacy
thread-local streams (including cached Gaussian draws), and copied object-owned
Generator/RandomState objects are preserved. The private Numba RNG ABI is checked
against installed LLVM layout/offsets/size and fails clearly if unsupported.
This is not a cross-process/file checkpoint, parallel worker RNG checkpoint or
support for arbitrary external RNG libraries.

Nine tests pass (5.87s; two existing pandas warnings), including whole Simulation
copying without iteration, independent arrays/residuals/adapter ownership,
TimeSeries aliases/cycles, dedicated RNGs, legacy-shaped copying with Country's
copy override disabled, thread rejection, unsupported layout rejection, and cached
Numba normal replay/ambient restoration. Ruff lint/format and whitespace pass.
The old-main copy seam is exercised by disabling the copy override on the current
fixture; execution of the exact archived main revision remains a later smoke gate.
Synthetic random consumers check replay, not full macro zero-shock equivalence.

No experiment registered or run, no nonzero interventions or model iterations,
no commit/push/PR. Evidence: vault outputs/implementation/2026-11-05-increment-3-checkpoint/.
Next task after review: implement the one-time externally funded income-transfer
booking path and its targeted tests; leave LFA intervention for a separate task.

## Task 3 completion — one-time income transfer, 5 November 2026

User approved task 2 and advancing. Added isolated
run_model/src/experiments/income_transfer.py and
tests/test_macromodel/unit/test_income_transfer.py. Existing production income,
wealth, banking/fiscal formulas and the existing MPC runner remain unchanged.

OneTimeIncomeTransfer(country_code, year=..., month=..., amounts_nominal=...)
accepts a frozen finite nonnegative nominal H-vector from registration, including
zero for a control. install(model) attaches pre/post hooks to one cloned arm.
The prehook uses native stage_household_income_shock exactly once on the target
date; duplicate native staging, missed dates, layout changes and reuse across arms
fail. No separate LFA or social-transfer/fiscal edit occurs.

The posthook requires native clearing of the staged receipt, exactly one new
income row, realised income = native component income + frozen transfer, native
closing household-to-bank aggregation and the bank balancing deposit equation.
It then creates one immutable TransferBooking with household receipts, per-bank
gross funding allocation and external payer outflow. Gross outflow is not inferred
from later income/deposit differences after support, spending or portfolio changes.
No new reserve asset is added. Failures leave the experiment booking unset; they
do not undo a partly executed model period, which must be discarded/restarted from
the checkpoint. Hooks are installed after cloning, as required by task 2.

Validation: 32 targeted checks pass (22 new transfer cases, 9 checkpoint cases,
and the existing country income-shock regression), 17.89s, four existing pandas
warnings. Includes a deterministic synthetic-fixture native Simulation iteration
through normal income/wealth/bank settlement, plus zero control, two-pass receipt
reconstruction, gross-versus-net accounting, immutable amounts, invalid inputs,
ownership/layout/date/duplicate guards and incomplete/nonfinite settlement errors.
Ruff lint/format and git diff --check pass. This fixture integration is not a
registered real-data macro experiment; production-scale/archived-main smoke and
full response evidence remain later gates.

No LFA intervention, experiment registration, real-data campaign, refit, commit,
push or PR. Evidence: vault outputs/implementation/2026-11-05-increment-3-income-transfer/.
Next task after review: one-time LFA endowment booking and native opening-bank
synchronization with targeted tests. Stop before that task.

## Task 4 completion — opening LFA endowment, 5 November 2026

User approved advancing from the income-transfer task. Added isolated
run_model/src/experiments/lfa_endowment.py and
tests/test_macromodel/unit/test_lfa_endowment.py. No existing production or prior
helper code changed. OneTimeLFAEndowment reuses the prior helper's fixed-vector,
date/arm/layout validation and hook installation, overriding flow staging and
settlement completely. No staged-income call or fiscal/earnings edit occurs.

Install on a checkpoint clone after unshocked initialization/residual anchoring.
On the target prehook, the helper replaces current opening LFA, its total,
financial/total/net wealth and existing wealth histogram. It synchronizes bank
deposit aggregates/totals, signed balancing deposits, liability contribution and
existing histograms without appending a period. Bank aggregation/formulas execute
on a disposable scratch TimeSeries; live history lengths remain unchanged.

EndowmentSnapshot retains immutable before/after household and bank account
vectors. Native current rows now represent the intervened opening, so saved
pre-shock evidence must use this snapshot/the untouched checkpoint, not relabel
an overwritten native current row as unshocked. Old rows, initial_states,
income, debts, priors and residuals are untouched. Zero endowment is an exact
no-op on live accounts. Duplicate calls cannot apply it twice; stale captures,
uninitialized residuals, mixed staged income, inconsistent baseline accounts,
missed dates and nonfinite arithmetic fail before account writes.

Native closing bank liability uses the previous deposit position. The helper
preserves that timing offset and adds only the endowment's current liability
contribution; it does not recompute the baseline liability against a newer row.
This was exposed by the native-fixture test and corrected locally, with a specific
lag-offset regression. Production bank behavior remains unchanged.

The posthook checks one completed income period with no added income receipt and
native closing bank aggregation/balancing, then records immutable EndowmentBooking
with gross household endowments, per-bank funding and external payer outflow.
Later endogenous deposit/wealth changes are not required to equal the injection.
Failed full settlement requires discarding/restarting the arm; no model rollback
is claimed.

Validation: 45 selected tests pass (14 LFA,22 income-transfer,9 checkpoint),23.32s,
four existing pandas warnings. Includes deterministic synthetic-fixture native
initialization/continuation, two-bank signed/overdraft accounts, before/after
snapshots, old-row and initial-state preservation, exact zero arm, independent
checkpoint/control, one-time/date/capture/income/baseline/overflow rejection and
bank-liability lag-offset preservation. Ruff lint/format and whitespace pass.
Prior production/helper hashes remain unchanged. No registered real-data campaign,
refit, commit, push or PR. Evidence: vault outputs/implementation/2026-11-05-increment-3-lfa-endowment/.
Next after review: implement two-pass diagnostic capture, without response
measurement or experiment execution in that next task.

## Task 5 completion — two-pass diagnostic capture, 5 November 2026

User approved advancing from the LFA task. Added isolated
run_model/src/experiments/two_pass_diagnostics.py and
tests/test_macromodel/unit/test_two_pass_diagnostics.py. No existing production
or prior helper code changed. TwoPassDiagnosticCapture(model) is a temporary
context manager installed only after arm cloning and removed before checkpointing.
It temporarily wraps each country's household target method on that instance,
calls the original exactly once, returns its original result and restores all
prior instance/class dispatch on normal exit, exceptions or partial setup failure.

capture.records contains immutable TargetPassSnapshot objects with country,
period/date, pass kind, native rule class, actual arguments, pre-call current/prev
household state, rule components and returned VAT-exclusive goods-demand matrix.
Prepared input capture retains b/annual_b/p, M/q, opening real/nominal stocks,
unit metadata, floor/window diagnostics, B/S window and declared price history.
No learner/evaluator replay, response computation, RNG draws or series appends.
First-pass copies survive replacement and subsequent mutation of source arrays.

All available native component keys are copied. Missing required components are
listed explicitly in unavailable_components; old-revision outputs are not
reconstructed with branch equations. Zero-growth bypasses omit potentially stale
rule component caches. In-memory records are not a persisted artifact format;
end-of-quarter realised outcomes and response measurement remain later tasks.

Validation:6 targeted tests pass, including integrated native first/replacement
calls on independent checkpoint clones with identical household histories with
and without observation; prepared-input/opening-stock copies; unchanged NumPy
ambient state; exact native call counts; immutable output and first-pass retention;
missing/bypass behavior; existing override/exception/partial-setup restoration;
and re-entry rejection. The direct-call fixture supplies finite expected inflation
because it deliberately skips estimation, matching existing integration tests.
Ruff lint/format and git diff --check pass; prior production/helper hashes match.
No model iteration or registered real-data campaign, refit, commit/push/PR.
Evidence: vault outputs/implementation/2026-11-05-increment-3-two-pass-diagnostics/.
Next after review: implement response measurement; do not register or run an
experiment as part of that task. Stop here for review.

## Task 6 completion — pure response measurement, 5 November 2026

User approved advancing from diagnostic capture. Added isolated
run_model/src/experiments/paired_response.py and
tests/test_macromodel/unit/test_paired_response.py. No existing production or prior
helper code changed. No simulation/data collection/experiment registration occurs.

measure_paired_response accepts explicit nominal T x H outcome paths, exact paired
household IDs/order/consecutive period rows, path-specific C/C0 deflators, one-time
nominal intervention vector and pre-shock control deflator. Nominal units must be
explicitly declared. Each native outcome is supplied separately: no inferred
rent/VAT/financing conversion, survey weights, .14 enforcement or response clipping.

Outputs preserve immutable per-period nominal/real differences and denominators,
plus impact/4q/8q aggregate responses and optional predeclared boolean-mask groups.
The cumulative numerator uses the same one-time denominator, not a repeated or
endogenous income increment. Stock outputs are labelled lfa_spending_response,
not income MPC. Zero injections/empty groups produce explicit NA/status/counts,
retaining their spending numerators. Individual-ratio quantiles/negative shares
use defined ratios with counts disclosed; primary aggregates include the full
supplied cohort, including nonrecipients' endogenous spending responses.

summarize_seed_responses requires registered_seeds explicitly. Missing seeds are
retained as None and make pooled mean/SD/min/max unavailable rather than silently
excluded. Complete summaries report arithmetic mean, sample SD (ddof=1), min/max;
a singleton has no sample SD. No confidence intervals or empirical acceptance
thresholds are inferred from the small diagnostic seed set.

Validation:27 targeted tests pass in0.06s; Ruff lint/format and whitespace pass.
Independent hand examples cover own-path deflation, fixed pre-shock denominator,
time-varying prices, explicit shorter smoke horizons, stock/income labels,
scale5000 invariance, negative/>1 responses, group/cohort/zero-denominator cases,
immutability, nonfinite/shape/unit/identity/horizon rejection and registered-seed
missingness. A mistaken nominal test expectation was corrected to
(640-300)/60=17/3, retaining the independently checked real response10/30=1/3;
no implementation criterion/tolerance was weakened. Prior source hashes match.
Evidence: vault outputs/implementation/2026-11-05-increment-3-response-measurement/.

This is arithmetic over caller-supplied outcomes, not collection/serialization,
classification-bin design, log-target calibration-operator reproduction or macro
validation. Next after review: implement end-of-quarter outcome collection.
No registration/run, refit, commit/push/PR; stop here for review.

## Task 7 completion — end-of-quarter outcome collection, 5 November 2026

User approved advancing from response measurement. Added isolated
run_model/src/experiments/quarter_outcomes.py and
tests/test_macromodel/unit/test_quarter_outcomes.py. No existing production/prior
helper code changed. QuarterOutcomeCollector is an in-memory posthook context,
entered after cloning and intervention hook installation, restored on exit.
Declare revision/arm/seed, next iteration and the common country/household ID order.

Immutable QuarterOutcome records include iteration/completed-row/date, frozen
household membership and IDs, selected native price series/C/C0, available native
per-household nominal income, VAT-exclusive goods, cash and housing-inclusive
consumption. Household stocks/cash-ledger/support fields, actual feasibility-plan
funding quantities, bank/government/macro fields and external payer booking are
copied separately. No financed-consumption variable, household spending from an
aggregate scalar, response computation or weighting is invented.

Missing optional/native fields and nonfinite optional values have explicit
availability reasons, with raw optional values retained. Source history counts
are recorded. Present primary outcomes must be finite H-vectors with completed
income rows; selected CPI history must align. Cohort/order, duplicate/skipped
iteration and date mismatches fail before records are committed. A due external
intervention must have passed its booking posthook first. Funding outflow appears
only in the actual booking quarter; later quarters explicitly say no_new_booking.
Existing unrelated hooks survive context cleanup.

Validation:14 targeted tests pass in12.96s, one existing pandas warning. Includes
native synthetic-fixture settlement with identical household histories with and
without collection, after one-time income-transfer verification; explicit missing
and nonfinite optional data, immutable detached values, duplicate/date/row/price/
cohort rejection, hook cleanup and one-time funding reporting. Collected nominal
paths feed response measurement with a hand-checked own-path-deflation example.
Ruff lint/format and whitespace pass; all prior production/helper hashes match.
Evidence: vault outputs/implementation/2026-11-05-increment-3-quarter-outcomes/.

No experiment registration, real-data campaign, refit, commit/push/PR. This task
collects in memory only; persisted artifacts and paired runner orchestration
remain next. Next after review: assemble the paired validation runner. Do not
register or execute a production experiment as part of that implementation task.
Stop here for review.

## Task 8 completion — paired-case runner and artifacts, 5 November 2026

User approved advancing from outcome collection. Added isolated
run_model/src/experiments/paired_validation.py and
tests/test_macromodel/unit/test_paired_validation.py. No production or prior
helper source changed. run_paired_case connects unshocked warm-up/checkpoint,
control/optional zero/transfer/LFA arms, two-pass observation, quarter collection,
response measurement and saved artifacts for ONE revision/seed/country case.
Use separate source environments/processes for the two revisions. No data loader,
refitting, configuration promotion, implicit experiment registration or real-data
launch was added. The caller must supply frozen vectors, cohort IDs, horizons,
provenance and an explicit zero-arm choice. Registration must pin/verify them.

The source model and ambient RNG state remain unchanged. Each arm starts from
the same whole-model/RNG checkpoint; the LFA arm follows completed unshocked
initialization. A requested zero arm must match control outcomes/accounts and
both target passes before nonzero arms execute. Responses include native realised
goods/cash/total consumption and both passes' requested goods, calibrated total
and nominal long-run target where available; missing native target components
are explicitly unavailable, not reconstructed with branch equations.

Artifacts stay under /private/tmp in a fresh, non-overwritten directory. Numeric
NPZ plus strict JSON preserves immutable record content, NaNs, RNG bytes, native
numeric pandas series/dataframe labels and missingness without pickle. Audit
checkpoint evidence includes household histories, income-belief state, residual,
global/owned consumption RNG states, configuration and pre-shock deflator; it is
not a portable whole-model restart file. Saved arm evidence includes both pass
records, quarter outcomes, payer booking and LFA before/after snapshots. Final
manifests include environment/resource metadata and artifact SHA256 checksums.
Caller provenance is explicitly marked unverified until the source/data pinning
step; successful fixture execution is not evidence of those production pins.

Resource checks apply at iteration/artifact boundaries, not preemptive process
kills, and case limits do not replace the registered campaign-wide budget.
Failures retain manifest/error trace and completed/partial arm evidence; no seed
replacement, silent retry, automatic refit or deletion of failed runs. Existing
output directories are refused. Unknown/non-numeric serialization types fail
clearly; native numeric pandas Series support was added after the native fixture
exposed that required source representation, with a labelled roundtrip test.

Validation:104 selected tests pass (12 runner/artifact cases plus92 existing
component cases),47.76s, seven existing pandas warnings. Includes full native
synthetic-fixture warm-up, all four arms, zero-control equivalence, measurements,
artifact reload and unchanged source model; deterministic toy tests exercise
budget abort, zero divergence, partial-arm failure preservation, preflight and
exclusive output handling. Ruff lint/format and git diff --check pass; previous
production/helper hashes match. No registered experiment or real-data campaign,
refit, branch, commit/push/PR. Evidence: vault outputs/implementation/2026-11-05-increment-3-paired-runner/.
Next after review: prepare the pinned experiment specification (source/data/config,
exact launcher and budget) for approval before registration/execution. Stop here.


## Task 9 completion — frozen experiment preparation, 5 November 2026

User approved proceeding after task 8. Frozen both effective source trees, the
common FRA2014 single-wave real cache, runtime auxiliary inputs, environment and
eight native effective configurations under
/private/tmp/inet-spec4-validation-20261105. Constructor-only inspections passed
for both revisions without regeneration or simulation steps: 5760 households,
scale5000; cohort/order/memberships/initial accounts and shock vectors match exactly.
Verified 1671 source-file hashes, 10 common inputs and 8 configuration schemas.
The 33 configuration differences are preserved for review, without retuning.

Prepared the DRAFT smoke specification, serial source-isolated launcher and
aggregate provenance/config/source-diff evidence in vault
outputs/implementation/2026-11-05-increment-3-experiment-preparation/.
The launcher defaults to verification; execution without an explicit matching
registered-specification approval record is rejected. Check mode, rejection
guard, syntax and Ruff pass. No approval record or run directory exists.

Proposed ordinary-household injections are EUR99.49096886898135 flow and
EUR397.9638754759254 stock; aggregates EUR2.865339903bn and EUR11.461359614bn,
respectively 0.508513% and 2.034052% of INITIAL quarterly GDP. These are not
measured row20 ratios. Review the exact sizes/exogenous-funding assumption before
registration; do not infer an unspecified endogenous-funding threshold.

This was preparation for smoke approval. Task 11 below completes the fixed
subgroup/tie definitions, sequential channel decomposition and analysis
validation specification, plus a controller carrying smoke time/output into the
six-hour budget. Private household artifacts remain under /private/tmp. The
controller remains verification-only until a separate campaign approval record
binds the registered design and smoke review.


## Task 11 completion — campaign analysis and budget preparation, 5 November 2026

The registered seed-12 smoke completed for main and branch. Completion, zero-arm,
cohort, booking, settlement and response gates passed. The main transfer
growth-guard increase at rows 21–22 remains a predeclared review flag. The smoke
checkpoint does not retain row-20 GDP, and `funding_plan.credit_requested` is
unavailable in every arm. See the smoke results in the vault output report.

Prepared `knowledge-vault/outputs/implementation/2026-11-05-increment-3-campaign-analysis/specification.md`.
It fixes the full 5,760-household cohort; income, LFA and aggregate-debt quintiles
with `(value, household_id)` tie-breaking and recorded cutpoint ties; raw tenure
code groups without inferred labels; all-household and subgroup outcome tables;
seed summaries; a sequential ten-block target replacement decomposition; and a
separate realized resource/accounting waterfall. Missing native fields remain
unavailable, and the decomposition is explicitly order-dependent rather than a
unique causal attribution.

Prepared `/private/tmp/inet-spec4-validation-20261105/launch_campaign.py`. Its
verification mode validates the frozen source/input/config/environment pins,
completed smoke and review-report presence, then prints the case order and
campaign budget projection without iterating. Campaign workers capture native
row-20 expenditure GDP after the unshocked warm-up and attach the value and shock
ratios to aggregate case metadata. A full run requires a separate approval JSON
binding the campaign specification, preparation manifest, smoke report, both
launcher hashes and explicit acknowledgements of the growth-guard and missing
credit-requested review flags. The controller refuses existing campaign output,
uses serial subprocesses, keeps failures, and does not retry.

Verification: Python AST parse and controller verification passed. The smoke
consumed 57.597 seconds and 742,042,079 output bytes. Estimated full-campaign
work is 208.9 seconds and 3.50 GB before reserve, or 417.9 seconds and 7.00 GB
with a 2x reserve, inside the remaining 21,542-second/20.7-GB ceilings. This is a
projection from the seed-12 two-quarter smoke, not a measured full-campaign
runtime. No full-campaign approval record or simulation exists; no tests, source
runtime edits, commit, push or PR were made.

Next gate: resolve the frozen-environment mismatch without weakening the
registered pin. The controller requires approval bound to the registered
specification, preparation manifest, smoke review and launcher hashes.


## Task 12 completion — full-campaign specification registered, 5 November 2026

Registered the frozen analysis contract at
`knowledge-vault/wiki/experiments/specs/2026-11-05-increment-3-paired-validation-full-campaign.md`.
It binds the six cases, seeds 12–14, both revisions, eight-quarter horizon,
all-household and fixed subgroup outputs, target/accounting tolerances, abort
conditions and resource limits. Its SHA256 is
`bd6196283e740085285f5518da4d3b51ff707ffd7ed243fb64992a6d9e973676`.

The guarded controller verifies the registered page's frontmatter/status and
requires its canonical path and digest in any future execution approval.
Default verification passed without simulation. No campaign approval record was
created; no full-campaign simulation was launched. Next is a separate explicit
execution-approval decision.


## Task 13 — approved campaign launch preflight blocked, 5 November 2026

The user approved execution of the registered campaign. Before creating any
campaign status/output directory or launching case 1, the controller rejected
the current environment. Python 3.12.13 and all 176 pinned distributions match;
the shared venv now also contains `polars 2.1.0` and `polars-runtime-32 2.1.0`,
which are absent from the registered 176-package manifest. No case process,
campaign status or run artifacts were created.

Evidence is recorded in
`knowledge-vault/outputs/implementation/2026-11-05-increment-3-campaign-launch-preflight/report.md`.
Restore the exact registered environment or amend and reapprove the environment
pin/controller before retrying. The environment check remains strict.


## Task 14 — campaign environment amendment registered, 5 November 2026

At the user's direction, registered amendment 1 for the campaign-only pin. It
adds exactly `polars==2.1.0` and `polars-runtime-32==2.1.0` to the unchanged 176
base distributions. The original preparation manifest and smoke environment
remain unchanged. The amended environment must contain exactly 178 packages.

Amendment JSON SHA256: `57b867f9aaa74f69edf52adc382d2b95de3e2b858731af7aecfa02ea102b9930`;
amendment page SHA256:
`77d3245d3a00ee388887c85f24b43055efcfa8a1f29be641df313bab09a85f27`;
amended campaign controller SHA256:
`f4d6640f224d243a3712e12274c275f7e621d743f9baabe2cc2692d25f5a43a7`. AST
parsing and verification-only preflight pass; no simulation ran.

The initial execution approval predates these hashes. The next gate is fresh
explicit approval bound to the base specification, amendment page/JSON,
preparation manifest, smoke artifacts and amended controller. Do not launch
until that approval is recorded.


## Task 15 — amended campaign approval recorded; launch deferred, 5 November 2026

The user approved the exact amended campaign hashes. A controller-compatible
approval record is saved at
`knowledge-vault/outputs/implementation/2026-11-05-increment-3-campaign-environment-amendment/campaign-approval.json`
(SHA256 `40c0cc3c614f80f9bcb8b5fd1afafca713c8a016682abab05a13311dff0092c4`).
It binds the campaign specification, amendment page/JSON, preparation manifest,
smoke report/launcher and amended campaign controller, and records acknowledgement
of the growth-guard and missing-credit-field flags plus row-20 GDP capture.

The user requested that this session stop before execution. No campaign case has
started. Next session: run the six approved cases serially with the recorded
approval file, then review all outcomes, accounting gates, channel decompositions,
subgroups and economic flags under the registered specification.


## Task 16 — approved six-case campaign completed; review inconclusive, 5 November 2026

The amended campaign controller passed verification with exactly 178 distributions
and executed main seeds 12–14 followed by branch seeds 12–14. All six case
manifests report complete, with three arms, eight response quarters and 5,760
ordered households. All 60 registered artifact hashes match. Row-20 expenditure
GDP is captured in every case. Campaign active time was 222.2 seconds; combined
smoke-plus-campaign output was 3,990,007,799 bytes. No controller wall/output
budget abort or case error occurred.

Pins, cohort, complete primary paths and transfer/LFA booking and account links
pass within registered tolerances. Transfer responses are positive at horizon 1;
seed signs disagree at horizons 4 and 8, with negative and above-one aggregate
responses. The main transfer growth-guard difference at rows 21–22 is 1.02–1.08
percentage points. Keep these as review flags; do not retune. The household
summary arrays omit household mean and share-above-one fields; these were derived
from retained response panels for the review. Case manifests retain
provenance_verified=false despite launcher pin checks. funding_plan.credit_requested
is unavailable in every arm and was not inferred.

Economic acceptance remains inconclusive. Main target passes explicitly mark
eight operands unavailable; branch captures expose them, but the branch-native
sequential reconstruction has not been performed. Next: report branch telescope
residuals; document unavailable main blocks and dependent channels; explain
late-horizon dispersion through target, budget, financing, price and account
paths. Do not refit or promote based on this campaign alone. See
`knowledge-vault/outputs/implementation/2026-11-05-increment-3-campaign-review/report.md`.


## Task 17 — branch-native target reconstruction complete; economics inconclusive, 5 November 2026

Replayed all 144 branch target-pass captures with the frozen branch evaluator.
Selected log-target, long-run level, ratio/map, unguarded-log-change, real-budget
and calibrated-total endpoints match the retained native outputs at saved
precision. Completed all 96 control/intervention comparisons in the registered
10-block order. Raw target, guarded and unguarded budget, calibrated total and
goods-target telescopes have zero maximum absolute residual at saved precision,
within the registered tolerances. Block 10 reports the endpoint guard/floor
effect separately; the decomposition remains an order-dependent accounting
path, not a unique causal attribution.

All 144 main target passes still mark eight operands unavailable: behavioral
income base, annualized income base, original PI ratio, monetary income M,
residual epsilon, raw log target, raw target level and unguarded dynamic log
change. Thus main block 1, block 6 and block 10 cannot be independently
reconstructed; dependent sequential contributions and the main telescope are
unavailable. The saved guarded budget and growth-guard rates remain observations,
not equation validation. Main row-28 target-budget differences include
-EUR 1,230.276bn for seed-12 transfer and -EUR 2,001.434bn for seed-13 LFA,
while realized consumption changes are +EUR 0.568bn and -EUR 2.318bn. Do not
infer raw operands from the branch equation.

Branch row-28 transfer target-budget differences are +EUR 3.124bn, -EUR 2.875bn
and -EUR 3.275bn for seeds 12–14; realized consumption differences have the
same signs. The ordered decomposition places most of this sign spread in the
combined income/price/PI, debt/rate and ECM blocks. Branch LFA target-budget
differences are negative in all seeds; seed 13 nevertheless has a +EUR 0.318bn
realized-consumption difference. Its CPI difference is only +0.000063, while
LFA/debt/GDP paths differ materially from control; this remains unresolved
mechanism-level transmission, not a decomposition failure.

The branch numerical target gate passes. Main reconstruction and overall
economic acceptance remain unavailable/inconclusive. Keep all registered
response and guard flags. No run was repeated, no missing value was inferred,
and no coefficient or specification was changed.

Follow-up calculations and per-seed paths: `knowledge-vault/outputs/implementation/2026-11-05-increment-3-target-review/report.md`.
The analysis script and numeric artifact remain under
`/private/tmp/inet-spec4-validation-20261105`; the hash-bound campaign
specification remains unchanged.
