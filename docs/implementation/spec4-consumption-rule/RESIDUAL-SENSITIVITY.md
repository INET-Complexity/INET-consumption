# Once-drawn residual sensitivity — detailed proposal

**Date:**6 November2026. **Revision:**2, technical-review corrections incorporated. **Status:**REVISED DRAFT FOR REVIEW; no implementation, registration, initialization, draw or simulation authorized by this document. Governing contract: INET `docs/implementation/spec4-consumption-rule/PLAN.md`, section4. Increment2 remains approved; Increment3 economic acceptance remains inconclusive. This diagnostic does not waive its outstanding limitations or select a production residual policy.

## 1. Question and comparison

Does the retained initial-state residual anchor materially determine initial target gaps, consumption dispersion, transition dynamics, financing stress and subsequent transfer/LFA spending responses?

Use the same integrated consumption implementation in both policies:

| Policy | Residual | Interpretation |
|---|---|---|
| A: retained anchor | `idiosyncratic_persistence: initial_state` | Back out epsilon once from the previous desired real budget and deterministic target at the first native evaluation, then keep it fixed. |
| D: once-drawn | `idiosyncratic_persistence: fixed_effect`, explicit residual seed | Draw epsilon_i~N(0,sigma_hat²) once in the existing dedicated generator and retain it for the entire run. |

Current source/config inspection finds sigma_hat=0.2887146213259912 and long-run intercept−0.3536169894042738. These are existing fit values, not new estimates. Verify them again in the effective pinned config; mismatch blocks registration pending explanation, rather than silently overriding newer work. No normal-mean correction, demeaning, sample rescaling, truncation, residual clipping, refit or quarterly innovation. A finite normal sample is not required to have exactly zero mean or exactly sigma_hat SD.

Use the integrated branch, not old main, as the policy comparison. Main has a known SMIC population-scale defect and differs in other target settings. Preserve it only as existing historical evidence. Live YAML remains `initial_state`; proposed experiment configuration overlays select D in isolated files, never overwrite the live policy.

## 2. Falsifiable engineering claims and limits

Before guards, on identical prepared inputs and lagged budget:

`epsilon_A = log(C_previous_real) − log(b) − x_deterministic`

`log(C_star_D) − log(C_star_A) = epsilon_D − epsilon_A`

The latter equality is checked directly in log space with fixed inputs and coefficients; it is not checked by subtracting huge exponentiated levels. For A, verify the backed-out raw long-run target against the lagged desired budget on finite, unguarded fixtures; separately disclose numerical raw-level caps/floors. For D, reproduce the exact generator draws with the pinned NumPy/bit-generator/seed/version and require identical residual-vector hashes at every later pass/clone.

These are testable translation/lifecycle claims. The economic question has no preselected preferred sign, MPC value, dispersion target or production winner. A changed intercept/sigma affects D differently from backed-out A; this main comparison does not add coefficient perturbation arms. Any such follow-up would be separate. Three-by-three seed diagnostics do not estimate household sampling uncertainty or identify structural residual persistence.

## 3. Proposed frozen design and amounts

| Item | Proposed setting |
|---|---|
| Source | New private frozen snapshot of the reviewed integrated branch, including approved dirty implementation and diagnostic-only additions; record HEAD, diff and per-file SHA256. Reconcile live changes against the earlier freeze before choosing the exact snapshot. |
| Data | Same verified FRA2014 cache/cohort as Increment3, H5760, scale5000, fixed household IDs/order/membership. Hash and verify; no regeneration, new wave or fixture substitution. |
| Config | Identical native effective config except persistence and explicit D seed. Preserve income/PI/stock timing, caps/tail, maps, coefficients, ECM, floor/support, finance and VAT policies. Report every effective config difference. |
| Model seeds |12,13,14. |
| Draw seeds |12001,12002,12003, crossed with every model seed for D. Existing dedicated `default_rng` mechanism; pin the concrete bit generator after inspection. The same draw seed gives the same vector across model seeds with the same cohort. |
| Cases |3 A cases plus9 D cases =12 unique cases. Reuse A within each model seed when comparing with the three D seeds; never count those repeated references as additional independent A replications. |
| Startup |Select policy/seed in effective config before model construction or first target evaluation. Keep supplied row0 balances and desired budget; do not overwrite row0 consumption to force a match to a random target. Capture constructor/startup target events if any. |
| Warm-up |20 unshocked quarters per policy/model/draw case. Retain all startup/warm-up diagnostics; they are outcomes of the policy, not discarded initialization noise. |
| Responses |After warm-up, clone each case into control, transfer and LFA arms; eight quarters, rows21–28, same timing as Increment3. |
| Flow |Same frozen nominal household vector as Increment3: ordinary-household99.49096886898135, aggregate2,865,339,903.426662 model EUR, delivered once. |
| LFA |Same frozen nominal vector: ordinary-household397.9638754759254, aggregate11,461,359,613.706648 model EUR, delivered once at opening. |
| Financing |Existing externally funded convention and native receipt/LFA/bank links; one payer debit. Report intervention/GDP and intervention/income ratios under each policy's own row20 state. No new institution or domestic financing assumption. |
| Primary response |Own-path CPI/C0 real spending difference, normalized by the one-time injection deflated at that policy case's pre-shock control price. Also show nominal differences and raw real numerator differences between policies. Policy-specific denominators may differ; do not interpret their ratio difference as a pure numerator difference. |
| Subgroups |Reuse registered common-initial-cache income/LFA/debt quintiles and raw tenure codes/tie-breaking. Do not regroup by policy outcomes. |

Every source/config/vector/seed identity becomes exact only at preparation and registration. This proposal does not pin a future dirty source state by HEAD alone.

## 4. Initialization, reset and clone contracts

1. Create a fresh model and fresh consumption-rule instance for each unique case. Configure persistence/seed before any target call, including potential constructor evaluations. Inventory where first evaluation occurs; install the startup observer early enough to capture it. The first-evaluation boundary must be confirmed before implementation is accepted.
2. A uses the existing native back-out; D uses `_epsilon`'s normal draw. Do not back out or re-anchor D to the supplied initial consumption level. If the cache is touched early during construction, log that first draw and require all later calls to reuse it.
3. Save the initial supplied desired budget, deterministic raw target at epsilon0, residual, raw log target, native guarded budget and associated prepared inputs. Distinguish supplied row0 state from the first computed target/realized row; do not call them the same observation.
4. Fresh construction with the same policy/seeds must reproduce the residual vector and dedicated RNG state. Different D seeds must have distinct residual hashes. No diagnostic observer may consume randomness or populate the cache earlier than the native lifecycle.
5. Live `set_run_seed` currently returns immediately when `idiosyncratic_seed` is explicit. Integrated `Households.reset` currently force-rebuilds the consumption rule when aligned income and income-belief learning are enabled; this is evidence of an intended fresh cache, not proof of a reset defect. Therefore do not use it to claim a fresh D realization or cleared cache. The run controller must construct a fresh rule/model for a new case; test normal country/simulation reset to ensure it reconstructs/reseeds as intended. If it does not, stop and propose the smallest isolated lifecycle fix with impact analysis before editing existing symbols. No production reset change is pre-authorized by this plan.
6. Clone only after each policy's own unshocked warm-up. All control/shock/zero clones in that case retain exactly the same residual vector, beliefs, histories, lagged budgets and RNG snapshot. Residual arrays must not alias mutable arrays across clones. Never re-draw or back out a residual after applying a shock.
7. For each model seed, establish and record one reproducible **pre-construction** non-residual RNG snapshot: Python, NumPy, used Numba streams (including cached normal state) and other owned model generators. Restore it before fresh A and every D construction for that model seed; new subprocesses must perform the same explicit seeding/restoration. Dedicated residual generators remain separate and use only the declared policy seed. Native Simulation.from_datawrapper seeds NumPy/Numba after country/market construction, so its configured seed alone does not satisfy this requirement. Before the first residual-dependent evaluation, require exact cohort/order/membership equality, numeric equality of supplied balances, desired budgets, histories, beliefs and other non-residual economic inputs (rtol1e-10/atol1e-12 ordinary euros for monetary states), and exact non-residual RNG-state equality across A/D. Allowed differences are persistence, D's explicit draw seed, dedicated residual generator and the residual cache when naturally initialized. Enumerate these exceptions; do not exclude arbitrary state differences. If first use occurs during construction, install the observer before construction and compare the last common boundary before that use. An unexplained mismatch blocks smoke. Do not overwrite later policy-dependent outputs to make the comparison pass. After residual-dependent evaluation begins, endogenous paths and RNG consumption may diverge; compare each case to its own unshocked checkpoint. Their row20 states need not match.
8. Fixed cohort only. Reject changed household order/count/membership, including append/shrink, for this experiment rather than invoking the runtime cache-extension feature. The residual cache uses array positions; no silent remapping.

## 5. Required observer and accounting coverage

Implement an isolated experiment observer/collector extension; retain existing production arithmetic. Inspect the exact native sources before finalizing field names; a concept below is not permission to invent a native field.

At initialization, both planning passes during warm-up and every response quarter retain:

- Source date/pass/cohort; b/annual_b, B/S/M/q/rebased PI, CPI/C0 and opening stocks/debts/principal; coefficient maps/ratios/cap/tail flags; deterministic/raw log target, residual/vector hash, guarded target and ECM/cashflow/uncertainty terms.
- Supplied initial and lagged desired budgets; before/after raw-level and growth guards; household count and source row lengths. Obtain intermediate operands directly; do not define a guard block as a balancing remainder. Prefer log-space checks before exponentiation and explicitly report precision limitations.
- Native first-pass actual `live_credit_requested`, pre-exclusion carrier request if present, received/granted credit and any native credit-rationing gap as distinct objects, with within-quarter timing recorded.
- At post-labour feasibility refresh: financing income, mortgage proceeds, goods cash uses, cash rent, taxed investment, scheduled consumer/mortgage service, property payments, available LFA, LFA draw, reserved/settled liquidation and fixed credit grants. Capture the actual passed vectors read-only at the refresh boundary, not recomputed approximations from closing income.
- Native before-floor residual, allowable cut, actual cut, top-up, before/after consumption, remaining shortfall, payment relief and support. Top-up can also be reported as explicitly derived if no native field exists; retain its source operands and label it. Do not read `credit_requested` from a post-grant object that has no such attribute.
- Closing real/nominal native goods/cash/including-housing outcomes, wages/benefits/support/taxes where native, LFA/debt/deposits, account residuals, external payer, employment/GDP/prices/rates. Missing optional fields remain explicit; required cash-need operands missing at source block the sensitivity's economic interpretation gate.

Retain full household panels and vector/RNG evidence only under a new `/private/tmp` run root; export aggregate tables/figures/hashes to the vault. Preserve the earlier Increment3 freeze unchanged. Source additions outside the observer/harness require a separate concrete scope finding.

### Measurement ledger: sources, dates and distribution domains

This ledger fixes estimands before any result inspection. At each completed quarter t, read native `households.ts` rows for `consumption`, `consumption_cash_expenditure`, `consumption_including_housing` and `income`. Require matching economic date/completed row and source-history lengths; a stale row is an error. Closing `income` is the realized-income denominator for descriptive C/Y only, never a substitute for the financing-income operand captured at its own boundary.

Let d_t be that case/arm's closing selected fixed-basket CPI/C0. Define C_t=nominal native `consumption`/d_t and Y_t=nominal native `income`/d_t. Primary realized C/Y is C_t/Y_t, equivalently the same-date nominal ratio; Y must be strictly positive. Report separate secondary ratios for `consumption_cash_expenditure` and `consumption_including_housing`, using that same Y and explicit field labels. Never rename desired `goods_demand` as realized C.

| Object | Exact source / convention | Availability and domain |
|---|---|---|
| Supplied desired budget | Supplied row0 `cacf_real_consumption_budget`; already real in base-C0 units. | Separate startup distribution; not first computed target or first realized consumption. |
| Raw target | Each first/replacement pass's `target_consumption_raw_log_target`, and `target_consumption_raw_target_real` explicitly labelled as native numerically capped raw level. | Log-target distributions use the native raw log directly; do not recover it by logging a capped level. |
| Guarded desired budget | Each pass's `target_consumption_real_budget`, already real; keep pass/date tags. | No second deflation. |
| Realized consumption levels | Closing native `consumption`, cash expenditure and including-housing fields, separately, each divided by closing d_t. | Report all-household finite level distributions, including zero/negative values; nonfinite required native outcomes fail capture. |
| Primary realized C/Y | Closing native `consumption` divided by same-row native `income`. | Defined where Y>0, including C<=0. Count Y=0 and Y<0 separately. No floor/replacement denominator. |
| log(C) and log(C/Y) | Natural log of real C; natural log of the primary C/Y. | log(C) requires C>0; log(C/Y) requires C>0 and Y>0. Report excluded-domain counts; do not clip/floor inputs. |
| Behavioral C/b diagnostic | Closing real C divided by **replacement-pass** `target_consumption_income_base` for the same economic quarter, already real relative to the same C0. | Separate planning-income diagnostic, explicitly labelled with planning/closing observation times. Verify common C0 and b>0; never relabel C/b as realized C/Y. |
| Startup realized measures | Native row0 realized C/income only if genuine same-date initialized values are available and tagged. | Otherwise NA with reason; do not use the first future realized row as startup data. |

For every level/ratio/log distribution, report total cohort count, finite/defined count and applicable nonpositive-income/consumption counts, unweighted arithmetic mean, cross-sectional population SD (`ddof=0`), median and p10/p90 (`numpy.quantile`, `method='linear'`). Every model household appears once; no survey weights or extra population expansion. Conditional-domain statistics must state their domain/count in the table heading. Keep all households in native levels/aggregate accounting and in the full response cohort; domain restrictions affect only the specific undefined ratio/log statistic. Empty domains return NA, not zero. Share/count statistics use the full cohort denominator unless explicitly named as a defined-domain conditional share. Across-seed SD remains `ddof=1`, distinct from cross-sectional SD.

No new empirical dispersion target or automatic policy-selection threshold is introduced by these measurement definitions.

## 6. Verification before real-data execution

Targeted tests only where they verify a meaningful contract:

| Test | Required result |
|---|---|
| Cross-policy initial state |Mandatory pre-construction snapshot restoration and same pre-residual economic inputs/non-residual RNG states across A and all D cases for each model seed; only declared exceptions permitted. Unexplained mismatch blocks smoke. |
| Measurement definitions |Same-date C/Y nominal-real ratio equivalence, correct closing versus planning deflators, explicit ratio/log domain counts, ddof/quantile conventions and empty-domain NA behavior verified on synthetic fixtures. |
| Exact draw reproduction |D vector matches dedicated generator with declared sigma/seed/cohort; no demeaning/rescaling; global RNG snapshots unchanged by residual draw. |
| First-use persistence |Same epsilon for first/replacement, base/perturbed formula probe, ordinary target calls and later dates. A backs out only once; D never invokes initial-state back-out. |
| Identity and size |Same IDs/order; reject unexpected cohort change. |
| Clone/reset/fresh-run |Independent owned arrays/generators; same-case clones preserve vector exactly; fresh same-seed case reproduces; reset behavior verified despite explicit-seed short circuit. |
| Conditional raw equation |Log-target D−A equals epsilon_D−epsilon_A on common inputs within rtol1e-10/atol1e-12. Verify prepared path and both passes; no raw-level cancellation used to establish the log identity. |
| Native cash waterfall |Recorded inputs reproduce financing need/LFA/grant/liquidation/floor arithmetic at registered ordinary-euro tolerance. Verify observer leaves histories, targets, RNG and settlement unchanged on synthetic deterministic cases. |
| Isolated configuration |Only declared policy/seed differences, matching residual standard deviation/intercept and every other effective parameter. Live configuration unchanged. |
| Zero intervention |Same-case zero/control paths agree at smoke, including both target passes, residual/RNG/state and closing outcomes. |

Refresh graph impact before existing-symbol edits; run relevant focused regressions and Ruff/format/whitespace. Do not claim the existing `fixed_effect` implementation is sufficient until these lifecycle/integration tests pass. New observer fields must not mutate or trigger native producers twice.

## 7. Proposed smoke and full-run budgets

Smoke: model seed12, A and D(draw12001), each20-quarter warm-up plus2-quarter control/zero/transfer/LFA continuations. Two unique cases,56 iteration-equivalents. Constructor and initial conditional checks precede smoke. Capture all warm-up rows, both passes and cash operands from the outset. Zero/control equality is required separately in both policies. Review initial gaps/guard/funding stress before full campaign approval; same shock amounts and external financing remain proposed until exact specification is approved.

Full:3 A+9 D cases, each20 warm-up+3 arms×8 quarters =528 iteration-equivalents. Including smoke,584. Execute serially, one worker, BLAS threads1. No implicit reuse of partial smoke as a full case; full outputs get fresh roots.

Proposed ceilings: smoke15 minutes; smoke+full2 hours total; process RSS16GiB boundary checked; retained output40GiB, including smoke and added warm-up diagnostics. Previous six-case timing suggests several minutes, but expanded capture and9 draw combinations have not been benchmarked. Preparation must project actual output/time from the smoke with a2×reserve before full approval; if it exceeds the proposed cap, stop and amend explicitly. No seed substitution, automatic retries, row drops, silent horizon shortening or overwrite. Keep partial evidence/errors.

The control policy's three warm-ups are reused only as reference results in analysis, not rerun for each D seed. Full12-case results, not a27-independent-comparison fiction, are the primary complete matrix.

## 8. Analysis and predeclared assessment

Report separately:

1. Initial supplied C versus deterministic/epsilon-shifted raw target and guarded desired budget; distributions of epsilon, raw target gaps and residual correlations with initial income/LFA/debt/tenure. A's residual need not be normal or match fitted sigma. D's finite-sample moments are descriptive, not forced pass criteria.
2. All20 warm-up quarters: target and realized levels, the separate level/C/Y/log(C)/log(C/Y)/C/b distributions defined in the measurement ledger, with domain counts disclosed, growth-guard and ratio-cap/tail frequencies, financing/floor/payment/support/debt/deposit paths. No dropped “unstable” households.
3. Own-policy control/flow/LFA paths: impact and cumulative4q/8q nominal/real spending responses, household mean/median/p10/p90/negative/above-one shares, all fixed groups. Report actual request/grant and cash/floor waterfalls with dates and source fields.
4. Policy contrasts: D−A at each model/draw seed, both baseline levels and within-policy intervention responses. Show all9 D outcomes, the3 model-seed means for each draw, draw variation within each model seed, and the complete matrix. Correlated repeated A references and the crossed seeds preclude treating nine paired differences as independent Monte Carlo replications. No confidence intervals are claimed from this grid.
5. Independent channel equations: raw log residual identity, native guard effects and observed cash-resource reconciliation. A sequential decomposition must declare order and verify both endpoints; do not force closure with a final balancing allocation.

| Gate | PASS | FAIL / INCONCLUSIVE |
|---|---|---|
| Pins/cohort/effective config |Exact frozen source/cache/env/vectors/config and cohort match. |Mismatch:abort before execution. |
| Residual lifecycle |Exact seed/vector/RNG hash reproduction and persistence across each case's calls/arms. |Redraw, re-anchor, stale cache or RNG contamination:fail. |
| Cross-policy startup |Exact pre-residual cohort/non-residual RNG metadata; common non-residual economic inputs meet rtol1e-10/atol1e-12 ordinary-euro state tolerance, with only listed residual-policy exceptions. |Unexplained mismatch:fail before smoke; within-policy zero equality cannot substitute for this gate. |
| Pairing/zero arm |Exact identity/RNG metadata; numeric states within rtol1e-10/atol1e-12 ordinary euros. |Unexplained divergence:fail; no full campaign. |
| Conditional target |Log/ratio rtol1e-10/atol1e-12; monetary rtol1e-10/atol1e-7 ordinary euros. |Unexplained discrepancy:fail. Precision failure stays failed until justified independently, not “fixed” by wider tolerances. |
| Funding/booking |One receipt/endowment/payer entry; native accounting and cash waterfall meet rtol1e-10/atol1e-7 ordinary euros or stricter existing identity tolerance. |Missing required native operands:inconclusive economic gate; duplicate/absent booking or identity breach:fail. |
| Matrix/required outputs |All12 cases/seeds/horizons/households and finite required data; both passes and warm-up retained. |Missing case/rows/nonfinite required outcomes:abort and inconclusive campaign; retain evidence. |
| Economic initialization |Explain target gaps/transition/financing and all response flags through verified native stages. |Mechanically correct but economically unclear:inconclusive. No predeclared superiority or empirical MPC target. |

Retain Increment3's economic review triggers: negative aggregate transfer response, transfer response>1, mixed seed signs, baseline growth-guard mass>1%, intervention-control difference>1pp. Apply baseline guard flag during both warm-up and response windows, labelled separately. Negative conditional stock responses alone are not failures. These flags require explanation, not refitting. Do not add retrospective rejection thresholds for initial dispersion/gaps after inspecting results.

## 9. Deliverables and independently reviewed implementation tasks

1. Review this proposal and settle substantive ambiguity; record exact source/call map and fresh-instance/reset contract. No run authorized by plan review alone.
2. Implement/test isolated configuration and lifecycle verification using existing modes; any required runtime fix gets its own impact analysis/review. No real model run in this task.
3. Implement/test read-only startup/warm-up/funding captures and reuse existing checkpoint/interventions/measurement helpers. Verify non-interference and required field availability before freezing.
4. Pin source/data/config/env/vectors, draft concrete smoke specification/controller and budget. Register only the final reviewed design; require an explicit approval record bound to its hashes for smoke execution.
5. Run approved smoke; report numerical checks, initial transitions and resource projection. Full12-case campaign requires its own concrete registered matrix and execution approval after smoke review.
6. Run approved full campaign, report all gates/tables/plots/limits and propose whether any residual-policy change is justified. Production default stays A unless the user separately selects a change; final PLAN§6 review remains later.

Stop after each independently reviewable implementation task. No live YAML change, new fit, experiment launch, commit, push, PR or merge is authorized by this draft.

## Source observations used for this proposal

Read-only inspection on6 November2026: `CreditAugmentedConsumption._epsilon`, `_initial_state_epsilon`, `set_run_seed`, constructor persistence/seed validation and active YAML. `_epsilon` already implements cached `fixed_effect`; initial_state back-out is in the native compute path. Explicit seed short-circuits `set_run_seed`. Exact construction/reset/first-target call chains and effective-config routing remain task1 verification, not assumed proven by those method reads. No tests or initialization were run to prepare this document.

## Technical review disposition — revision2, 6 November2026

The two P2 findings in the vault's `outputs/implementation/2026-11-06-residual-sensitivity-review/report.md` are incorporated: mandatory pre-construction RNG/common-state gate and exact measurement ledger. The integrated reset force-rebuild is now explicitly acknowledged; lifecycle tests remain required without presuming a runtime bug. Source/lifecycle mapping is the next independently reviewed task. These documentation corrections do not implement, register or authorize the experiment.

## Task1 completed — source/lifecycle mapping, 6 November2026

Source/call mapping and bounded synthetic verification completed.61 existing prepared/country-reset/checkpoint regressions pass;31 additional two-household lifecycle assertions pass for draw seeds12001–12003 (PCG64). First prepared call must draw natively: pre-calling `_epsilon` fills a cache without prepared layout and is rejected. Integrated reset force-rebuild remains the intended path; explicit-D country reset and full cross-policy real-data initialization gates are still future verification. Startup/warm-up observers must detach before checkpoint and reinstall per cloned arm. No runtime changes, real-data initialization or experiment. Evidence: vault `outputs/implementation/2026-11-06-residual-lifecycle/report.md`. Next independently reviewed task2: isolated config/lifecycle assertions, including explicit-D reset and common initial boundary; no draw for the production cohort or simulation authorized by this completion.

## Task2 completed — isolated config/lifecycle tests, 6 November2026

Added experiment-only residual_sensitivity_config.py and30 tests. All91 tests (new30+prior61) pass,25.67s; Ruff/format/whitespace pass. All12 synthetic A/D constructions preserve common restored non-residual state/global RNG and empty initial cache. Explicit-D full Simulation.reset rebuilds rule and exactly reproduces all three draw-seed vectors; no reset fix needed. Twelve validated draft configs exported privately from Increment3's frozen branch config, not a new runtime/source pin or execution approval. Protected production source hashes match task1; live YAML unchanged. Evidence: vault `outputs/implementation/2026-11-06-residual-config/report.md`. Next task3 read-only startup/warm-up/funding captures after review. No real-data initialization or simulation.

## Task3 completed — in-memory observers, 6 November2026

Added isolated residual_diagnostics.py and8 tests.67 combined tests pass,45.59s; Ruff/format/whitespace pass. Native startup/first-use residual/two-pass target, actual credit request, scheduled-service, cash-refresh component and floor/settlement snapshot boundaries captured read-only; derived epsilon-zero log target/top-ups explicitly labelled. Both-policy native histories/RNG match unobserved synthetic controls; cleanup/errors/immutability/thread/cohort/checkpoint guards covered. Post-payment routing is stub-tested, not full accounting proof. Active checkpoint capture refused; detach/reinstall perarm. Protected existing core/config hashes unchanged. Evidence: vault `outputs/implementation/2026-11-06-residual-observers/report.md`. No real-data initialization/run, persistence/controller or final source freeze. Next reviewed preparation task: compose observers/closing outcomes, strict source/config/cohort/RNG/capture gates, draft smoke registration/controller/budget; execution requires separate exact approval.

## Task4 completed — frozen smoke preparation, 6 November2026

Prepared frozen source/data/config/environment and isolated controller with reviewable two-policy smoke specification (model12, D draw12001,20q warm-up plus four2q arms per policy;56 iteration-equivalents). Real-data constructor-only common economic/global RNG/cohort/empty-cache gates pass;15 non-simulation controller tests and verification-only CLI pass. Full worker/native capture/zero/accounting gates remain unexecuted. Evidence: vault `outputs/implementation/2026-11-06-residual-smoke-preparation/report.md`. No real-cohort target evaluation, residual draw or simulation iteration; canonical specification not registered and approval template false. Next review concrete specification/controller, then registration and separate explicit smoke execution approval. Full12-case campaign requires later separate approval.

## Task5 completed — approved smoke, 6 November2026

User explicitly approved registered two-policy smoke and final hashes. Both cases passed initial/first-target common-state/RNG, native residual persistence, funding/floor/payment, zero/control and intervention gates:56 quarters/112 passes,72.982s/1.298GiB. h2 goods transfer response A.541046/D.394209; LFA A.320383/D.163466. D initial guard8.697917%, quarter20 consumption23.26% lower and support higher; tiny-income C/Y spikes retained. Engineering confirmed, residual choice inconclusive. Evidence: vault `outputs/implementation/2026-11-06-residual-smoke/report.md`. Full528-equivalent crossed design projection with2xreserve22.94min/24.48GiB; concrete full registration/budget and separate execution approval remain. Live initial_state unchanged; no full campaign/refit/production change.

## Full task6 preparation registered — 6 November2026

Prepared exact3A+9D/528-equivalent full matrix/controller from unchanged smoke source,12 native real-cache constructor-only pairing checks and21 controller tests pass. Frozen common-cache subgroup rules and h1/h4/h8 responses, all warm-up/native funding evidence retained; exact source/config/environment/smoke prerequisite verification passes. Registered `wiki/experiments/specs/2026-11-06-residual-sensitivity-full.md` in vault. Measured2xreserve projection22.94min/24.48GiB; actual ceiling1h/16GiB boundary RSS/40GiB retained incl smoke, within proposed budgets. Evidence: vault `outputs/implementation/2026-11-06-residual-full-preparation/report.md`. No full simulation; approval false. Next explicit execution approval then continuous full run/analysis/report.

## Task6 complete — approved full comparison and analysis, 6 November2026

Executed all3A+9D cases under exact registered approval:528 quarters/1056 target passes,672.218s; combined retained simulation/analysis≈13.94GiB. All runtime engineering and independent residual/log/seed/cross-smoke/12960-response arithmetic checks pass;9720 D−A contrasts and all groups/metrics/plots reported. Q20 D goods mean190.760bn vs A243.585bn (−21.69%); support4.505bn vs2.903bn. Longer-horizon responses retain model/draw sensitivity; near-zero native income produces ill-conditioned C/Y, preserved with domain counts. Evidence: vault `outputs/implementation/2026-11-06-residual-full/report.md`. Recommend retaining live initial_state; no production change, refit, commit or merge. This completes the registered residual diagnostic; no additional residual experiment required to close it. Final integrated implementation/config review and explicit disposition of retained economic/numerical/provenance limits remain under PLAN§6; diagnostic success does not grant acceptance.
