# Final calibrated consumption review and acceptance — 6 November 2026

**Verdict: implementation ACCEPTED with the limitations below. No blocking correctness findings.** The retained calibrated rule, its live income/stock integration, diagnostics and registered validation implementation satisfy the reviewed implementation contract. Retain the current `initial_state` residual policy and existing effective coefficients/maps/caps/dynamics. No further implementation change or residual experiment is required by this review.

Authority: the user explicitly requested “Do final review and acceptance.” This record completes PLAN section6's technical review and implementation acceptance; it does not claim empirical macro-response validity, silently pass historical failed checks, or authorize commit/push/merge. Acceptance is for the intended calibrated-consumption implementation and controlled research use with disclosed limitations.

## Scope and exact version

Reviewed `feature/spec4-consumption-rule`, HEAD `96c897e507844adb7f5545bcd813c600cf3daa49`, including dirty tracked and untracked implementation/tests. Compared the complete effective credit_augmented configuration and consumption/integration diff to pinned main `a3d48e5c`; that remains the rollback reference. No runtime source/test/YAML edit during review.

The tracked baseline diff has13 files, but alone omits the new adapter, experiment helpers and tests. scope.json separately inventories/hashes all tracked/untracked files; all30 changed/new Python files were inspected at their applicable contracts and linted.242 live runtime/config files match the approved full-comparison freeze byte-for-byte. Current acceptance applies to these pinned runtime/config bytes, not an arbitrary future dirty tree.

The preexisting committed Orbis dataset-creation notebook (`96c897e5`) is separate data-preparation work, excluded from consumption acceptance; no notebook execution or data regeneration. No separate initial-state PR/worktree changes are accepted by this review.

## Findings and changes in this review

No actionable code correctness, API compatibility, state/race, accounting or input-timing defect identified. No correctness fix was required. The graph identifies HIGH household target/housing call paths and critical aggregate change scope; these were checked rather than interpreted as automatic defects. No code symbol was edited.

One documentation inconsistency corrected: PLAN's upper status and section4 still called residual sensitivity unrun/queued while its latest ledger recorded completion. Those passages now reflect completed smoke/full execution and unchanged-default decision; section6 and the consolidated status link this acceptance record. Historical experiment outcomes and frozen registered specifications remain unchanged.

## Contract-to-evidence review

| Contract | Verdict | Evidence/interpretation |
|---|---|---|
| Retained fit and explicit effective YAML | PASS | Effective parameters resolved from production FRA loader, saved in effective-credit-augmented.json; full YAML saved. Intercept−0.3536169894042738, sigma0.2887146213259912, separate LFA gamma map, alpha[.1,.9], fixed IFA.022/HA.013, fixed references, explicit winsorization and log tail tau.95 retained. |
| Notebook/static translation and .14 calibration operator | PASS | Fresh35-check replay over9679 retained households, protected16 input/source hashes unchanged. Finest result0.14000000006141156 within original absolute1e-7; no fit, endpoint adjustment or notebook execution. |
| Income base and CPI alignment | PASS | Same-date floors on each behavioural-income history row; up-to-four-quarter log smoothing of individually deflated observations, period b and annual_b; initial CPI need not1. Current-price missing-row contract explicit, no price padding by guess. |
| Permanent-income rebase | PASS | Existing M/q copied from one ready pass-specific producer result: p=q+log(M_nom/(CPI/CPI0))−log(b). No B-based substitution, producer duplication, belief re-estimation or lognormal correction. |
| Opening balance sheet and exposures | PASS | Country captures closing t−1 stocks/principal before first target; replacement reuses snapshot, updates pass-specific b/PI/rates. Main-residence housing; NLA uses ledger debts including arrears; rate weights use distinct contractual principal. |
| Maps, rates and dynamics | PASS | Ordinary alpha/gamma update from opening stocks and current b; income/asset probes retain declared distinct conditioning. Annual real rate level−.72*r, nominal repricing cashflow−.003*delta_rate*ledger/(annual_b), uncertainty change−.005, ECM.56, beta1=0 retained. Wealth-tail, ratio caps, raw-log±50 and quarterly desired-growth±.5 guards distinct. |
| Residual lifecycle and policy | PASS | Native initial anchor to previous desired real budget; same epsilon for both passes/probes/clones. Reset reconstructs integrated rule and restores priors; same explicit D seed/reset verified. Sigma does not impose A dispersion. Cold-start intercept/sigma fixture records intercept offset by backed-out epsilon; full sensitivity documents A/D distributions/levels. |
| Legacy dispatch and pure probes | PASS | Aligned/learning four-way dispatch and zero-growth bypass covered; missing ready integrated operands fail explicitly. Conditional income fixes logYp/alpha/gamma, conditional assets fix gamma and update alpha after ratio clipping; historical formula response remains separately labelled. Pure evaluation does not initialize residuals or consume RNG. |
| Nominal goods/housing/VAT and finance boundaries | PASS | Desired full budget, rent/imputed rent removal, VAT-exclusive goods, post-labour feasibility/floor/payment and closing outcomes remain distinct. Existing two-period accounting tests plus registered native funding/booking checks pass. Floor unit fix expands external SMIC once by scale, fallback unchanged. |
| Reset/deepcopy/checkpoint and observer integrity | PASS | Immutable input snapshots; independent mutable state/RNG, TimeSeries aliases preserved. Scoped observers detach before checkpoint/reinstall per arm, never call producers merely to obtain data. Serial process/thread ownership enforced; private Numba RNG layout validated, not treated as portable/parallel API. |
| Registered implementation diagnostics | PASS for candidate engineering | Increment3 conditional endpoints and288 independent branch guard checks; approved residual smoke and12-case full run pass source/RNG/cohort/epsilon/native cash/floor/payment/booking gates. Full saved-data audit1056 passes,528 quarters,12960 response cells/9720 contrasts; common-first-log identity max≈2e−15. Historical main failures remain failed. |

Key current source anchors: `consumption_inputs.py:288` preparation and `:397` capture; `country.py:1766` planning dispatch, `:423` deepcopy; `households.py:1838` target caller and `:710` reset; `consumption.py:1314` prepared validation, `:1344` evaluator, `:1406` probes, `:1515` native target equation. Exact files/hashes and complete tracked diff are saved beside this record; line numbers are navigation aids, hashes define the accepted version.

## Fresh verification

- **1106 tests passed**,147 existing warnings,219.97s. Combined country/household/config/credit/accounting/ratio tests and all new checkpoint/intervention/response/outcome/observer/config helpers. All changed/new test files included. This is the relevant combined regression selection, not a claim of a clean entire repository test suite.
- **35 frozen replay checks passed**,9679 households; original response rtol1e−5/atol1e−7 and aggregate absolute1e−7 unchanged. e=1e−5/1e−6/1e−7 gives0.13999999991198778/0.13999999999751445/0.14000000006141156. Arrow sandbox CPU-cache sysctl warnings do not alter replay numerics.
- Ruff lint and format checks pass on **30 Python files**; whitespace checks pass.242 runtime/config files equal full-run source exactly.
- Existing current full-run evidence inspected: all12 cases complete in672.218s, no retry; combined output≈13.94GiB, approved budget respected. No additional macro run in this review.
- Graph review: detect_changes compared with default main, explicit diff against pinned a3d48e5c,13 key upstream impacts; direct planning/reset/probe/stock-map/floor callers inspected. The graph's96 matched symbols include documentation and omit untracked diff matching; direct source/test inspection covers those omissions. Dynamic deepcopy protocol and closures are not inferred unused from zero graph callers.

Commands, versions and selection in reproduction.md; pytest.log, lint.log, frozen-replay.json, scope.json, graph-review.json, graph-extra-impacts.json and verification.json are evidence. Effective config/reference YAML are saved; no optimizer, profile search, new calibration or source mutation.

## Explicit disposition of retained limitations

| Limitation | Disposition for this implementation acceptance |
|---|---|
| Historical main separate-guard split fails123/480 from cancellation | Retain FAIL in historical comparison. Candidate branch independent guard checks pass; this is a limitation of quantitative old-main attribution, not a demonstrated failing candidate implementation. No fabricated balancing block or widened tolerance. |
| Original branch seed13 complete funding mediation lacks saved native financing/property/investment/service operands | Retain incomplete historical attribution. Sign reversal mechanically traced; actual live requests were saved. New approved sensitivity captures native operands and passes cash/floor checks, but does not retroactively fill the old campaign's missing fields. |
| Legacy campaign provenance_verified=false flags despite controller pin checks; broader source provenance | Preserve raw flags and qualify claims. Exact source/config/cache/env checks and current full-run manifest verified; do not rewrite historical flags or claim independently validated source-data construction. |
| Four failed calibration endpoint profile searches and absent sampling intervals | Retain as calibration-identification/uncertainty limitations. They do not invalidate reproduced retained coefficients or authorize changing bounds, fitting again or claiming precise endpoint identification. |
| Once-drawn initialization materially shifts levels/support and model/draw responses | Reject promotion of D on this evidence; retain A. At q20 D mean goods−21.69%, support4.505bn vs A2.903bn. Fixed-effects choice not identified by the cross-section. |
| Negative/above-one/sign-changing macro responses under both policies | Economic interpretation remains INCONCLUSIVE; not silently marked empirically valid. Implementation accounting/pairing/translation tests pass, but causal/quantitative macro claims need their own scientific justification. |
| Tiny positive native income creates extreme C/Y | Preserve registered domains/counts and disclose ill-conditioning (e.g.3.73e−13 ordinary EUR income). Never treat the resulting huge ratio as an economically meaningful propensity or silently winsorize it. log(C/Y), log(C), desired C/b remain separate. |
| Multi-change main/branch contrast and SMIC repair | Do not claim isolated causal effect of the floor fix or pure income/map translation from that comparison. Unit tests and conditional replay establish narrower implementation claims. |

These are accepted, explicitly recorded limits of evidence and use. They are not hidden passes, claims that economics is solved, or new implementation tasks needed to keep this diagnostic perpetually open. There is no further implementation blocker identified in the reviewed scope.

## Acceptance and next action

**Accept the calibrated consumption implementation with current initial_state and pinned effective settings.** Implementation, validation/reporting and final technical review are complete. No further residual experiment is needed to close PLAN's implementation work. Research limitations above remain live, especially seed-sensitive macro responses and calibration/provenance uncertainty.

Commit/push/PR/merge/release actions remain separate user instructions under PLAN; none performed or authorized implicitly here. Keep a3d48e5c and the accepted-source hashes as rollback/review references. If settings/source change later, acceptance does not automatically transfer to that new state.
