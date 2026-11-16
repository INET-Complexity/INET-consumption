# HFCS-ratio consumption initialisation

Branch: `feat/hfcs-cy-initialisation`, based on main `4ecf10b0`.

## Increment 1: allocation arithmetic

Implemented `macro_data.util.consumption_initialisation.allocate_hfcs_ratio_consumption`:

1. Apply each household's source total C/Y to comparable, unfloored model-period income.
2. Subtract cash and imputed rent, then divide by `1 + effective VAT` once.
3. Scale goods proportionally to the national-accounts net goods aggregate.
4. Rebuild total purchase-price consumption, leaving housing flows unchanged.

The result contains both candidate and reconciled allocations and the scaling factor. Post-reconciliation C/Y need not equal the source ratio. The aggregate here includes housing-input goods: downstream code must not add that allocation again. Inputs are not mutated, no cash transaction is booked and no saving-rate multiplier is applied.

Budgets below housing expenditure and nonpositive comparable incomes fail explicitly. No legacy fallback or new support rule is selected by this arithmetic helper.

User decision: winsorise instead of percentile trimming. `winsorise_hfcs_consumption_ratios` takes the calibration pre-percentile hard-validity mask and the source survey weights, calculates p01/p99 bounds with the calibration midpoint-interpolation convention, and caps C/Y for every hard-valid source row. The optional threshold pool follows the registered estimation waves. Households in other-variable percentile tails remain eligible. Invalid/hard-bound failures retain their row positions as NaN with valid=False; winsorisation does not repair data faults. Calculate thresholds before model sampling; never survey-weight the sampled draws a second time.

Validation: 36 focused tests pass; Ruff lint and formatting pass. Tests cover source-ratio recovery, national-accounts reconciliation, VAT/housing accounting, period/population scaling and invalid/infeasible observations.

## Pending integration

The data-config selector and new default are not wired yet. Existing model initialisation remains unchanged in this arithmetic-only increment:

- Percentile-tail handling is resolved: retain hard-valid source households and winsorise their C/Y. Wire the source preparation mask and threshold pool before sampling. Hard-invalid source observations still require explicit treatment at integration.
- Decide treatment of candidate total budgets below model housing expenditure or nonpositive comparable income; never silently impose a new support/floor policy.
- Carry source ratios using a stable household linkage and an explicit preparation provenance; do not substitute the legacy DOCOGOODP goods-spending share for the notebook's total C/Y.
- Connect the selected method through cache construction and runtime initial alignment, accounting for effective VAT, housing-input goods and stale-cache invalidation. Retain the present path as `legacy` and make the completed new path the default.

No simulation, refit, runtime consumption-rule change or minimum-income change is included in increment 1.

## Review of increment 1 — 2026-11-16

No blocking correctness defect found in the isolated arithmetic/winsorisation helpers after review. Checked source-ratio application, weighted percentile convention, source-row alignment, validity versus tail handling, unchanged housing flows, single VAT conversion, aggregate reconciliation and rejection of infeasible inputs. Tests exercise the composition of winsorisation and allocation, including the expected change in final C/Y after aggregate reconciliation.

Scope limits are material: callers must supply the actual calibration hard-validity mask, source C/Y and comparable model income. This increment does not recreate that preparation, attach source households, change the data configuration, invalidate old caches or replace runtime alignment. In particular it must not be described as a completed new default. Negative residual goods and nonpositive comparable incomes still fail explicitly rather than being silently clipped or floored. Housing-input goods are included in the helper aggregate and must not be added again by downstream integration. Mutable NumPy result arrays are caller-owned; the frozen dataclasses prevent field rebinding, not array mutation.

Actual-data review against the registered French 2014 preparation reproduces the calibration C/Y cutoffs to 1e-13: lower .2560414862050745, upper 2.122713828510414. Of 12,035 source rows, 10,872 are hard-valid (versus 9,679 after the historical joint percentile trim); 337 valid C/Y values are capped and 1,163 invalid rows remain flagged. Only aggregate evidence is saved in the vault at `outputs/audits/2026-11-16-initial-cy/winsorisation-review.json` with a replay controller. No household microdata are committed.

## Increment 2: validated source table and linkage

Implemented `macro_data.util.hfcs_consumption_source.read_hfcs_consumption_source` and `HFCSConsumptionSource.ratios_for`.

The private Parquet handoff contains `source_id` (stable string), `wave` (integer), `c_y`, `survey_weight`, `hard_valid`, and `threshold_pool`. Pandas/Parquet attributes must state `consumption_concept=total`, `income_concept=net_of_interest_plus_imputed_rent`, and `preparation_stage=before_percentile_trimming`. The caller supplies the expected SHA-256. Reading and hashing use identical file bytes. Each selected wave has unique source identifiers. Quantiles are computed on source survey weights before any model draws. Hard-invalid rows are preserved as flags, not repaired.

`ratios_for` links stable IDs in requested draw order and supports replacement draws. It rejects unknown IDs and hard-invalid records explicitly, preserves leading zeros, and returns a copy. It does not apply survey weights to draws. This is a linkage API, not yet an attachment to the INET reader's raw IDs.

Validation: 47 tests across increments 1–2 pass; Ruff lint/format and whitespace checks pass. A private handoff exported from the real French calibration preparation round-trips all 12,035 source rows, reproduces the 10,872 valid ratios and 337 caps, and passes reordered/repeated-source-ID checks. Source microdata are stored only at `/private/tmp/inet-hfcs-cy-initialisation-source.parquet`; aggregate evidence is in the vault `outputs/audits/2026-11-16-initial-cy/source-handoff-review.json`.

Still pending: wire source IDs into the actual HFCS reader/sampler, choose explicit hard-invalid/model-infeasible handling, expose the data-config methods/default, apply allocation after comparable income and effective VAT are known, and handle stale caches. Existing runtime/default remains unchanged. No simulation, fit, support-rule change, commit or push.

## Increment-2 review fixes

The source loader now additionally requires `registration=HFCSConsumptionRegistration(...)`, independently registered from complete calibration preparation. Required fields are `source_count`, `source_ids_sha256`, `preparation_manifest_sha256`, and lower/upper quantiles (default p01/p99). The preparation-manifest hash and explicit `income_scale_var=disp_geom_avg_income` must be present in Parquet attributes. Candidate file hash alone is insufficient. See `REVIEW_INCREMENT2.md` for resolution and fresh actual-data verification; 53 tests pass. Historical three-argument loader examples need the new registration and updated metadata.

## Increment 3: reader identity and sampling

The HFCS reader now preserves original `SA0010` as `HFCS Source ID` (as text, including leading zeros); the household file is authoritative when joining derived data. The sampler and `SyntheticHFCSPopulation.from_readers` accept an optional loaded `consumption_source`. Without it, the existing weighted sampling path is unchanged.

With a source, the sampler checks the complete source-ID set and valid-row survey weights against the registered handoff, rejects eligible households without linked individuals, excludes hard-invalid records before drawing, and draws the requested household count using the existing survey-weight probabilities once. Each replacement draw carries `HFCS Source ID` and `HFCS Consumption Ratio`. Population restriction preserves those diagnostics. The reader-level constructor checks the source wave and guards against later loss of requested household rows.

Identity discovery: the calibration parquet's `hh_id` is a generated row ID, not raw `SA0010`. Initial direct linkage correctly failed the ID-set check. The real-data handoff was translated through the prior audited raw/parquet mapping, rechecking all source weights and finite reported-goods values before registering the original IDs. This translation is in the aggregate replay controller; never relabel calibration IDs as raw IDs without this verified mapping. The generated-ID source files from earlier increments remain historical artifacts. For this verified extraction, use the private reader-ID source `/private/tmp/inet-hfcs-cy-reader-source-registered.parquet`.

Validation: 66 focused tests and 36 existing population tests pass; Ruff lint/format and whitespace checks pass. The real French single-implicate reader supplies 12,035 households; 10,872 are eligible and 1,163 invalid rows are excluded before a complete 5,760-household draw. There are 12,942 sampled individuals and 3,957 distinct source households in the fixed-seed smoke. All sampled ratios and household–individual links reconcile. Evidence/controller: vault `outputs/audits/2026-11-16-initial-cy/increment3-sampling.json` and `verify_sampling_increment.py`. This is data sampling, not a model simulation or economic validation.

Current scope: APIs are integrated at the reader/population level but no country builder passes `consumption_source` yet. The data-config selector/default, cache invalidation and consumption allocation remain pending. The new path requires exactly the registered source extraction; pooling other implicates without corresponding registered preparation fails explicitly. No live configuration, support rule, parameter fit, commit or push changed.

## Items 1 and 3: configuration/source wiring and caches — 2026-11-16

Implemented `HouseholdConsumptionInitialisationConfiguration`, country source loading, explicit extraction overrides and data-cache signatures. FRA's data configuration now selects `method: hfcs_cy` by default; change that single value to `legacy` for the old allocation. `hfcs_implicate: 1` applies consistently to both methods. Countries without an explicit selection retain their previous reader settings. Registered source and registration paths are relative to RAW_DATA_PATH, with both file SHA-256 values pinned. New source wiring currently supports EU survey countries; an attempted proxy-country HFCS configuration raises explicitly.

`export_consumption_source` exports and validates the calibration's complete pre-percentile table and independent registration without fitting or imputing data. Its input must use verified original source IDs; the earlier audited translation remains the source of that crosswalk. Private exported FRA inputs are installed in the worktree's ignored `run_model/data/raw_data/hfcs/consumption_initialisation/` directory. No household microdata are tracked. The portable config uses relative paths, not /private/tmp paths. Other raw datasets are not copied into the worktree.

Comparison of all five French implicates: the household and personal substantive values agree; only IDs/imputation labels differ. Derived records additionally differ in nine euro-area ranking/top-decile columns, none of which is mapped by INET's HFCS reader. Thus the files are not byte-identical, but the inputs used here are identical. The new extraction uses implicate 1. The comparison is France-specific; it makes no claim about other countries. Full aggregate comparison/controller: vault `outputs/audits/2026-11-16-initial-cy/implicate-comparison.json` and `compare_implicates.py`.

DataWrapper loads verified sources before expensive readers, requests a single implicate per configured country, passes source ratios into EU population construction, and records the source hash and sampling-only stage. Caches store a signature covering configuration, extraction, source/registration content, historical-reader flag and implementation stage. Ordinary and benchmark notebook loaders detect missing/mismatched signatures; benchmark dataframe signatures include the private-source fingerprint too. Source hash mismatches reject stale input rather than silently rebuilding from changed unregistered files. Both method-switch directions invalidate incompatible caches. Existing pickles are backed up with content-addressed filenames before replacement, including explicit forced rebuilds. Other supported command-line runners always build fresh DataWrapper data.

The new data remains marked `hfcs_sampling_only_v1`. Country construction refuses to run it with legacy consumption allocation. This guard is intentional while item 2 is deferred; select `legacy` to simulate now. When opening accounts are implemented, replace the stage version so sampling-only caches rebuild. Legacy first-implicate caches carry their own signature and do not pass as old unmarked five-implicate caches.

Validation performed here is focused: config/source/extraction checks; source and registration tampering; old/missing/stage cache metadata; both-direction method switching; pickle backup; benchmark source fingerprints; constructor rejection of partial data. The real FRA config resolves 12,035 registered rows and 10,872 eligible rows. Focused suites passed (105 tests before the final backup case, with directly affected selections rechecked); 12 existing DataWrapper tests also passed. No new-method full data build, Country end-to-end validation, model simulation, or refit was run. Item 4 is explicitly deferred by the user. Item 2 (actual opening consumption) remains the next implementation step.

## Opening allocation implemented; final acceptance blocked — 2026-11-16

The runtime hook `initialise_hfcs_consumption` now runs after Country setup establishes opening aligned income and effective VAT. It uses `expected_behavioural_income_raw` (before the planning floor), the source ratio and model rents, then reconciles net goods to the existing aggregate. It replaces opening goods, cash/total expenditure, real CACF budget, target matrix and expenditure diagnostics without changing income or balances. The target matrix uses a common opening sector basket to preserve sector column totals and new household row totals simultaneously; dynamic consumption preferences are untouched. Candidate/final ratios, income and scale remain explicit diagnostics. Legacy initialisation is unchanged. Source IDs are excluded from the numeric household input cast.

Data/cache stage is now `hfcs_opening_allocation_v1`; old sampling-only inputs remain rejected and the new cache signature forces rebuilds. Later reset restores the final opening TimeSeries rows. No model-specific income floor or housing clipping has been added to the strict allocation helper.

Validation: full new/legacy FRA data builds (seed38, first implicate,5760 households) completed. Legacy seed1/1-quarter smoke has finite core household variables and passes reset-budget identity. New construction **fails its feasibility gate**:33 households have nonpositive raw comparable model income;112 have implied totals below housing, including all33.30/33 have zero non-property income. These are model-reconstructed states, despite source HFCS validity.

79 allocation/config/notebook tests,70 income/household integration tests and261 country/household regressions pass;lint/whitespace pass. Validation spec requires constructor gates before the paired seeds1–2/four-quarter comparison, so that campaign was not launched. A user decision is pending: investigate the income/housing construction, allow explicit zero-goods/housing accounting boundaries, or authorise a flagged legacy fallback. None is selected automatically. This is an implemented strict rule with a failed real-state validation gate, not a completed/accepted default. Aggregate evidence/spec/manifests are in the vault; private data builds remain under /private/tmp/inet-hfcs-final-validation.


## Pension eligibility correction — 2026-11-16

Positive observed household public-pension receipts now preserve household pension shares, including households without retired members. Retired members carry the share where present; otherwise adults age>=16, then all members for minors-only households. This carrier convention does not identify individual source beneficiaries. Retirement status and the aggregate pension envelope are unchanged. Runtime planning and settlement use positive pension weights for eligibility.

Recipient-schema markers and cache signatures require both methods to rebuild stale pension inputs; direct stale population loading is rejected. Strict consumption feasibility checks remain unchanged.

Controlled replay of the same saved population, changing pension allocation and recomputing income only: nonpositive incomes33→16 (9 zero,7 negative); positive-income housing shortfalls79→45; total infeasible112→61. Evidence: outputs/audits/2026-11-16-infeasible-income-housing/pension-replay.json and replay_pension.py. This is not a fresh full data build or simulated period. The remaining61 block acceptance; no floor, clipping or fallback was added.

Validation:170 population/country/notebook regression tests pass; final13 pension/config tests pass (overlapping selection, with new runtime nonretired-recipient regression). Focused Ruff and whitespace pass. No commit/push. Next investigate remaining income/participation/housing constraints before rebuilding and completing paired dynamic validation.


## Observed individual pensions and retirement assumption implemented — 2026-11-16

Supersedes the provisional household-carrier allocation above. The reader now preserves PG0310/PG0410 as individual annual gross pension source amounts, converts currency, and carries them through population processing and proxy-currency conversion. Public-pension allocation preserves household source shares and uses observed individual amounts for shares within each household. On actual French inputs, individual and household amounts reconcile, so this equals global individual-source weighting. Positive household receipts without observed individual recipients fail explicitly; missing/negative/nonfinite individual-source data are not assigned to proxy household members (structural missing receipts are zero). The pre-existing household allocator nonnegative-source convention is retained; anonymised test household amounts include negative values and do not exactly match personal magnitudes, so no new absolute-magnitude reconciliation gate is imposed on that fixture. Real-source reconciliation was checked separately.

Retirement assumption applies to PUBLIC PENSION RECIPIENTS: retain reported retirees (HFCS status5, including early retirees), additionally mark recipients age>=65 retired/inactive using the existing model cutoff. This is a model assumption, not a newly determined statutory age. Keep HFCS Labour Status unchanged. Pension receipt below65 does not itself alter activity status; existing macro participation/unemployment calibration still applies to others. Retirees cannot be reactivated by participation calibration. Those above65 without observed public pension receipts are not automatically reclassified by this change.

Both consumption-method cache stages and notebook population/benchmark schemas are versioned v3; direct old pension caches require rebuild. Shared income/initialisation callers reviewed with impact analysis. No consumption clipping, support floor, legacy fallback, calibration refit, commit or push.

Fresh FRA first-implicate builds completed for both methods (seed38,5760 households). HFCS C/Y:3246 public pension recipients;49 additional retirees by age;295 younger nonretired recipients;0 retired members active. Legacy:3220 recipients;59 additional retirees;251 younger nonretired recipients;0 retired members active. Individual shares, household component payments and final allocator equivalence with saved fresh builds pass. Legacy seed1 one-quarter smoke has finite core series and reset-budget identity passes.

New constructor still FAILS feasibility:17 nonpositive raw incomes,53 candidate totals below housing (all17 included); no nonfinite income. This is a fresh build with changed labour initialisation, not a causal extension of the earlier same-population pension replay (61). Dynamic new/legacy acceptance remains pending. Final tests:86 pension/population/config/notebook tests plus109 country/opening-allocation tests pass; focused lint/format/whitespace pass. Initial broader runs caught household negative-value fixture compatibility; final runs resolve it without changing fixture data. Evidence/controllers are under outputs/audits/2026-11-16-infeasible-income-housing/; private full caches retain the data-individualpensions-* names.
