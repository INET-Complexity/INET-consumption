# Increment 2 review — 2026-11-16

Reviewed the private source-table reader/linkage API and tests. 47 combined tests pass. No runtime initialisation, data-config selection or INET reader attachment exists yet. No code changed during this review.

## Findings to resolve before integration

1. **P2 — source completeness is asserted but not checked** (`hfcs_consumption_source.py:63`, `:96`, `:105`). A file with only two hard-valid source rows, marked `before_percentile_trimming` and supplied with its correct hash, loads and computes different thresholds. The hash proves byte integrity, not that the input contains the calibration pre-percentile source population. Bind the handoff to a reviewed preparation manifest including source count, ID-set fingerprint and quantile settings, and validate these before calculating thresholds. Reproduce with a truncated-source regression that expects rejection after the fix.
2. **P2 — denominator variable is not checked** (`hfcs_consumption_source.py:86`). `income_concept` distinguishes tax/interest/imputed-rent treatment but not the choice of current versus permanent income. A table declaring `income_scale_var=perm_income` is accepted. The intended source uses `disp_geom_avg_income`. Include that setting in the validated contract and registered provenance; test rejection of conflicting settings. This is independent of verifying the income concept.

No defect found in exact string-ID lookup, replacement-draw ordering, selected-wave isolation, invalid-record rejection or array-copy behaviour on return. Synthetic tests and the actual 12,035-row source roundtrip validate those operations, not the two missing provenance checks above.

## Exclusions, in counts

French 2014 source: 12,035 rows. Pre-percentile validity failures: 1,163 unique rows, comprising 506 other-data-validity-only failures, 569 finite-hard-bound-only failures and 88 failing both. Other validity includes missing/nonfinite fields, nonpositive income, invalid deflator/weights and debt-service checks; it is not synonymous with missing C/Y. The 594 other-validity failures and 657 finite-bound failures overlap by 88.

Of 10,872 pre-percentile-valid rows, the original joint percentile trim removes 1,193, leaving 9,679. Winsorisation retains those 1,193 tail households; 337 of all 10,872 valid C/Y ratios need capping. Other tail exclusions concern wealth/resource ratios and do not necessarily require changing C/Y.

Frozen model cache: 5,760 draws = 4,941 originally retained + 353 valid percentile-tail draws + 466 pre-percentile-invalid draws. Winsorisation makes 5,294 existing draws usable under the present source-validity contract; it does not repair the remaining 466. These are draw counts, including repeated source households, not survey-weighted percentages.

Evidence and executable aggregate review probes are in the vault `outputs/audits/2026-11-16-initial-cy/increment2-review.json` and `review_increment2.py`. Probes used temporary private files and changed no source microdata.

## Resolution — 2026-11-16

Both findings are fixed. The loader now requires an independent `HFCSConsumptionRegistration` carrying complete source-wave count, order-independent ID-set SHA-256, preparation-manifest SHA-256 and quantile limits. It validates row count and ID set before computing percentiles, verifies preparation-manifest metadata, and uses the registered quantile settings. The registration must be derived from the complete reviewed calibration preparation, never from the candidate handoff itself; it is the caller's trusted input, not a self-certifying file attribute.

The file must explicitly declare `income_scale_var=disp_geom_avg_income`; missing or permanent-income settings are rejected. Regression tests cover truncation despite a correct candidate hash, same-size ID substitution, wrong/missing denominator settings, manifest mismatch, order-independent IDs and registered quantile limits. Combined tests: 53 pass; Ruff lint/format and whitespace checks pass.

Fresh French-source verification accepts all 12,035 registered source rows, reproduces 10,872 hard-valid ratios and the unchanged bounds .2560414862–2.1227138285, and rejects truncated, ID-substituted and permanent-income probes. Aggregate evidence/controller: vault `outputs/audits/2026-11-16-initial-cy/increment2-fixes.json` and `verify_source_handoff_fixes.py`. The old review probes and handoff files are historical; use the registered replay for the revised API.

Invalid source records remain flagged in the table and rejected by `ratios_for`. To follow calibration in the subsequent sampler integration, exclude hard-invalid source records from sampling eligibility and draw the full requested population size from valid records with survey weights. This changes the sampled population; it must use a fresh data build and must not silently drop draws from an existing run. No runtime/default/sampler change was made in this fix.
