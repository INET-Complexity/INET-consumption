# HFCS consumption initialisation

FRA defaults to `household_consumption_initialisation.method: hfcs_cy` in the data configuration. `method: legacy` retains the previous opening-consumption allocation for comparison. The wage/benefit accounting correction applies to both methods.

## Source preparation and sampling

Use the first HFCS implicate. The French model input values were compared across all five implicates; the relevant substantive fields agree. Preserve original `SA0010` household IDs as strings, including leading zeros. Generated calibration `hh_id` values require the independently verified crosswalk; they are not source IDs.

The private calibration handoff supplies total C/Y, survey weights, the pre-percentile hard-valid mask and the threshold pool. Its denominator is current `disp_geom_avg_income`, net of borrowing interest plus owner-occupied imputed rent. C/Y is not income-adjusted. Cap C/Y at the calibration's weighted 1st/99th percentiles before sampling, using cumulative-weight midpoints. Percentile tails remain eligible; hard-invalid records are excluded before full-size weighted sampling.

The handoff and preparation registration are independently hash-pinned in the data configuration. Validate source count, ID-set fingerprint, preparation-manifest hash, income/consumption metadata and weights. French source counts are 12,035 complete and 10,872 hard-valid households. Private Parquet/registration files are excluded from Git and must be installed at the configured paths. `macro_data.util.consumption_source_export` validates the prepared input and produces the registration/config fields. Legacy mode does not require this handoff.

## Opening allocation

Country initialisation supplies raw aligned model income before the planning floor and smoothing:

`Y = sqrt(max(N, 0) * max(T, 0)) - borrower_interest + imputed_rent`,

where N is wages plus transfers and T includes property receipts. Use model-period flows and the same currency/population scale throughout. Let r be winsorised source C/Y, H cash plus imputed rent, tau the effective VAT rate, G0 existing legacy opening net goods, and Gbar aggregate net goods.

1. For finite inputs, classify a household as infeasible when `Y <= 0` or `r*Y < H`. Retain its exact G0.
2. Reserve the sum of those retained goods from Gbar.
3. For feasible households compute `(r*Y-H)/(1+tau)` and scale these goods proportionally to absorb the remaining aggregate.

Final total consumption is `(1+tau)*goods + H`. The common sector basket preserves both household goods totals and aggregate sector totals. The goods anchor already includes housing-input goods; do not add them again. Income, financing, assets, activity, properties, housing flows and the population are unchanged. No support receipt, income floor, donor substitution or population rebuild is introduced.

`allocate_hfcs_with_legacy_reserve` implements the reserved budget. Malformed/nonfinite inputs, an overcommitted reserve, and a remainder without feasible positive goods shares remain errors before any budget writes. `allocate_hfcs_ratio_consumption` remains strict; `initialise_hfcs_consumption(..., legacy_fallback=False)` provides strict runtime diagnostics.

Reconciliation generally changes final C/Y away from the source ratios. `candidate_total` retains the raw HFCS target, including infeasible values; fallback candidate goods use G0. The goods scale applies only to feasible households. Existing final-C/Y diagnostics are NaN for nonpositive raw income, where the ratio is undefined.

Opening goods, cash/total expenditure, the real CACF budget, target matrix and expenditure diagnostics are updated consistently. Reset restores the resulting opening rows. Dynamic preferences and consumption-rule coefficients are unchanged.

## Income and pension accounting

The synthetic household `Employee Income` field historically combines wages and unemployment receipts. At the Country boundary, seed employee-income components from individual wage-only receipts, then include unemployment benefits once in transfers. Preserve cached total monetary income and financing income.

Observed PG0310 individual public-pension amounts determine recipient shares within household pension allocations; PG0410 private-pension source amounts are also retained. Public pension receipt is separate from labour status. Preserve reported retirees and additionally classify observed public-pension recipients aged 65+ as retired/inactive using the existing model cutoff. Retain original HFCS labour status; younger recipients are not automatically retired, and existing labour calibration otherwise applies. Retirement cannot be reversed by participation matching. This is a model assumption, not a determination of statutory eligibility.

## Caches

Registered source/config/extraction signatures and the pension population schema invalidate incompatible data caches. Back up existing caches before replacement, including forced rebuilds. Sampling-only caches cannot enter completed HFCS initialisation.

The reserve policy changes runtime construction without changing the stored population. Notebook benchmark signatures include `opening_income_schema: separate_wages_unemployment_v1` and `opening_consumption_schema: hfcs_reserved_legacy_v1` so cached benchmark outputs rebuild.

## Validation and limits

On the saved FRA seed-38 population of 5,760 households, 54 retain legacy budgets (0.9375% of households), reserving 0.434968% of aggregate goods. The other 5,706 receive HFCS allocations; their goods scale is 1.5631940515. The retained households comprise 12 zero-income renters, 5 negative-income renters, 7 zero-nonproperty-income owners, and 30 other positive-income housing shortfalls.

Same-population checks verify exact retained goods, matching retained cash/total budgets, unchanged income/assets/activity/property states, remaining-budget allocation, and aggregate/sector reconciliation. Matched legacy versus HFCS-reserve simulations for seeds 1–2, four quarters each, have finite core series and exact resets. These are numerical/accounting checks, not long-horizon economic validation. HFCS-reserve goods consumption is about 12% below matched legacy at quarter four; this is an overall initialisation effect, not an isolated estimate of the small fallback's effect.

The fallback preserves an existing starting condition; it does not resolve the 54 households' underlying income/funding issues. Minimum-income/support redesign remains separate. The abandoned donor-rebuild experiment was removed from the shipped code; its audit evidence is retained in the research vault. Detailed validation controllers, aggregate results and source/code manifests are in the vault's dated initialisation audits. No private microdata or simulation output is committed here.
