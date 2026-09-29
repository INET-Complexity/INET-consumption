# Total-consumption Increment 3

Base: Increment 2 `b0834637` (PR #154). Implements Increment 3 and Section 5
of the existing 2026-09-23 integration plan, with the 2026-09-28 accounting
basis decision. No calibration, settlement, financing, or trade-balancing
policy changes are included.

## Accounting boundary

`Economy.compute_gdp` still receives VAT-inclusive **goods** consumption,
actual rent R, and imputed rent H separately. It forms household FCE as
`goods + R + H`. Both raw and recorded expenditure GDP use this FCE, without
adding R again. Output and income GDP each gain H once. Existing cash-rent
terms stay in place, including private/public rental receipts on the income
side. H never enters the household income, payment, goods-order, VAT or credit
interfaces.

Three scalar economy HDF5 series explicitly expose the national-accounts-only
bridge, from initialization onwards:

- `owner_occupied_housing_output`
- `owner_occupied_housing_value_added`
- `owner_occupied_housing_operating_income`

All equal the existing aggregate imputed rent. This is the specified first
implementation: no housing intermediate consumption or housing depreciation.
`total_output`, `total_gross_value_added`, sectoral GVA, and
`total_gross_operating_surplus_and_mixed_income` retain their prior firm scope.
For presentation, combine firm output/GVA with actual housing rent plus the
respective owner-occupied bridge; combine operating surplus with the existing
private/public rental-income terms and owner-occupied operating income.
Do not relabel a firm component as the combined national-accounts total.
The existing initial/runtime firm-GVA tax convention is not changed here.

The captured Increment 0 fixture JSON remains unchanged. Its replay tests now
require each GDP result to gain H and FCE to gain R+H; all prior raw residuals
and import/export adjustments remain. Independent fixtures preserve a 37.5
expenditure gap and a 23.0 income gap with and without adjustment. Adjusted
GDP equality is not evidence that raw accounts balance.

## Initial anchor and ECM

Runtime initial accounts are reconstructed from agent data. They are not
copied from synthetic GDP, which already includes H. Initial runtime GDP
therefore gains H once, while synthetic formulas/data stay unchanged.
Initial and subsequent GDP/FCE rows now use the same basis for growth.

Household net goods observations previously summed above the separate industry
anchor. Allocate that unchanged aggregate anchor in proportion to the original
household goods shares, with explicit invalid/zero-source guards. For the
pinned FRA data the factor is **0.9962496421214523**. Initial gross goods become
254.674343640bn LCU, actual rent is 14.082326799bn, and imputed rent is
41.065648654bn; household total consumption/FCE is **309.822319094bn**.
Initial runtime GDP is **563.474205089bn**.

This is an explicit reconciliation of the starting consumption observation,
not a retroactive payment: deposits, debt, income, rents, investment, initial
goods targets and tax receipts are unchanged. `consumption` and the initial
consumption part of `realised_household_expenditure` use the aligned goods
allocation; new cash and total outcomes follow it. Industry/aggregate goods
anchors retain their values and meanings. No housing amount is added to goods.
All consumption rules share the aligned initial observation; their runtime
formulas and target semantics are unchanged.

The initial `cacf_real_consumption_budget` is C_total divided by initial CPI.
Country refreshes it after effective VAT overrides and CPI initialization.
The usual two-pass append/replace and lag behavior remains intact. Non-CACF
rules do not consume this ECM state. An initial total-target/MPC evaluation is
still unavailable rather than manufactured.

## Housing rent/value relationship boundary

PR #155 deliberately keeps the initial rent/value coefficients fixed during a
run. `HousingMarket.compute_observed_fraction_rent_value()` still records
completed rental transactions in its diagnostic histogram, but does not refit
the pricing coefficients from the current-period sample. This is a model-policy
choice, not an accounting identity: small or compositionally changing rental
samples can produce unstable slopes, including negative offered rents. The
freeze therefore protects the housing-pricing input used by household rent,
credit, and GDP flows while leaving realised rental observations available for
diagnostics. The dedicated housing-market regression test verifies that new
transactions do not change the coefficients; the total-consumption validation
also verifies that the same non-negative housing-flow convention is used by
households, initial economy accounts, and runtime economy aggregates.

## Validation and reproduction

Registered spec: vault
`wiki/experiments/specs/2026-09-28-total-consumption-increment3.md`.
Use the canonical model Python 3.12.13 environment, unchanged country config,
and the pinned Increment 1 data cache/reference files. The capture script
saves resolved configuration, package versions, source/data hashes, exact GDP
calls, reconciliation factor and HDF5 output. The verifier separately checks
fixed-input accounting and measures endogenous differences from Increment 2.

```sh
PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/private/tmp/inet-increment3-numba \
  /Users/andone/Documents/python_projects/INET-consumption/.venv/bin/python \
  docs/implementation/total-consumption-increment3/capture_smoke.py \
  --repo . --output /private/tmp/inet-increment3-smoke-t2 --t-max 2
# Proceed to t_max=5 only after the t2 verifier passes.
python docs/implementation/total-consumption-increment3/verify_smoke.py \
  --candidate /private/tmp/inet-increment3-smoke-t2 \
  --baseline /private/tmp/inet-increment2-smoke \
  --output /private/tmp/inet-increment3-smoke-t2/validation.json
```

The bounded test command covers `tests/test_macromodel/unit/test_agents/`
`test_households` and `test_economy`, `tests/test_macromodel/unit/test_country`,
both `tests/test_run_model/test_mpc_{analysis,consumption_outcomes}.py` files,
and the two simulation-order tests named in the Increment 2 validation record.
New tests cover initial anchor failures/zero cases, all initial VAT modes,
first-period GDP/FCE growth, all housing HDF5 bridges, and raw income as well
as expenditure discrepancies.

The first suite found two list/array mistakes in new test assertions (700
other tests passed); corrected tests passed. The first verifier used the wrong
existing tax dataset name; corrected to `taxes_on_products`. Neither required
production changes. Existing pandas/deprecation and markup-anchor warnings
remain outside this increment.

Paired seeds 12–16 and the temporary-transfer experiment remain Increment 4.
The smoke measures the joint dynamics of the newly initialized ECM and GDP;
it does not attribute their effects separately or establish macro validity.

## Results

**704 targeted tests pass**, including 15 new accounting/anchor cases and the
extended initial VAT/ECM integration coverage. Ruff check/format and whitespace
checks pass. Codebase-memory was rebuilt on this checkout and the GDP inbound
trace remains `Simulation.iterate -> Country.update_realised_metrics ->
Economy.compute_gdp`; those callers' runtime paths are unchanged.

Seed-15 t2 and t5 smokes pass: zero identity failures over 17,280 and 34,560
household rows respectively, including initialization. Initial financial/tax
invariants match Increment 2 exactly. `validation.json` records all five
fixed-input bridges and the overlapping-period dynamic comparison.

At t5, output GDP is 446.218563488bn LCU and H is 41.836426676bn. The raw
expenditure residual is **+0.267812842bn**; it remains visible despite adjusted
GDP equality. This unrelated residual was not repaired. Raw income residuals
are within 0.001 LCU of zero in this smoke.

At t2 relative to Increment 2, gross goods consumption rises 21.061790661bn,
cash consumption rises 21.059187178bn, total consumption rises 21.059176330bn,
and GDP rises 67.613668448bn. CPI is 1.006126499 versus 1.006547954; new
consumption credit is 2.886518852bn versus 1.956666462bn; deposits are
598.783262224bn versus 644.144637651bn. Mean gross-target formula MPC is
0.516576153 versus 0.438316336. These are joint dynamic effects, not the
fixed-input H-only accounting effect and not paired-transfer realised MPCs.

Local full artifacts and logs: `/private/tmp/inet-increment3-smoke-t2/`,
`/private/tmp/inet-increment3-smoke-t5/`, and
`/private/tmp/inet-increment3-suite-final.log`. The committed manifest pins the
production source even though its HEAD is the pre-change base during execution.
