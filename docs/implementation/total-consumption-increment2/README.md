# Total-consumption Increment 2

Base: `89c9e118`, Increment 1, PR #153. This increment records realised
consumption concepts and adds MPC analysis. GDP expressions, household cash
settlement, goods orders, the ECM state, and existing aggregate meanings stay
unchanged. The initial household/aggregate basis choice remains Increment 3.

## Outputs

The existing allocation of goods spending between consumption and investment
supplies net goods `consumption`. Each realised period appends exactly once:

- `consumption_cash_expenditure = (1 + VAT) * consumption + max(rent, 0)`
- `consumption_including_housing = consumption_cash_expenditure + max(rent_imputed, 0)`

Both are household vectors of nominal local currency per model period. HDF5
uses the normal time-series writer; household indexing matches `consumption`.
There is no runtime household-removal mechanism to extend. Investment and its
taxes are excluded. Imputed rent introduces no payment or balance-sheet entry.
Invalid housing-flow shapes or non-finite observations fail; negative flows
retain the existing CACF zero normalisation.

Initial values use existing household consumption and rents. The existing
country VAT-override reconciliation also refreshes the two new initial rows.
This is necessary because the active FRA config overrides the data VAT rate
at that later boundary. It does not reconcile household sums to the separately
seeded aggregate goods series. Unevaluated/non-CACF calibrated totals use NaN;
legacy `formula_implied_mpc` values and all feasible-goods columns are preserved.

## MPC columns and old files

`build_household_mpc_panel` retains its existing columns and denominators.
New prefixes `cash_`, `total_`, and `behavioural_total_target_` identify realised
cash, realised total, and gross CACF total-target responses respectively.
Examples: `cash_mpc_impact`, `total_real_mpc_impact`,
`behavioural_total_target_cmpc_4q`. Cumulative names prefix the supplied legacy
cumulative column names; intermediate horizons follow the existing `2p/3p/4q`
convention. The existing default `cmpc_4q` name remains caller-controlled even
when a different horizon is requested, for compatibility.

`baseline_target_consumption_total_mpc` and its shock counterpart expose the
saved formula derivative of gross T. These differ from the paired response
of T: the latter can incorporate all endogenous changes in a simulation.
The legacy formula derivative remains VAT-exclusive full-target units.

`imputed_rent_mpc_impact`, `imputed_rent_real_mpc_impact` and corresponding
cumulative columns expose the housing bridge. Unchanged nominal H implies
identical nominal cash/total MPCs, including goods floors and rationing. Real
responses use each arm's CPI and the existing real shock denominator; their
difference equals `(H_shock/P_shock - H_base/P_base) / real_shock_amount`.

Persisted new datasets take precedence. Old files can reconstruct gross goods
using the saved `total_consumption / total_consumption_before_vat` ratio for
each row. Missing rent, missing VAT evidence, or an unidentified factor (zero
aggregate net goods) produces NaN. Missing imputed rent never means zero;
cash can remain available when total is unavailable. Wrong household shapes
and non-finite saved housing observations fail explicitly. Old CACF zero
placeholders are unavailable unless the saved derivative identifies an
evaluated zero target. A positive target does not require a finite derivative.

## Verification and reproduction

**659 targeted tests pass, including 31 new cases; Ruff and whitespace checks pass.**

Tests cover initial values including VAT overrides, both consumption rules,
renters/owners/social renters/no flow/tenure changes, investment and rationing,
negative and malformed flows, both diagnostic passes, HDF5 shapes and values,
legacy fallback and columns, nominal and own-CPI MPC identities, and custom
cumulative names. `validation.json` records the registered FRA seed-15
quarterly two-period smoke and source hashes.

Use the canonical model `.venv` (Python 3.12.13), the Increment 1 cached data,
and the unchanged FRA config:

```sh
python docs/implementation/total-consumption-increment0/capture_baseline.py \
  --repo /private/tmp/inet-total-consumption-increment2 \
  --output /private/tmp/inet-increment2-smoke
python docs/implementation/total-consumption-increment2/verify_outputs.py \
  --baseline /private/tmp/inet-increment1-smoke-final/baseline.h5 \
  --candidate /private/tmp/inet-increment2-smoke/baseline.h5 \
  --output docs/implementation/total-consumption-increment2/validation.json
```

The first local Python 3.11 attempt lacked PyTables in broader simulation tests
and differed dynamically from the pinned Python 3.12 baseline. It also exposed
the initial VAT-override omission, fixed and covered before final verification.
Only the final pinned-environment run is used for compatibility claims.
This two-period check does not establish macroeconomic validity.

The bounded suite uses `pytest` on the household and country unit-test folders,
`test_total_consumption_increment0.py`, both simulation-order tests
(`test_iterate_runs_credit_and_feasibility_before_single_labour_clear` and
`test_iterate_skips_supply_capped_assignment_without_credit_supply_transient`),
`test_mpc_analysis.py`, and `test_mpc_consumption_outcomes.py`.
GitNexus was refreshed (9,824 nodes, 19,157 edges); the high aggregate impact
is confined to the expected shared initialisation/diagnostic paths. AST review
confirms that existing readers, GDP formulas, and other model functions are untouched.
