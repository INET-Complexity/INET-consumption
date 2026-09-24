# Total-consumption integration — Increment 0

Date: 2026-09-24. Scope: source/data gates and pre-change fixtures only.
Production behaviour is unchanged. Increment 1 has not started.

## Versions and reproduction

- Branch: `pr/fix-imputed-rent`.
- Opening model commit: `16506c427e73cb89a12b0de96683627cb82951f1`.
- Captured model baseline: `d752cf9fc349b8ef7e282f63e185c029446caf98`.
  The branch advanced during this session through `59dc31e3` (notebook)
  and `d752cf9f` (the existing planning-snapshot regression). No production
  source changed between those commits; those changes were preserved.
- Calibration: `b82aab3f8312d9ce7fe897096034362552b7c298`, with pre-existing
  working-tree changes. `calibration_provenance.json` pins the actual notebook,
  source, SNA configuration, and raw household files by SHA-256.
- GitNexus CLI initially reported both indexes current. After the branch
  advanced, the model index was refreshed at `d752cf9f` (9,762 nodes,
  18,998 edges). Cached MCP line offsets are not authoritative. Codebase-memory
  graph discovery was also used; current source was checked directly.
- `baseline_manifest.json` records the cached population SHA, source hashes,
  exact GDP calls, initial accounts and HDF5 hash. `resolved_configuration.json`
  records the effective model configuration; `runtime_reference_hashes.json`
  pins supplementary SMIC, learning, forecast and portfolio files.
- Python 3.12.13; package versions are in `packages.txt`.

Run from the model root, with the original cache/config and referenced data:

```sh
NUMBA_CACHE_DIR=/tmp/inet-increment0-numba PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python docs/implementation/total-consumption-increment0/capture_baseline.py \
  --repo . --output /tmp/inet-increment0-replay
.venv/bin/python docs/implementation/total-consumption-increment0/verify_calibration.py \
  --calibration-repo ../consumption-notebooks --output /tmp/calibration-provenance.json
```

The baseline is FRA, seed 15, quarterly, `t_max=2`, 5,760 households,
scale 5,000, existing `data_benchmark.pkl`, current resolved country YAML,
no notebook preset or single-run CLI overrides. This is an accounting capture,
not a calibration rebuild or macro validation. The original HDF5 is retained
locally in the vault output directory referenced by the Increment-0 report;
only aggregate fixtures/manifests are committed here.

## 1. Persisted calibration provenance — closed

The canonical calibration notebook reads `data/hfcs/mpc_dataset_FRA.parquet`,
SHA-256 `90fb6273ca7e088a812152243038477c8a2175005bf7f74810792937d7c31b6b`.
Its producer is `show_distributions_and_MPCs.ipynb`, cell 6, calling
`build_wave_df` for 2014/2017/2021 and concatenating household medians across
five implicates. The verification script reconstructs consumption from the
local raw HFCS reader, independently checks row order against `HB0900`
(residence value) and `3*HB2300` (quarterly renter payments), and compares every
row. `HB0900` is not interchangeable with the ownership-adjusted `DA1110`.

| Wave | Rows | Max nominal reconstruction difference, EUR/year | HFCE deflator |
|---|---:|---:|---:|
| 2014 | 12,035 | 0 | 95.28575 |
| 2017 | 13,685 | 0 | 96.47150 |
| 2021 | 10,253 | 0 | 101.33525 |

`build_consumption_components_from_hfcs` constructs annual non-durables from
`12*(HI0220+HB2300+HB0410)` and adds reconstructed imputed rent, allocated by
`DA1110`; it adds reconstructed durables including annualised leasing. Thus
actual rent (including partial-owner rent) and imputed rent are already in the
persisted total. This verifies the consumption column, not historical execution
metadata for every one of the parquet's 93 columns.

Real consumption is nominal consumption divided by `hfce_deflator/100`.
Stored float32 values pass `rtol=2e-6, atol=0.02`; maximum absolute rounding
errors are 0.02438, 0.01608 and 0.01323 EUR/year respectively (the combined
relative/absolute tolerance passes). `prepare_household_frame` recomputes this
from nominal once, with CPI only a missing-HFCE fallback. There is no second
annual-to-quarter conversion in the calibration frame. At runtime CACF
`_evaluate_target` annualises period income by `12/time_unit` only for
stock/income ratios; the consumption level stays at model-period frequency.
Runtime HFCS monthly rents are multiplied by `scale*time_unit`; annual cash
flows by `scale/(12/time_unit)`. Stocks are not divided by four.

Decision: retain the annual, purchase-price total-consumption calibration.
Subtract current-pass actual and imputed rents from T in Increment 1; never
add imputed rent to disposable cash income or financing uses.

## 2. VAT and ledger mapping — decision closed, fix belongs to Increment 1

Verified source boundaries:

- `consumption.py::CreditAugmentedConsumption.compute_target_consumption`
  divides desired gross goods by `1+vat` before constructing the goods matrix.
- `Households.compute_expected_income` and `compute_income` sum employment,
  social transfers, net rental income and the configured financial/dividend
  income. Neither subtracts consumption VAT. Income tax/social contributions
  are separate; they do not make this a net-of-consumption-VAT income ledger.
- `update_consumption_and_investment` allocates net goods purchases between
  consumption and investment. `total_consumption` re-grosses the consumption
  allocation; it does not debit deposits.
- `update_wealth` subtracts net `nominal_amount_spent_in_lcu` and cash rent.
  It separately charges investment VAT/capital tax on the resolver path, but
  never charges consumption VAT.
- `CentralGovernment.compute_taxes` already credits `vat*consumption` to
  `taxes_vat`. Crediting it again would double government revenue.

There is therefore a missing household consumption-VAT debit, not an
algebraically equivalent net-income ledger. For net goods q, the required
consumption cash use is `(1+vat)*q+R`. Example: q=100, VAT=.2, R=30 requires
150; current wealth settlement charges 130 while government receives 20 VAT.

Increment-1 decision: thread the existing VAT rate to household cash settlement
and debit `vat*realised_consumption` once, outside investment. Supply gross
consumption amounts to liquidity/borrow-sell/legacy credit adapters while
leaving rent a separate single addition. Retain net matrices for goods orders.
Keep the current legacy financing policy, but charge the same consumption VAT
on both resolver settings. Never gross up the whole purchase matrix: it also
contains investment with its own tax rate. Reuse government tax booking.

## 3. Rent priority, support and fixed-grant refresh — decision closed

The two target passes and already-booked credit are real, not hypothetical:
`_set_household_target_demand(False)` precedes housing/credit;
`reconcile_post_grant_feasible_plan` reserves liquidation and books consumer
loans; post-labour `_set_household_target_demand(True)` recomputes diagnostics
but leaves the settled carrier's residual stale. Payment settlement then reads
that residual. Do not call reconciliation again.

Smallest refresh: add a pure `HouseholdFinancialFeasibility` operation using
`dataclasses.replace` on the existing `PostGrantFeasiblePlan`, called after
post-labour targets and before `settle_authoritative_household_payments`.
Use current expected cash-income components from that pass (not the prior
realised `income` row), final rent, gross desired goods, opening service
snapshots, and separately identified investment/property obligations.
The wealth ledger is still pending: do not count a mortgage/service flow twice
merely because its credit-market book has already been updated.

For the same ledger boundary, define cash need D = positive part of uses minus
income; available opening liquid balance L, fixed granted credit B, committed
executable liquidation Q. Reuse existing liquid-first arithmetic:
`F=min(D,max(L,0))`, `residual_after_lfa=max(D-F,0)`, and
`residual_after_grant=max(D-F-B-Q,0)`. Do not subtract F again from deposits.
Mortgage proceeds and property payments must be netted exactly once when
computing available resources/uses, not folded into B (consumer credit).
Preserve B, bank allocation, debt/asset-booking fields, credit-rationing
history, Q and its reservation/base, and any settled-liquidation fields.
Update only funding-use and residual fields. A lower need does not cancel
loans or undo committed sales. A higher need does not authorise extra sales.

Existing mandatory-payment policy:

1. `settle_consumption_floor` cuts discretionary goods to the subsistence
   floor and exposes the remaining shortfall, including rent-driven gaps.
2. `CreditMarket.settle_consumer_payments` makes consumer service junior:
   `unpaid=min(shortfall, consumer_due)` and records arrears/distress.
   Mortgages were processed earlier; mortgage-suspension diagnostics do not
   themselves reverse a payment.
3. `prepare_goods_market_clearing` sends the carrier's remaining shortfall to
   government support; `settle_household_social_transfers` adds that support
   to cash income, and government records the settled amount as expenditure.
   There is no separate rent-arrears mechanism. Without consumer service,
   zero goods, zero resources and R=30, the existing policy's support is 30.
   Preserve the existing support/consumer-arrears priority, including its
   treatment of the pre-support shortfall; do not invent an overdraft.

Two directly relevant unit/priority defects must be covered by Increment 1:

- The shortfall is nominal, and the SMIC floor is already CPI-indexed nominal
  (`_compute_subsistence_consumption_from_units`). Yet
  `compute_realised_stage5_subsistence_support` multiplies support by CPI again.
  Keep this carrier in nominal cash units through settlement; remove the
  extra inflation at this existing boundary, with CPI != 1 coverage. Do not
  introduce a new benefit programme or alter eligibility/floor calibration.
- `populate_post_grant_early_repayment_capacity` subtracts net goods and debt
  service but omits cash rent and investment. Use the same gross cash uses
  and mandatory-rent reservation before calculating optional early repayment.

Run floor arithmetic in gross consumption-cash units, then divide executable
goods by `1+vat` for orders. Preserve the existing policy that the subsistence
floor protects goods consumption; do not subtract H from that floor implicitly.
The resolver-disabled path has no corresponding support/settled-carrier
contract. Unsupported mandatory-payment funding must fail explicitly before
orders/payment, rather than be labelled financed. Non-finite or malformed
funding inputs must not be converted into a successful zero shortfall.

## 4. Initial accounts and ECM seed — one basis choice pending

`SyntheticCountry.gdp_output`, `gdp_income`, and `gdp_expenditure` already add
imputed rent. For the pinned population they all equal 546.625690884bn LCU;
H0=41.065648654bn. `create_economy_timeseries` independently reconstructs
runtime accounts from agents and includes cash rent but omits H0 from every
GDP approach; it does not copy synthetic GDP. Runtime initial GDP is
522.408556435bn after runtime configuration/tax changes. The difference from
synthetic GDP is not simply H0, so do not use that difference as a housing test.

Initial runtime gross goods=254.674343640bn, R0=14.082326799bn,
H0=41.065648654bn. On the plan's national-accounts basis, initial household
FCE becomes 309.822319094bn and initial runtime GDP gains H0 once to
563.474205089bn. Leave synthetic GDP formulas unchanged. Increment 3 must
update the runtime initial row and dynamic equations atomically; growth must
compare new-basis rows. Do not add H0 a second time to the synthetic accounts.
Firm output/GVA/operating surplus retain firm scope; the separate bridge is H.

Initial household rents come from synthetic matched properties/social housing;
`Households.compute_rent` subsequently uses settled housing-market properties.
Use those authoritative flows, not a housing-wealth yield invented here.

The initial `cacf_real_consumption_budget` equals `data['Consumption']`, the
VAT-exclusive goods seed (226.223946104bn), not gross total consumption.
Subsequent rows are the gross behavioural T deflated to the model base.
This is a verified concept mismatch. The coherent CACF ECM seed is
`(gross_goods+R0+H0)/initial_CPI`, after effective VAT and housing initialisation.
Do not multiply the old seed blindly by VAT or change non-CACF behaviour.

A second initial inconsistency prevents silently choosing that goods basis:
`1.13*sum(consumption[0]) = 255.633059098bn`, but
`total_consumption[0] = 254.674343640bn`, a 0.958715457bn difference.
`create_households_timeseries` seeds the vector from HFCS `data[Consumption]`
and the aggregate from a separate industry-consumption calibration. The
proposed 309.822319094bn FCE above uses the existing aggregate basis; it is
not the sum of gross household initial consumption plus housing.

User decision requested: preserve the aggregate anchor and proportionally
align household initial values (factor 0.9962496421214523), or preserve the
household values and reconcile aggregate initial accounts. Neither adjustment
has been made. This is the remaining Increment-0 exit gate, before changing
ECM initialisation or initial GDP/FCE. Whichever basis is selected must be
consistent across household outputs, ECM and national accounts, without
silently changing non-CACF behaviour. A zero initial residual is not grounds
for hiding this difference.

## 5. Baseline and fixtures

The two-period run completed with finite GDP and housing inputs. GDP output
is 510.590230235bn at t=1 and 469.818394387bn at t=2; imputed rent is
41.135749999bn and 41.274547444bn. Pre-balancing expenditure residuals are
-256.891052 and +58,296,334.218445 LCU. Adjusted equality is not evidence that
the latter discrepancy is absent. Existing markup-anchor warnings were emitted;
no exception or non-finite GDP occurred.

`gdp_prechange_fixtures.json` stores eight hand-computed cases (balanced /
37.5 unbalanced; adjusted / unadjusted; H=0 / 80), plus both exact production
GDP calls. The synthetic case has cash rent 100, private rental income 30,
public rental income 40, and independently balanced GDP=450 at H=0. The
pre-change function ignores H; Increment 3 must change every GDP expectation
by H and FCE by R+H while preserving the raw residual and trade adjustment.
The replay test calls real `Economy.compute_gdp`; it does not reimplement it.

Validation: 13 targeted tests pass (10 new GDP fixture cases plus the two
existing rent/GDP tests and planning-snapshot test); Ruff check and format
pass. The initial-basis decision above remains open, so Increment 0 is not
claimed fully closed. No VAT, refresh, initial ECM, support-unit, or GDP
behavioural correction has been implemented.
