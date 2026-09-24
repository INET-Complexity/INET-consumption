# Total-consumption Increment 1

Base: `0b7aacf3`, Increment 0 on `pr/fix-imputed-rent` (PR #152).
Scope: behavioural target split and its financing. Initial ECM/GDP basis,
realised cash/total output series, and MPC analysis remain later increments.

## Cash contract

CACF preserves its gross total target T and real ECM state, but sends
`max(T-R-H,0)/(1+VAT)` to the goods market. Both housing-flow diagnostics and
the gross desired-goods aliases use the same target-pass inputs. Negative
housing observations retain the existing zero normalisation; invalid shapes
and non-finite values fail explicitly.

The existing `formula_implied_mpc` remains a finite difference of the
VAT-exclusive full target. The new `target_consumption_total_mpc` differences
T using the same two evaluations and random draw. It is NaN initially and
when the rule is unavailable/non-CACF. Both target passes share one row.

Planning stores the cash-rent snapshot for legacy consumption/mortgage credit
helpers. Live resolver credit still comes from its existing capped carrier.
Liquidity, borrow/sell and floors use nominal gross consumption cash amounts;
investment carries its separate taxes. Executable goods are converted back to
VAT-exclusive orders. Zero desired goods can receive a supported subsistence
top-up using existing industry weights.

After the second target pass and before authoritative payment settlement,
`refresh_cash_needs` replaces only funding/residual fields on the settled
carrier. Opening liquid resources, fixed consumer credit, and committed
executable liquidation count once. Mortgage proceeds and property costs are
separate cash flows. Bank/debt booking, rationing history and liquidation
reservation remain unchanged; no reconciliation or credit booking is rerun.
The existing funding diagnostics are replaced with that authority.

Household wealth debits consumption VAT once on realised consumption, which
excludes investment. Existing government VAT booking is unchanged. Rent and
investment reserve cash before optional consumer repayment. Stage 5 support
is already nominal and no longer receives another CPI multiplication. The
legacy resolver-disabled path retains its financing policy and fails explicitly
if its resources cannot cover mandatory cash rent before goods orders.

## Validation

Focused tests cover target identities, overlapping and malformed housing flows,
nonzero VAT and investment, both planning passes and resolver settings,
fixed finance authority, zero-goods top-ups, early repayment, support at CPI=2,
loan/payment retry guards and the new MPC lifecycle. The simulation-order
regression checks target → cash refresh → payments.

The local FRA seed-15 quarterly two-period smoke uses the unchanged Increment 0
capture script and configuration. The saved HDF5 is checked against
`G=max(T-R-H,0)` and `MPC_total=(1+VAT)*MPC_legacy`, using rtol=1e-10 and atol=1e-6.
704 targeted tests pass: 595 household/country and 109 credit, simulation-order
and pre-change GDP fixtures. Ruff check/format and whitespace checks pass.
The broader suite was interrupted during unrelated data-fixture generation
after 723 passes; it is not a complete-suite result. Run outputs and source
hashes are recorded in `validation.json`.

The initial consumption discrepancy (household gross goods 255.633bn versus
aggregate 254.674bn LCU) remains unresolved. No ECM seed or GDP/FCE accounting
expression is changed. The short smoke is a runtime check, not macro validation.
