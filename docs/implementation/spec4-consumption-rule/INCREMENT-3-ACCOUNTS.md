# Increment 3, task 1 — funding contract and source/account map

Completed 5 November 2026. Governing documents: [PLAN.md](PLAN.md) and
[INCREMENT-3.md](INCREMENT-3.md). This task records the user-resolved funding
assumption and maps existing source seams. It implements no shock harness and
registers/runs no experiment. Source inspection and arithmetic checks are not
proof of complete simulation settlement.

## Confirmed funding decision

The user accepts an exogenous external fund for this controlled experiment.
Use the model's existing household transfer/receipt and bank-household LFA links.
Do not build a new behavioural institution or endogenous financing policy.
Record the external payer's outflow once and distinguish income receipts from
capital/endowment transactions. Report gross injection / baseline quarterly
nominal GDP and / baseline quarterly household cash income, with units/date and
zero-denominator handling. Do not assume a proposed shock is small before these
ratios are measured. Large interventions require an explicit endogenous financing
design on user direction; no numerical large/small cutoff is inferred here.
This does not prevent implementing the isolated harness and its tests.

Existing public pensions, unemployment benefits, other transfers and subsistence
support remain endogenous under their existing rules. Exogenous funding does not
freeze household choices, bank responses, prices, credit or employment.

## Native flow receipt route

| Step | Existing source seam | Meaning for the harness |
|---|---|---|
| Stage | country.py: Country.stage_household_income_shock | Copy one H-vector T once for the shock quarter. Reject duplicate staging; never apply an LFA top-up as well. |
| Expected cash | Country._set_household_income_expectations | Existing _apply_household_income_shock wraps expected income; append/replacement computes baseline+T afresh, not cumulative additions. |
| B and M | Country._record_behavioural_income and _set_household_target_demand | T enters the non-property bridge once per pass; existing reconciliation/floors and ready M/q producer remain in charge. No extra learner call. |
| Gross resources/support | Country.household_gross_cash_income and reconcile_realised_household_support | The staged receipt is visible to underwriting and support reconciliation. Record support crowd-out, not just the gross injection. |
| Realised cash | Country.update_realised_metrics | income.append(_apply_household_income_shock(compute_income())) books T once; aligned realised-income diagnostics also see the receipt. |
| Clear | Country._clear_household_income_shock | Called after realised income recording; no repeat in the next quarter. Reset clears it too. |
| Wealth | Households.update_wealth | income feeds cash_saving_before_financing/base_lfa under the active feasibility resolver, then liquidation/portfolio settlement determines closing LFA. Legacy settlement uses its own native wealth recurrence. Do not add T again to closing LFA. |

Reuse the staged external-income route, not a second edit to social-transfer
components. It is explicitly an externally funded household receipt: existing
income_social_transfers and government transfer-spending totals do not contain T.
Do not falsely label it a domestic government expenditure or add T to government
debt. This respects the available government-household channel while avoiding a
second booking through it. If a fiscal pass-through experiment is later wanted,
that is a distinct instrument with matching government grant revenue/expenditure.

The existing completed-income belief update calls compute_non_property_income,
which uses employment plus recorded social transfers, not the staged external
receipt. Preserve that convention and disclose it: the current planning M/q bridge
sees the receipt, but the completed earnings-learning signal does not acquire a
new government-transfer component. Do not silently change learner mathematics or
reclassify this external validation receipt to make the two signals identical.

## Native LFA and banking route

After cloning the unshocked pre-intervention state, add A once to the shock
copy's current liquid_financial_assets before opening-stock capture. Preserve the
original pre-shock row in saved evidence: changing the current mutable array must
not rewrite the reported control/pre-shock checkpoint. Use an owned replacement
array and an explicit intervention snapshot. Keep opening totals/wealth diagnostics
consistent with that same intervention state; do not edit initial_states or older
history rows. No stage_household_income_shock call for this arm.

Households.update_wealth reads current LFA as its opening stock, so the addition
continues through the native cash/portfolio recurrence. It is not an extra income
receipt. Subsequent returns, support, portfolio movements and spending may change;
closing shock-minus-control LFA need not equal A.

Banks.update_deposits groups signed household LFA with np.bincount using
households.states['Corresponding Bank ID'], and persists deposits_from_households
and total_deposits_from_households. Do not independently add A/T to those closing
series after this aggregation: that would duplicate the transfer.

Banks.compute_deposits supplies the existing signed balancing position:

```
D_bank = deposits_from_firms + deposits_from_households + equity - total_outstanding_loans
```

At fixed other accounts, an additional household deposit of A increases both
its bank's household-deposit balance and D_bank by the same allocated amount.
D_bank can be negative; do not introduce a positivity floor or invent an additional
reserve asset on top of this native balancing position. Native liabilities use
Banks.compute_liability with the existing sign conventions.

Before the first planning pass of the LFA arm, synchronize current bank deposit
aggregates/balancing position to the intervention snapshot without appending a
spurious model period. At ordinary closing, let native aggregation recompute them.
The harness tests must check this boundary, alias independence and that the
intervention is not overwritten. Bank liabilities/equity/loan and aggregate
histogram/total consistency must follow native conventions, not independent cash
creation. Source mapping identifies the seam; the exact reusable synchronization
helper and its execution tests are later implementation tasks.

The fund's external outflow provides the outside counterpart to the domestic
resource addition; the native bank balance equation supplies the represented
settlement position. Do not claim this experiment adds a fully modelled foreign
sector balance sheet. Do not add a second settlement asset merely because the
external payer is represented in the experiment ledger.

## Government accounts and invariants

Country.settle_household_social_transfers records pensions/unemployment/other
transfers/support, and late government aggregates read those components.
CentralGovernment.compute_deficit uses total_unemployment_benefits,
total_public_pension_benefits, total_other_social_transfers and
 total_necessity_support, government goods expenditure, interest and revenue.
CentralGovernment.compute_debt adds that deficit to debt.

Therefore leave domestic fiscal formulas untouched. They may change endogenously
through support, tax revenue and other responses. External-fund outflow is not
silently domestic deficit financing, bank equity loss or household borrowing.
Check payer outflow against the booked gross intervention, not net income after
support crowd-out. For each bank, report opening/closing native balance residuals
and all intervening flows rather than equating the later deposit difference to the
original injection.

## Validation and remaining boundary

Source/function references were checked on the current branch. Four local numeric
checks exercise native Banks.update_deposits/compute_deposits and
DefaultWealthSetter.compute_wealth_in_deposits on signed household balances,
multiple bank assignments and a fixed endowment. Existing focused income-shock
regression checks the expected/realised wrapper and clearing behavior. These are
local arithmetic/regression checks, not a new macro experiment or a claim that
the full intervention harness exists.

Evidence: vault outputs/implementation/2026-11-05-increment-3-accounts/.
Next task: implement a checkpoint/clone primitive with RNG preservation and
independence tests, without any nonzero intervention or production simulation.
The funding decision is resolved; full receipt/snapshot/settlement integration
will be tested in its own subsequent task. Stop after this mapping task.
