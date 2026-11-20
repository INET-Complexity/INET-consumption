# Operating facility: deferred repayment

Status: proposed timing and accounting contract; runtime implementation pending.

## Problem

Current settlement can repay a new operating draw in its origination period.
Interest is charged on opening facility balances only, so such draws incur no
facility interest. Deferring repayment changes debt duration, cash holdings,
bank exposure and future credit capacity as well as interest.

## Proposed period contract

1. Freeze opening facility principal B at the start of period t. Only this
   carried principal is eligible for repayment or ordinary short-term refinance.
2. Charge B once using the existing current-period short-term quote. New draw
   N in t receives no t interest and remains outstanding through closing t.
3. Ordinary refinancing replaces facility principal with ordinary loan debt;
   it does not create profit. Extinguish the corresponding facility exposure
   exactly once. Verify ordinary-loan interest timing to avoid overlapping charges.
4. Draw N for realised eligible operating needs using existing funding rules.
   Exclude N from repayment eligibility in t, even when sales generate cash.
5. Reserve taxes and senior contractual debt service before eligible facility
   repayment and dividend settlement. Preserve existing overdraft-repair priority.
6. Closing facility principal = B - refinanced principal - cash principal
   repayment + N. Reconcile this liability with lender closing exposure.
7. At t+1, N becomes opening principal eligible for repayment/refinancing and
   interest. Unpaid eligible principal must follow an explicit existing or
   revised arrears/default rule; never silently disappear or count as new income.

## Accounting requirements

Draws increase firm cash and facility liabilities. Principal repayment reduces
both; neither is revenue or expense. Interest reduces firm profit and cash
when paid and creates matching bank receipts. Any unpaid interest needs an
explicit accrual/capitalisation and default treatment; bank cash income must
not include uncollected interest. Refinancing must reconcile both loan types
without duplicate cash, principal or interest entries.

Track opening principal, new draws, refinancing, cash repayments, closing
principal and paid interest separately. Report aggregate flows alongside debt
stocks in notebook output and preserve old saved-output loading.

## Decisions and verification still required

Verify the current refinancing and loan-interest ordering against source;
specify unpaid principal/interest treatment and applicable settlement priorities.
A full period of interest at t+1's quote is an explicit model convention.
Check cash retained by delayed repayment against existing dividend affordability
rules before implementation. Profit-payout redesign belongs in the next PR.

Validate no draw, new draw, carried balance, partial/full refinance, constrained
cash, multiple banks, zero/positive rates, default and firm/bank reconciliation.
Then register and run matched multi-seed economic validation separately from
unit/accounting checks. No economic acceptance is implied by this document.

## Source verification — 2026-11-20

Verified against runtime at 12b1976d (documentation branch d3a5432a).

- Interest preparation precedes credit clearing: charge the prior closing
  facility balance at the current quote (`firms.py:2022`, `country.py:2517`).
- Credit requests include the full carried facility balance as ST refinance
  (`firms.py:1892`). Cleared refinance is separated from spendable ordinary
  ST credit (`credit_market.py:1080`); the converted amount reduces facility
  principal before activity planning (`firms.py:2034`, `country.py:3070`).
- Ordinary loan settlement services existing cohorts and merges new cohorts
  afterwards (`credit_market.py:2464–2545`). Thus new refinancing loans are not
  charged ordinary contractual interest in their origination period; the
  carried facility charge covers that period without this double charge.
- Realised ordering is draw, reserve corporate tax, allocate ordinary loan
  debt service, repay facility, settle previous dividend, update cash/equity,
  accounting controls, then insolvency (`country.py:4025–4165`).

### Existing insufficient-cash protocol: preserve it

Ordinary debt service pays opening interest arrears, contractual interest,
opening principal arrears and contractual principal in that order. Unpaid
interest is capitalised into loan principal; unpaid principal remains arrears.
Either shortfall sets the liquidity-failure flag (`firms.py:2147–2242`).
Unpaid facility principal instead stays in closing facility exposure and is
requested for ST refinancing next period. It is not automatically put through
ordinary-loan arrears accounting. Facility nonpayment alone does not set the
ordinary debt-service liquidity-failure flag.

Facility interest is included in operating commitments and hence can induce
an additional facility draw. It is deducted before ordinary debt service and
booked to the bank, rather than entering the ordinary-loan unpaid-interest
capitalisation mechanism. Preserve and test this existing funding convention;
do not invent a separate facility-interest arrears policy.

Default requires negative equity AND the settlement liquidity-failure flag.
On default, ordinary loans, residual overdraft and closing facility exposure
are written off to the corresponding bank; facility balance is cleared
(`firms.py:3940–3998`). Deferring new-draw repayment must retain this protocol.

### Dividend planning dependency — user clarification

A provisional payout based on t profits must account for expected t+1 debt
repayment, including the carried facility, without treating principal as a
profit expense. At t+1 the proposed payout must enter cash planning; actual
payment remains subordinate to senior obligations and the unpaid proposal
expires. Current operating draws include wages, intermediate purchases,
production/corporate tax and signed non-loan interest, but no dividends
(`firms.py:2044–2067`). Current final dividend settlement is cash-capped after
ordinary debt and facility repayment (`country.py:4053–4061`).

Do not simply add the whole provisional dividend to an unconditional facility
entitlement: that could guarantee the proposal with new borrowing and defeat
the affordability/expiry rule. Specify the affordable planned payout first,
then whether/how it enters funding needs. Expected repayment must distinguish
cash repayment from refinancing so converted principal is not reserved twice.
This forecasting/funding dependency belongs in the stacked payout contract;
record it explicitly before runtime implementation of that contract.
