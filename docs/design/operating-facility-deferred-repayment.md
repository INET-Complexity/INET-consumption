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
