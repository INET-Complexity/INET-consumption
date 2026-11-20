# Operating facility implementation plan

Status: proposed implementation sequence for draft PR #210. No runtime edits
or simulations performed by this planning change.

## Scope and source baseline

Implement deferred repayment on work/operating-facility-interest, then build
the profit-payout contract as a separate stacked PR. Use the isolated clone;
preserve the canonical checkout's fiscal/configuration/notebook edits.
Runtime baseline is main 12b1976d; specification commit is 31983205.

GitNexus repo inet-facility-pr-audit was refreshed in index-only mode at the
specification head. Query, context and upstream impact identify these seams:

| Seam | Direct caller / dependent | Planned action |
|---|---|---|
| Firms.prepare_operating_revolving_facility_interest | Country.prepare_credit_market_clearing | Preserve pre-refinance opening interest base |
| Firms.begin_operating_revolving_facility_period | Country.prepare_post_credit_feasible_activity_plan | Preserve conversion of cleared refinance |
| Firms.settle_operating_revolving_facility | Country.update_realised_metrics | Restrict cash repayment to unrefinanced carried principal |
| Firms.draw_operating_revolving_facility | Country.update_realised_metrics | Preserve operating draw formula in facility PR; audit signed receipts |
| compute_cash_feasible_firm_distribution_settlement | Country.update_realised_metrics and dividend tests | Preserve seniority; measure changed cash cap |

GitNexus gives settlement/draw LOW upstream caller-count risk, one direct
country caller each. Economic/accounting risk is high: dynamic orchestration,
state-series consumers and notebook access are not fully described by those
call edges. Confirm source and tests as well as graph results.

## 1. Freeze the state transition and regression examples

Let B be prior closing facility exposure, R cleared refinancing, O=B-R the
remaining carried balance, N the current draw and C cash after tax reserve and
ordinary debt service. Existing clipping ensures 0<=R<=B.

- Facility interest I=B*current_period_ST_rate, prepared before refinancing.
- Cash principal repayment P=min(O,max(C,0)); new N cannot be repaid in t.
- Pre-default closing exposure F=O-P+N.
- Refinance moves principal to an ordinary ST cohort, with no discretionary
  cash gain. New cohorts are serviced starting next period.
- On default, write off F and clear facility debt through existing machinery.
- Interest may induce an operating draw, as currently implemented. Preserve
  ordinary-loan capitalisation/arrears and the existing default trigger.

Concrete oracle: B=10, R=4, N=10, C=20 implies I=10*r, P=6 and F=10.
Current test expects P=16,F=0; revise it to the new contract. Next period with
B=10,R=0,N=0,C=20 gives I=10*r_next,P=10,F=0. Full refinance R=10
instead gives O=0 and ordinary ST debt 10, with no duplicate current charge.

## 2. Implement the maturity boundary

Edit macromodel/agents/firms/firms.py, primarily
settle_operating_revolving_facility. Keep existing saved field names and
one-entry-per-period conventions. Update docstrings to distinguish original
interest base from post-refinance opening repayment balance. No new exposure
fraction or interest calibration is needed.

Inspect Country.update_realised_metrics without reordering unrelated steps:
draw -> tax reserve/ordinary debt plan -> eligible facility repayment ->
previous dividend settlement -> deposits/equity -> accounting -> insolvency.
Change country code only if necessary to propagate the defined amounts.

Keep compute_deposits and check_firm_accounting_controls aligned with
cash +draw -repayment and the existing refinance offset. Do not deduct
principal from accounting profit or count draws as revenue. Confirm bank
interest, closing loans, prudential exposure and default writeoffs include
facility state exactly once. Expected result: greater closing cash and debt
can arise together; disclose the resulting dividend and capital-capacity effects.

## 3. Add flow and stock diagnostics

In run_model/src/visual_helpers.py aggregate saved firm series into distinct
columns for facility draw, cash principal repayment, closing balance and
interest paid. Include opening pre-refinance balance and refinance flow where
needed to reconstruct the transition (avoid unnecessary duplicate state).

In run_model/config/config.py add labels and a plotting group consumed by
plot_output(). Add summed-firm views in run_model/run_model.ipynb, preserving
existing cells/outputs. Check grid capacity and units: nominal LCU per period
for flows; closing nominal LCU for stocks; no annualisation of amounts.

Missing historical series must be visibly unavailable or omitted, not silently
reported as observed zero. Preserve row alignment, including the t0 row.
Compare summed firm closing exposure to bank exposure, and firm paid interest
to bank receipts. Report pre-default closing plus writeoffs where required to
bridge to post-default saved closing debt.

## 4. Focused accounting and integration tests

Extend existing suites rather than adding tests that merely repeat a formula:

- test_agents/test_firms/test_firms.py: revise opening-balance test; exercise
  two consecutive periods, ample cash with new draws, partial/zero cash,
  mixed refinance plus new draw, and existing arrears-priority protection.
- test_agents/test_firms/test_deposit_income_finance_regression.py: retain
  signed deposit-income effects on draw and feasible investment budgets.
- test_markets/test_credit_market/test_credit_market.py: preserve normal CAR
  limits and refinance ring-fencing; verify new refinance cohort starts next
  period and cannot produce duplicate current interest/principal payments.
- test_agents/test_banks/test_banks.py: closing facility exposure reconciles
  to bank loans; multiple-bank interest and writeoff allocation reconcile.
- test_country/test_diagnostic_dividend_fund.py: dividend cash cap follows
  eligible repayment, remains subordinate to senior claims, and receipts and
  withholding match actual debits. Explicitly show the facility-only change's
  effect on the old payout rule without treating it as the new profit rule.
- Existing distress tests: unpaid facility carries/refinances; ordinary unpaid
  interest capitalises and principal enters arrears; negative equity plus
  liquidity failure triggers bank writeoff including facility exposure.
- tests/test_run_model/test_visual_helpers.py: aggregates/units/t0 alignment,
  historical missing series, and figure layout.

Use INET-consumption/.venv/bin/python for tests. Run focused suites first,
then the full model suite and repository Ruff checks. Execute necessary
notebook/plot checks without rerunning simulations in notebook cells. Inspect
final GitNexus detect_changes against main and review direct dependents.

## 5. Register and execute economic validation

Before any simulation write a vault experiment specification with frozen
source/config/input hashes, commands, seeds, shock timing, tolerances, runtime
and memory budget, abort triggers and analysis. Compare frozen current main
with the facility candidate on identical inputs; do not import parallel fiscal
or wage changes implicitly.

Proposed campaign: serial seed40 T6 smoke for both arms, then seeds40–44 T50
baseline and matched +100bp annual innovation at zero-based q20 for four
periods (20 runs total across source and shock arms). Confirm shock convention
against the existing launcher before registration. Measure one run peak RSS
and duration; estimate total cost before launch. Start serial; at most two
workers only if measured memory permits and existing resource rules allow.

Mechanical acceptance: all runs exit cleanly, outputs finite, prices positive,
no accounting-control failures, no new-draw repayment, no duplicated interest,
nonnegative facility principal within numerical tolerance, and all firm/bank
cash, exposure and writeoff identities pass registered scale-aware tolerances.
Require exact paired pre-shock replay within each source arm. Candidate and
main paths may diverge from period1 because the facility contract changes.

Report paired seed distributions for draw/repayment/debt, interest, refinancing
requests/grants, ordinary credit constraints, deposits, defaults/bank losses,
dividends/household receipts, investment, GDP and unemployment. Distinguish
first-period cash/debt effects from later interest and general-equilibrium
feedback. Do not require a preferred macro sign or tune payout/rates to obtain
one. Mechanical success and economic acceptance are separate conclusions.
Use spec -> smoke -> campaign -> report; abort on identity/nonfinite failures.

## 6. Stacked payout implementation dependency

Before the payout PR changes funding, define these explicit quantities:

1. At t: eligible after-tax profit, provisional maximum payout, expected t+1
   senior interest/principal including carried facility. Forecast cash principal
   repayment separately from refinancing; do not reserve both for one liability.
2. At t+1 planning: load the provisional payout once, reserve senior obligations,
   and compute an affordable planned payment using an explicit cash forecast.
3. Include that planned payment in funding requirements, as the user requests;
   distinguish a cash reservation from an approved entitlement to new credit.
   Current operating draw formula excludes dividends. Specify whether the
   existing operating facility's eligible uses expand or another financing
   allocation funds the planned payout. Do not infer automatic full funding.
4. At settlement: final affordable declaration/payment after realised senior
   obligations; expire unpaid provisional balance. Reconcile firm debit,
   household gross/net receipts and withholding, with no double reservation.

The forecast information set and precise credit eligibility for dividend cash
are remaining design details, not implemented decisions. Resolve them in the
payout specification before its code. Test profit/no cash, cash/no profit,
expected repayment with full/partial/no refinance, new t+1 draw, binding bank
capacity, forecast error, expiry and recipient accounting. Validate both the
facility-only PR and the combined stack, so payout effects are identifiable.

## Completion evidence

PR #210 must contain the runtime diff, tests, diagnostics, accounting evidence
and registered experiment report before readiness for review. Refresh PR title
and description to the actual final scope. Keep it draft until those exist.
The later payout PR targets this branch until the facility PR is merged.
