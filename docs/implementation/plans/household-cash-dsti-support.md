# Household cash receipts, DSTI and support: proposed repair plan

Status: proposed for user review. The three repairs below are not implemented.
Branch: `fix/household-cash-dsti-support`.
Starting point: `3a83c625`, the PR #163 branch after integration of main through PR #162.

## Current branch wrap-up

The configured `idiosyncratic_persistence: initial_state` now has its implementation. Main's epsilon initialization, Kalman common-signal and initial-share transfer fixes were merged cleanly. The consolidated consumption block, intercept0.25, income window4 and aligned-income switch remain. Production configuration loading and rule construction succeed; lint/whitespace checks pass. No tests or simulations were run. The new branch is stacked on PR #163 until that PR is merged.

## Common scope

Repair accounting and origination rules before choosing coefficients by unemployment outcomes. Keep epsilon initialization as selected by the user. Preserve the existing household income concept for learning; correct its units without silently extending it to property income. Keep cash income distinct from imputed rent and capital gains. Do not refit consumption coefficients or use frozen holdouts. Do not overwrite saved runs; register fresh outputs if simulation is later requested.

Implement in order: wage receipts -> DSTI -> residual support. Each increment should have a separate commit and a short change/accounting review. Before implementation, review the proposed choices below with the user.

## Plan 1 — Nominal wage receipts and consistent cash/learning inputs

### Problem

Individual employee_income stores real earnings. Individual realised income and firm payroll convert it with CPI, but household employment-income aggregation omits that conversion. Expected income is a separate mechanical forecast; learning observes employment plus transfers after deflation. Correcting only one consumer would leave mixed units.

### Proposed changes

1. Establish one nominal net employment-receipt vector, using the configured consumer price index: `nominal_wage_receipt = CPI * real_employee_earnings` for employed individuals.
2. Route household employment-income aggregation through this component. Do not use full individual total income, because pensions, unemployment and dividends have their own household postings.
3. Retain the existing CPI conversion for pensions, unemployment and regular transfers. Do not multiply current rent, dividends, interest, support or imputed rent by CPI again. Keep IFA gains outside cash income.
4. Reconcile nominal household net wage receipts plus employee deductions/employer contributions with the corresponding firm payroll. Preserve the configured wage CPI elasticity; a non-unit elasticity must use a common conversion rather than silently split firm and household conventions.
5. Feed corrected nominal income into wealth/cash settlement. Form the learning signal from corrected non-property income divided by CPI exactly once, including consistent initial/history rows.
6. Separate forward expected income from receipts available for current feasibility. After wages and benefits are known, refresh the financing plan from current nominal receipt components. Preserve an explicit expected-versus-realised diagnostic for uncertain rental/dividend receipts; any unresolved shortfall must reach the financing/support authority rather than become an unapproved overdraft.

### Expected files

- `macromodel/agents/individuals/individuals.py` and/or `func/income.py`: expose/reuse the wage-only nominal component if needed.
- `macromodel/country/country.py`: aggregate and route current receipts, expectations and learning signal consistently.
- `macromodel/agents/households/households.py`: feasibility income handoff and cash-component use.
- `macromodel/agents/firms/firms.py`: inspect/reconcile payroll; change only if the shared conversion requires it.
- Household diagnostic helper only if new receipt/timing outputs are needed.

### Acceptance conditions

At CPI different from1, wage receipts are converted once; the other components retain their units. Household net wage/tax/contribution postings reconcile with firm payroll under the maintained tax convention. Learning recovers the intended real wage signal. A nominalisation change must not also add benefits/dividends twice. Current feasibility and final cash settlement must identify any remaining receipt error explicitly.

### Proposed decision

Retain real wage storage and nominal cash ledgers. Use current known nominal earnings for current financing; keep forward expectations separately. Do not redesign permanent-income learning or its income coverage in this increment.

## Plan 2 — One all-debt DSTI origination rule

### Problems

- The paper constrains mortgage plus total consumer service and assumes costless consumer refinancing when extra credit is taken.
- Current capacity deducts mortgage service but treats consumer capacity as entirely new credit, leaving existing consumer schedules in place.
- The proxy uses12quarters, while bank contracts use2quarters.
- Paper income is quarterly gross income; current proxy uses expected net cash receipts. LTI and bank constraints do not replace a consistent total-service cap.

### Proposed changes

1. Define one underwriting snapshot: income concept, existing consumer principal/arrears/service, mortgage service, sanctioned overdraft service, rate, maturity and period units. Exclude imputed rent and capital gains from cash-service capacity.
2. Adopt the paper's refinancing interpretation for households requesting positive additional borrowing: compute capacity for the total consolidated consumer balance, then subtract the amount being refinanced to obtain the net new cash limit. Outstanding service/arrears must enter once; no automatic refinancing for non-borrowers.
3. Make refinancing a matched transaction: replace old consumer liabilities and lender assets with the consolidated contract; only net additional principal reaches household cash. Resolve creditor allocation, accrued interest/arrears and any FICP exclusions explicitly. Do not forgive principal or hide a refinancing shortfall.
4. Specify the timing of opening-period dues and the new schedule before coding. Old dues, refinanced balances and next-period annuities must not overlap or disappear. Refresh feasibility and bank supply using the same origination decision.
5. Use one maturity source for capacity and newly issued/refinanced contracts. Proposed value:12quarters, matching the paper. Preserve inherited contractual schedules until an explicit refinancing transaction; inspect initial synthetic-loan creation separately rather than silently reseason all opening loans.
6. Define quarterly gross cash income from the model's existing wage/tax and benefit components, not an exact HFCS tax reconstruction. If complete gross components cannot be supported, return the denominator choice to the user instead of applying35% silently to net income.
7. Apply bank capital, supply and other maintained restrictions to the same decision. Check the resulting combined service at the actual contract rate/maturity. Keep origination DSTI distinct from later realised service/income and voluntary early repayment.

### Expected files

- `macromodel/agents/households/func/residual_capacity_fallback.py`.
- `macromodel/agents/households/households.py` and `macromodel/country/country.py`.
- `macromodel/markets/credit_market/credit_market.py` and `func/clearing.py`.
- `macromodel/configurations/bank_configuration.py`: shared origination parameters.
- `run_model/config/country_config_FRA.yaml` and `consumption_paper_parameters.yaml`: one maturity source and explicit cap denominator.

### Acceptance conditions

An already indebted borrower cannot receive a full fresh consumer capacity on top of unchanged existing service. Consolidated principal equals debt refinanced plus net cash, with matching lender assets and no duplicate household proceeds. Actual contractual service satisfies the maintained all-debt cap at origination. Capacity and issuance use the same rate/maturity. Existing non-refinanced contracts are preserved. Arrears, zero/negative income and zero rates have explicit outcomes.

### Decisions proposed for approval

Use the paper's consumer refinancing pool, quarterly gross cash income and12-quarter new/refinanced maturity. If the user prefers incremental lending without refinancing, subtract existing consumer service instead; that is a different implementation and should not be mixed with the pool approach. Confirm the gross-income convention before coding this increment.

## Plan 3 — Support computed after actual payment relief

### Problem

Consumer suspension and government support currently use the same pre-suspension gap. Suspension records unpaid service; it is not itself booked as a fiscal transfer. Support ignores the resulting cash relief. Saved first-period overlap is about€29.8bn in each income arm. The mortgage path currently services full scheduled amounts despite the paper allowing mortgage suspension after consumer relief.

### Proposed changes

1. Put the uses-and-sources calculation behind one household financing authority, using consistently nominal receipts from Plan1 and committed financing from Plan2.
2. Allocate liquid assets/committed borrowing/executable asset sales, then cut discretionary consumption to the maintained minimum. Preserve cash rent versus noncash imputed rent distinctions.
3. Settle consumer service first under the paper's priority, then mortgage service if further relief is required. Record actual payments and unpaid amounts as matched borrower/lender arrears; diagnose rather than silently finance those amounts with negative deposits.
4. Recompute the gap from actual payable service and authoritative receipts. A saved pre-relief gap is not an authority for a transfer after obligations have changed.
5. Authorise only the remaining subsistence top-up. Include it once in existing social-transfer aggregates and once in government expenditure, retaining a separate diagnostic series. Existing regular transfers are already a funding source, not an additional second credit.
6. Reconcile final cash, debt service, arrears, asset sales and fiscal payments. If late receipts differ, route the difference through the same explicitly defined authority before final settlement.

### Expected files

- `macromodel/agents/households/func/financial_feasibility.py`: post-relief gap/plan fields.
- `macromodel/agents/households/households.py`: floor and financing handoffs.
- `macromodel/markets/credit_market/credit_market.py`: actual consumer/mortgage relief and matched arrears.
- `macromodel/country/country.py`: ordering, support calculation and transfer posting.
- Government/bank settlement and timeseries schema files if authoritative mortgage arrears need new carriers; discover exact scope before edits.
- Household diagnostic helper: before-relief gap, relief, remaining gap and paid support.

### Acceptance conditions

For resources80, minimum consumption100 and scheduled service20: suspending20 leaves support20, not40. If policy instead pays40 to finance full service, actual payment remains20 and unpaid service is0. Each obligation has exactly one outcome. Borrower arrears match lender records; top-up matches fiscal expenditure; current income/support is posted once. Final cash is funded or has an explicit unresolved obligation—never a silently clipped negative balance.

### Proposed decision

Maintain the paper's last-resort support after consumer-then-mortgage relief. Integrate the transfer in existing fiscal aggregates while retaining its diagnostic component. Do not adopt an implicit government debt-service subsidy or a fixed-budget redistribution rule without a separate decision.

## Review boundary

The initialization compatibility error is fixed on PR #163. The three plans above require user review before implementation. No new receipt, DSTI, maturity, refinancing, mortgage-relief or support policy is implemented by this document. Numerical/regression validation scope should accompany implementation; no tests/simulations have run for these proposals.
