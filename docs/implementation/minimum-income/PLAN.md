# Family-adjusted minimum income: design and implementation plan

Date: 2026-11-21. Revision: draft 7. Rebased to current main; D4b, D5, D8, D9 and the numerical guard decided.
Branch: `feature/minimum-income` (draft PR).
Base: INET `origin/main` `38e9be92` (draft 6 base was `a340e124`).
Worktree: `/private/tmp/inet-minimum-income`.

## Workstream resumed — 2026-11-21

The user resumed the workstream. Draft 7 records these decisions:

- **D4b:** a saved part of the payment counts only through next period's financial-asset stock. The previous payment is not added to next period's assessed income.
- **D5:** use the model's existing social housing (house ID −1, social rent paid to government) as the destination after missed rent. Any housing subsidy is in kind, meaning the social-housing place itself. There is no cash rent subsidy.
- **D8:** keep the existing fiscal closure.
- **D9:** compare policy arms by cloning one shared checkpoint (see §10.7).
- **Numerical guard:** a separate positive-domain guard, as §10.5 describes.
- **Assessed income:** the fields follow the real-world French RSA resource rules (§4.5). This mapping is still open research.

Section 3 has been re-checked against `38e9be92`. All listed symbols still exist. See §3.1 for what changed since `a340e124`.

## 1. Objective and decisions already made

Replace the expenditure-contingent Stage 5 necessity transfer with an income-based cash top-up. **Updated user decision (2026-11-17): compulsory minimum goods spending must be independently toggleable**, rather than permanently removed. This supersedes the earlier “replace both” instruction as a configuration requirement.

**Minimum level:** one SMIC times the household family coefficient, converted to the assessment period and model monetary units. Do not retain the old spending floor's 0.5 multiplier as the new income threshold. **Asset condition:** only households whose financial assets are strictly below the household minimum-income cutoff can receive the top-up. The user confirmed total financial assets (liquid plus illiquid, before subtracting debt) and a cutoff equal to one quarter of household minimum income.

The intended sequence is: determine entitlement independently of the household's expenditure choice; make that receipt available for planning and financing; allow the behavioural consumption rule to choose desired consumption; apply ordinary funding constraints and settle actual payments.

An income guarantee is not a promise that recipients spend a minimum on goods or that rent/debt service is always paid. User direction now permits exploring missed rent followed by temporary social housing until a market rental is affordable, with housing subsidies as the alternative to examine. The prior no-rent-arrears restriction is therefore no longer an unconditional constraint on the proposed housing extension. Neither arrears forgiveness nor an automatic subsidy is implied.

This branch currently contains a plan only. The SMIC basis, family scaling, asset-gated eligibility and independent spending-floor toggle are user decisions. After-tax cash income, euro-for-euro withdrawal, exemption of the MPC receipt and replacement of the subsistence-level input floor are now confirmed. The paper supplies the intended income-before-consumption sequence and consumer-before-mortgage suspension order. Payment is now confirmed to freeze after the post-labour assessment, before consumption; late receipts do not change it. The payment affects next-period eligibility, with the income-versus-saved-assets interpretation awaiting clarification. Asset assessment timestamp, housing transition details and numerical guard remain implementation/design work; do not ask the user to choose a new timing model before checking these sources. No runtime source/configuration or cache is changed. This draft does not authorise executing its later increments.

Scope is replacement of the extra Stage 5 transfer, not abolition of public pensions, unemployment benefits or the existing regular-transfer allocation. Keep the HFCS opening allocation and its 54-household legacy reserve, wage settings, credit contracts, house-purchase completion, liquidation fees and calibration estimates fixed unless a separately stated design decision requires a change.

## 2. Why this matters for the conference paper

The 30-seed matched initialisation experiment finds common-sample HFCS impact MPCs of .543/.544/.511/.474/.431 across income quintiles. Q1 has about 50% liquidity-constrained, 36.9% unconstrained, 6.1% self-funded floor-bound and 7.0% supported households. The legacy comparison has more constrained households and higher bottom-quintile MPCs. These are diagnostics, not minimum-income targets.

The new policy should have an interpretable income-insurance mechanism. Do not choose its level, taper or numerical income treatment to make an MPC plot look preferable. A zero response caused by benefit withdrawal is economically distinct from a zero response caused by compulsory spending or an artificial planning-income bound.

Evidence: vault `wiki/experiments/2026-11-17-hfcs-legacy-mpc.md`, `wiki/debugging/2026-11-11-floor-bound-self-funded-households.md`, and the seed proposal `wiki/plans/2026-11-16-minimum-income-design.md`.

## 3. Source-verified current mechanism

Locations refer to the pinned base; symbols, not line numbers, are authoritative.

| Responsibility | Current implementation | Consequence for the redesign |
|---|---|---|
| Benefit planning | `Country._update_benefit_planning_metrics`; `CentralGovernment.update_benefits` | Clears old Stage 5 support; updates pensions, UB and regular transfers. Keep those components identifiable. |
| Expected household income | `Country._set_household_income_expectations` | Builds regular social income and total expected income; called before credit and again after labour, using append versus replacement semantics. New entitlement must enter each required view once. |
| Spendable receipts | `Country.refresh_known_household_receipts` | Replaces expected wages/regular benefits with known receipts in `financing_income`; currently discards the Stage 5 component returned by the social-income helper. Adding a top-up only to expected totals can therefore omit it from financing or accidentally retain it twice. |
| Spending floor and support | `HouseholdFinancialFeasibility.settle_consumption_floor` | Support uses liquid assets, credit and net reserved liquidation as resources; imposes goods floor plus rent before debt service, extra goods and investment. An income test must not reuse this resource definition. |
| Authoritative payment routing | `Country.settle_authoritative_household_payments`; `Households.apply_consumption_floor_to_post_grant_plan` | This method also settles debt, unpaid service, loan balances and distress. Removing its entire call would remove required accounting. Separate support/spending policy from payment settlement. |
| Agent-level compatibility path | `Households.prepare_goods_market_clearing`; `Country.prepare_goods_market_clearing` | Both paths must respect the selected policy. Disabling one call site is insufficient. |
| Ex-post support correction | `Country.reconcile_realised_household_support` | Current support depends on realised goods up to the floor plus cash rent. In the new mode it must not regenerate the retired expenditure-based entitlement. |
| Household income booking | `Country.compute_realised_household_social_income_components`; `Country.settle_household_social_transfers`; `Country.update_realised_metrics` | Current component tuple is pension/UB/other/Stage 5. New minimum income needs an explicit component and once-only sum. |
| Government recording | `Country.update_realised_metrics`, government series `total_necessity_support` and `total_household_social_transfers`; `CentralGovernment.compute_deficit` | The deficit explicitly sums pension, regular other and necessity-support components plus UB; adding a receipt only to the umbrella transfer total would miss its fiscal cost. Extend the explicit sum and funding path once. Keep historical necessity support and minimum income distinguishable. |
| Behavioural income | `Country._record_behavioural_income`; `reconcile_behavioural_income`; consumption input adapter | `subsistence_consumption` is also a positive input floor, including initial bridge histories. Removing compulsory spending does not remove this separate response-flattening mechanism. Setting it to zero can invalidate logs and ratios. |
| Family scale and indexation | `Country._compute_subsistence_consumption_from_units`, `_update_subsistence_consumption`; household CU methods | Existing CU weights are 1/.5/.3, with age 14 as the boundary. Existing floor uses monthly SMIC, period months, CPI ratio and a housing deduction. A benefit schedule is a separate policy choice. |

Current settlement, in nominal cash units, is approximately:

```
A = income without Stage 5 + available LFA + granted credit
    + net reserved liquidation + eligible net housing-sale surplus
S_old = max(F + cash_rent - (A - exempt_resources), 0)
```

Ex post, `F` is replaced by `min(F, realised_goods_cash)`. Thus existing support is a spending/financing backstop, not a minimum disposable-income rule. Merely setting `F = 0` leaves rent support active. Reusing the ex-post correction would also make the proposed income transfer depend on actual purchases again.

### 3.1 Changes on main since `a340e124` that affect this plan

| Change | Location | Consequence |
|---|---|---|
| Replacement-rate unemployment benefit (rate derived from opening data, per-person base from `Wage Rate`) | `ReplacementRateSocialBenefitsSetter`; `Country._initialize_unemployment_benefit_base`, `_distribute_unemployment_benefits` | UB is now per person and depends on wage history, so it is a larger and more variable part of assessed income. Under euro-for-euro withdrawal, a higher UB lowers the top-up one for one. Fiscal tests must cover both setters. |
| Wage-indexed pensions and other transfers (`regular_transfer_indexation: average_wage`) | `CentralGovernment.update_benefits`, `uses_wage_indexed_transfers`; `Country._contract_wage_index_ratio` | M is indexed to CPI and these transfers are indexed to wages, so the gap between them drifts over time. Report this drift. Do not re-index M to hide it. |
| Behavioural-income bridge now excludes property income: `income_bridge_current = max(n, S)` | `Country` bridge initialisation | The floor at S that §10.5 replaces is still there in this new form. |
| Debt-service cash-flow consumption channel | `CreditAugmentedConsumption` (`debt_service_cashflow_propensity`, `DebtServicePayments`) | No change to the entitlement. Debt service is not deducted from assessed income (RSA practice, §4.5). |
| Existing social housing (house ID −1, rent = `rent_as_fraction_of_unemployment_rate` × UB, booked as `Rent paid to Government`) | `SocialHousing` config; `Economy` social-rent series; `Households` social-housing index | D5 reuses this. Moving households into it during a run is not yet verified (§11). |

## 4. Proposed policy definition

Let `M_it` be a household's nominal period income threshold and `E_it` its assessment income excluding the new top-up itself. The initial gap-filling candidate is:

```
M_it = monthly_SMIC_base_ordinary_currency * population_scale
       * months_per_period * family_coefficient_it * price_index_t
K_it = monthly_SMIC_base_ordinary_currency * population_scale
       * 3 * family_coefficient_it * price_index_t
A_it = max(opening_liquid_financial_assets_it, 0) + opening_illiquid_financial_assets_it
eligible_it = A_it < K_it  # validate finite inputs and nonnegative illiquid assets
T_it = eligible_it * max(M_it - E_it, 0)
```

The price index has an explicit base date, normalised to one there. Convert ordinary money to model units exactly once. A monthly level is multiplied by months per period; it is not multiplied by the annual-rate conversion `time_unit / 12`. Do not index both the source base and the resulting threshold.

SMIC, family scaling and euro-for-euro gap filling are confirmed design inputs; the exact current-income receipt mapping still needs verification against the recovered timing contract. Source recovery confirms the existing net-SMIC basis and family coefficients (1/.5/.3): manuscript `main.tex:403–407` gives net monthly SMIC EUR1,128.70 for 2014 and these age-based weights, with CPI indexation. Consumption-notebooks maps the existing OECD-modified consumption-unit field DH0002 (`data/hfcs/frm_all_variables.yaml:45`); it is not necessary to introduce a new family scale. The SMIC number was recovered from the manuscript and existing INET source/configuration, not newly located in a notebook. Reuse the existing dated source/reader; do not hardcode a new live amount or fetch a current-year statutory SMIC. The manuscript's half-SMIC multiplier concerns the old spending floor; this new user-selected income threshold uses full SMIC. The asset test uses total financial assets, liquid plus illiquid, without subtracting debt, against one quarter (three months) of the household minimum income. In the current quarterly model K equals M. If a monthly model is used, M covers one month but K still covers three months; do not change the confirmed asset cutoff with the simulation timestep. Both are expressed in the same model currency units. The asset test excludes households at equality as well as above the cutoff. Assess opening assets before paying this period's benefit so its own deposit credit does not retrospectively disqualify the household. If assessment income differs from actual current income, it guarantees the assessed measure only for households passing the asset condition. Treatment of negative assessment income must be explicit; silently clipping it changes the guarantee.

“Benefit withdrawal” means reducing the top-up when assessed income rises. With a 1,500 minimum and assessed income of 1,000, the gap-filling top-up is 500. If eligible income rises to 1,100, the top-up falls to 400: total income remains 1,500 while the guarantee binds. This is one-for-one withdrawal, not a separate tax. Whether an experimental windfall is counted or exempt changes this result. These amounts are examples, not selected SMIC values.

**User decision:** use euro-for-euro withdrawal (tau=1), and pay the MPC experimental receipt on top of the guarantee: exclude it from assessment income while including it once in actual spendable resources. This exemption is the primary MPC convention, not an undecided sensitivity. Asset eligibility uses its separately defined assessment-date stock; the receipt must not disqualify its own payment within the same assessment. Subsequent-period asset eligibility is still assessed normally.

For background only, households passing the asset test could instead use `T = max(M - tau * E, 0)` on an explicitly defined eligible-income domain. For `0 < tau < 1`, the cutoff is `E = M/tau`, not `M`; this is a different policy with different cost and coverage. `tau = 0` is a flat payment to asset-eligible households; with an asset gate it is not universal. No alternative taper is selected or required for the user's rule.

### 4.1 Independent controls and income ledger

Proposed configuration separates two controls:

```
household_support_mode: legacy_necessity | minimum_income
enforce_minimum_goods_spending: true | false
```

Names are provisional; the independence is the requirement. The audit recommends a typed `CountryConfiguration.household_support` block with legacy-safe defaults and explicit loader/YAML wiring, rather than free-floating flags; see section 10. Minimum income can run with compulsory goods spending on or off. Retain the legacy configuration for matched comparisons. Support mode selects one transfer mechanism, so switching the spending-floor flag must not secretly reactivate old necessity/rent support or pay both transfers.

The income threshold is full family-adjusted SMIC. The optional minimum-goods amount remains a separate spending-policy parameter; enabling it must not automatically force spending equal to the new income threshold. If available resources cannot cover an enabled floor (including an asset-ineligible household), the agreed payment/shortfall rule must handle that case explicitly; the toggle cannot create unbooked money. No toggle default has been selected.

Removing forced spending means the spending flag is off, not that all fields named subsistence are set to zero. In either mode ordinary affordability constraints still apply.

Eligible-income ledger:


| Item | Draft treatment | Decision/status |
|---|---|---|
| Net employee income | Include; reuse already-taxed wage receipts | **After-tax cash-income basis confirmed** |
| Public pension and UB | Include once | Proposed; preserve observed-recipient allocation |
| Existing other cash social transfers | Include once before assessing the gap | Proposed; do not relabel aggregate D62 as a newly identified statutory scheme |
| Cash rental receipts | Include on a declared net/gross basis | Proposed; align with the model's existing tax treatment |
| Dividends, deposit interest and other actual monetary returns | Proposed: count cash receipts once, with a declared receipt/assessment date | Returns are flows; holdings are a separate eligibility stock. User expressed uncertainty about assets in income; recommendation is recorded, not a new approval of every return component. |
| Imputed rent and capital gains/revaluation | Exclude from a cash-income test | Proposed; behavioural total income is a different object |
| Financial assets | Opening liquid plus illiquid financial assets, before debt subtraction; eligible only if strictly below three months of household minimum income | **Confirmed by user, 2026-11-17** |
| Property wealth | Not financial assets; exclude from this financial-asset gate | Any additional property test would require a separate decision |
| New borrowing, asset-sale proceeds, mortgage proceeds | Exclude as financing | Proposed; preserve native financing ledgers |
| Rent paid, interest and principal service | Do not deduct automatically | Housing and disposable-income definition pending; principal is not an income deduction |
| This minimum-income payment | Exclude from its own means test | Required to avoid circularity |
| MPC experimental receipt | Exclude from assessed income; include once in spendable resources above the guarantee | **Confirmed by user**; counted case is optional mechanism sensitivity only |

**Assets versus asset income:** the total LFA+IFA stock is already tested against the quarterly cutoff. Do not add that same stock to cash income or assume it is spent each quarter. Recommended: include actual cash returns such as interest/dividends/rent as after-tax income; exclude revaluation, reinvested non-cash returns, loan receipts and asset-sale principal. The stock gate and cash-income gap are different tests. The user accepted after-tax cash income but expressed uncertainty about asset treatment; this distinction is the recommendation to review, not permission to impose an asset-drawdown/deemed-return rule.

“Existing benefits included” avoids paying the same gap twice. It does not identify which statutory programmes are embedded in the aggregate regular-transfer data. Do not subtract a guessed RSA allocation or automatically renormalise existing benefits to hide the new fiscal cost.

### 4.2 Timing: recover the paper's contract, then map the implementation

The user directed us to the paper/vault rather than asking for a fresh timing choice. The local manuscript (`consumption_micro_macro_paper/main.tex`, HEAD `2310287` at inspection) specifies:

- Lines 357–364: realise disposable income and update expectations, then choose consumption.
- Lines 377–381 and 410–423: restore feasibility, suspend consumer-credit payments before mortgage payments, and use the old residual support last.
- Lines 403–407: net-SMIC/CU calibration and period CPI indexation.

The vault `wiki/plans/2026-11-09-floor-redesign-design.md` also records this payment order and the old absence of rent arrears. `wiki/experiments/2026-11-04-consumption-input-timing.md` records beginning-of-period stocks and separate planning/replacement passes. Do not treat the paper's old last-resort spending subsidy as if it already defined the new pre-consumption income top-up.

For the new rule, retain current-period income assessment before authoritative consumption choice. Assess the new top-up before demand, include it in financing resources, and preserve the existing two-pass append/replace lifecycle. Do not introduce lagged-income eligibility or a new withdrawal lag merely because it is easier to implement.

The code still forms early forecasts before labour and settles some receipts after goods clearing. The paper's economic sequence therefore does not by itself certify the available exact cash vector at both code call sites. The first implementation contract must list which income components are known before each pass, update current wages/regular benefits after labour, and ensure once-only final booking. Late-receipt forecast errors are recorded against the frozen post-labour claim; do not recompute the payment or pretend future dividends were known at the early pass. Escalate only a substantive remaining contradiction after that source mapping, not the already-stated broad period sequence.

Use opening LFA+IFA for the asset gate as the current proposal, in line with the existing beginning-of-period stock contract. Explicitly verify whether the paper provides a more specific asset-test timestamp: it describes the old support mechanism, not this new eligibility test. The user has confirmed the asset measure/horizon, not an intraperiod reassessment rule. Both planning passes must use the same frozen eligibility stock unless explicitly decided otherwise.

### 4.3 Housing and payment priority: what needs deciding

These are two practical questions. First, is rent paid entirely from the minimum, or is there additional rent assistance? For example, a household with minimum income 1,500 and rent 1,000 has 500 left for everything else unless another benefit is specified. Second, if rent, scheduled loans and desired shopping together exceed available money, which payments are reduced or left unpaid first? These examples are illustrative amounts, not policy calibration.

Replacing Stage 5 also removes its rent-financing backstop. A fixed family minimum may be below contractual rent. Therefore income-only protection and unconditional no-rent-arrears cannot both be asserted without another mechanism.

Decide among: rent is paid from the minimum and shortfalls receive an explicit settlement treatment; a separately defined housing benefit/backstop; or a threshold defined after eligible housing costs. The latter two require their own eligibility/cap and fiscal record. Do not quietly restore `max(rent - resources, 0)` or subsidise arbitrary rent through an uncapped threshold.

Separately specify payment priority among contractual rent, debt service, desired goods and investment. Keeping the old sequence with a zero goods floor would prioritise debt over all voluntary goods; this is a policy choice, not a neutral deletion. Preserve ordinary liquidity constraints, debt identities, loan relief and distress recording. Any new rent-arrears path is a distinct increment after its rule is agreed.

### 4.4 Replace the behavioural-income subsistence bound: confirmed

The consumption formula uses logarithms and income denominators. The existing code can substitute a subsistence-level income when actual/assessed income is smaller. This makes the formula evaluable but pays the household no money and can hide an income increase from the consumption rule. For example, if the formula treats incomes below 1,500 as 1,500, an actual change from 1,000 to 1,100 leaves that input unchanged. This is different from the cash guarantee and the compulsory-spending toggle. The amounts here are illustrative.

**User decision: replace the subsistence-level input floor.** Remove that economic lower bound from behavioural-income reconciliation and the initial bridge. Use a separately defined positive-domain safeguard where logs/ratios require it; a candidate is max(raw income, epsilon), with an explicitly unit-consistent epsilon, not SMIC or the minimum-income threshold. The exact guard and scale must be specified and checked before implementation, not fitted to produce a desired MPC. Positive changes comfortably above the numerical guard must remain visible even below the old subsistence amount. Nonpositive/near-zero cases and guard activations remain reported; do not assert that a tiny denominator is economically harmless.

Three objects must have distinct definitions and diagnostics:

1. Cash-income threshold and actual transfer `M`, `T`.
2. Behavioural income and its rolling/smoothed denominator, including the transfer's intended effect.
3. Numerical positive-domain protection for logarithms and ratios.

The old SMIC-based behavioural-income bound is retired under this decision. Do not replace it with `M` or set every protective input to zero. Trace both planning passes, the initial bridge, historical smoothing, and permanent-income learning. Existing learning observes non-property income including social transfers; moving support earlier can change that signal. Specify whether the top-up belongs in that learning input and avoid adding it twice.

No intercept change, wealth rescaling, residual redraw/re-anchoring after support, or consumption refit is part of this plan. The audit found that the initial-state residual is created lazily on the first target evaluation: identical input caches alone do not freeze it. Use a common initialised checkpoint with the native residual already created, then clone for policy arms; verify identical residual/layout/RNG state and do not call reset or set_run_seed after cloning. If a first-period or policy-consistent initialisation is required, specify and validate that bootstrap separately. See section 10.

### 4.5 Assessed income: follow real-world practice (open research)

User direction (2026-11-21): assessed income should follow what the real benefit does. The reference is the French RSA, the means-tested top-up that the SMIC/CU design most closely matches. **The statutory points below are unverified and must be checked against the primary source (CASF articles L262-2/3 and R262-6 onwards, or the CAF guidance) before increment 2:**

- RSA assesses average monthly resources over a three-month window. **User decision (2026-11-21):** the window is fixed in months and includes the current period: `Y_assessed = mean over the last ceil(3 / period_months) periods, current period included`. Quarterly model: the current quarter only. Monthly model: the current month plus the two before. This is consistent with D4: the current period enters through the post-labour assessment, then the payment is frozen. Earlier periods in the window use realised values. If history is shorter than the window (start of a run), average over the periods that exist.
- Earnings, unemployment benefits, pensions and most other transfers count. Family benefits partly count.
- Capital income counts. Savings that produce no income count at a notional yield (VERIFY the rate).
- Housing: a fixed "forfait logement" is deducted when the household pays no rent or receives housing benefit (VERIFY).
- Debt service is not deducted.
- RSA has an earnings-disregard (prime d'activité, a separate benefit), so in practice withdrawal is less than euro for euro. D6 stays at euro for euro as confirmed; note this as a simplification.

Model mapping (to be completed): for each item, give the model field, its timing, and whether it is net of tax. Then the user decides on each departure from practice.

## 5. Accounting and lifecycle invariants

The new top-up has an explicit household component, proposed `income_minimum_income`, separate from regular transfers and historical `stage5_subsistence_support`. Exact schema names are provisional. Carry expected, financing and realised amounts only where required; a forecast is not another booked cash receipt.

- Household social income equals pension + UB + regular other transfers + new top-up; old Stage 5 is zero in the replacement mode.
- Sum of household top-ups equals the government's new transfer expenditure. Follow that amount through `compute_deficit`, debt/cash financing and bank deposits; no second household deposit credit when wealth accounting already carries the receipt through saving.
- Pre-credit and post-labour entitlement calculations are idempotent within a period. Final settlement books once; repeated settlement fails or is explicitly idempotent.
- No top-up depends on desired goods, actual spending, borrowing, forced liquidation or an ex-post budget residual unless an explicitly separate housing instrument requires it.
- With compulsory spending off, desired goods below the historical floor are allowed and paid goods never exceed desired goods because of a floor. With it on, an explicit floor may raise spending only through the defined, funded settlement rule. Funding constraints and shortfalls remain visible.
- Entitlement is zero at or above the financial-asset cutoff. Export liquid assets, illiquid assets, their gross sum and the quarterly cutoff. Test equality, either side of the cutoff, high debt (which does not lower assessed assets), and monthly versus quarterly timesteps; do not smooth the discontinuity silently. Opening-date assessment remains the proposed timing convention, not a newly confirmed user decision.
- Goods VAT, imputed-rent accounting, household investment tax, property-sale financing, liquidation fees, principal and interest identities remain explicit.
- Rationed goods do not cancel an income-based entitlement. Unspent top-up remains available as saving through the native cash/wealth route.
- Reset, deepcopy/clone and cache lifecycle clear or restore each new per-period carrier correctly. Do not mutate the common initial cache or consume a second RNG draw for a policy calculation.
- Represent historical necessity support and minimum income separately in saved outputs. Update regime classification rather than labelling every new recipient “floor-bound”.

Keep a legacy comparison mode while developing the replacement. Configuration must prevent both support mechanisms being paid together. The final implementation exposes the independent minimum-goods toggle requested by the user. Turning it on or off is a supported configuration, not merely a temporary diagnostic. The default remains to be selected.

## 6. Implementation increments and acceptance gates

Do not start an increment with unresolved choices it depends on. Run graph impact analysis before changing its symbols, then agree the exact increment contract. No automatic PR/push/merge is implied by this planning task.

| Increment | Complexity | Deliverable | Acceptance gate |
|---|---|---|---|
| 0. Finish design and source ledger | Medium | Verify existing SMIC/family source wiring; complete eligible-income/assessment field mapping, housing settlement and numerical safeguard; enumerate every writer/reader and fiscal entry | No contradictory guarantee claims; worked household examples cover income loss, employment change, high rent and extra income |
| 1. Pure entitlement and configuration | Medium | Independently callable SMIC/family threshold and asset-gated top-up, exclusive support modes plus an independent spending-floor toggle; inactive by default during implementation | Scale and period conversion exactly once; finite inputs; threshold/full-withdrawal boundaries; family/age changes; negative income; asset cutoff below/equal/above; both toggle states; benefit and shock exclusions |
| 2. Numerical-income contract and histories | High | One unit-consistent guard through initial reconciliation, monetary bridge and active prepared-input adapter; explicit legacy comparison policy | Below-SMIC positive changes remain visible; aligned history/PI identities; no-positive-income and unit/price sensitivity; no hidden residual re-anchoring |
| 3. Planning and transfer accounting | High | One Country-owned per-period assessment record; early and post-labour entitlement replacement; financing and final government/household booking | No self-inclusion or duplicate D62/top-up; financing survives known-receipts refresh; exact once-only fiscal/deposit counterpart; lifecycle checks |
| 4. Independent spending settlement | High | Income-support mode has no extra residual subsidy; independent goods-floor flag; preserve debt/credit settlement and implement the agreed shortfall contract | Low voluntary spending stays low with floor off; rent/floor deficits never create cash; loan and arrears identities; both call paths consistent |
| 5. Diagnostics and MPC definitions | Medium | Income/needs coverage, recipients, gross/net receipts, tax/benefit withdrawal, housing and payment failures, fiscal costs, revised regimes | Do not divide by zero net receipts; preserve all-household impact sample; disclose cumulative selection; distinguish numerical/input bounds from transfer eligibility |
| 6. Registered validation and report | High | Mandatory smoke, controlled paired runs, all-arm reporting, manuscript/deck provenance | Mechanical gates first; no tuning to an MPC; report adverse outcomes and unresolved economic limits |

Increments 2–4 are interdependent at integration. Implement the pure numerical-input foundation before integrating the entitlement; keep both inactive under the legacy default until the whole vertical path passes. Keep helper/config work inactive until the complete replacement is internally consistent; do not publish an intermediate mode with both old support and the new entitlement or without debt settlement. Each increment needs its own concrete pre-code specification rather than treating this overview as an executable patch list.

## 7. Validation design to register after policy choices

Do not run simulations under this draft. Freeze source/config/cache/environment and the same initial HFCS cohort before registering a run.

Primary contrast to register: existing support plus compulsory spending versus agreed minimum income with compulsory spending off. Also compare minimum income with spending on versus off to exercise the user-requested independent toggle. Neither configuration is selected as the eventual default here. Use identical initial states/RNGs and a fixed inherited residual. Report baseline paths as well as responses. This is a policy transition at the same opening state, not two independently reinitialised economies.

To attribute mechanisms, separate: timing/eligibility of support, removal of forced goods spending, and replacement of behavioural-income protection. Only use hybrid diagnostic arms if their accounting and housing rules are fully defined; never create an infeasible arm merely to complete a factorial table. Show any policy-driven change in disposable income rather than forcing equal aggregate support or consumption by calibration. A fiscally matched comparison, if desired, must be a separately defined policy.

For a contemporaneous fully binding assessment, an eligible additional receipt `dX` gives `dT/dX = -tau`; net policy receipt changes by `(1-tau)*dX`. With tau=1 it is zero. If exempt, the top-up is unchanged and the full receipt is available. With lagged eligibility, contemporaneous and later withdrawal differ. These are mechanical predictions to test, not model failures.

Report MPC per gross transfer and, when defined, per net policy receipt (`experimental receipt + change in minimum income`). Report those net receipts separately from general-equilibrium changes in wages and other income. A zero or near-zero net denominator is undefined/unstable and must be flagged rather than clipped into an MPC. The confirmed primary MPC intervention is exempt and therefore on top of the guarantee. A counted experiment, if later requested, is a different intervention.

Suggested sequence, not a run registration:

1. Synthetic boundary/accounting/lifecycle checks and relevant existing regressions in the required project `.venv`.
2. One-seed shortest-horizon smoke on the same host; finite core series, exact treatment/control pre-treatment state, once-only transfer/fiscal postings and all rent/debt/cash identities must pass.
3. Paired small-seed pilot with all predeclared arms and the exempt MPC shock; a counted shock is an optional separately declared mechanism check. Record resource and duration estimates before a larger campaign.
4. At least 30 paired seeds for conference comparison, with declared transfer size/date, fixed euro amounts where intended, and four-quarter horizon. Use common initial groups and report each policy's own groups/regimes as a sensitivity. Extend horizon for lagged eligibility withdrawal if required.

Core outputs: assessed versus realised income shortfall to M; recipient and coverage shares; fiscal cost; desired/paid goods and forced-spending increment (zero when the spending toggle is off); rent due/paid/unpaid and debt distress; assets drawn down; numerical-bound shares; gross/net transfer MPCs by fixed initial income, LFA and all-debt NLA; aggregate consumption/income/output/price paths and uncertainty across seeds. Supply accounting tolerances and numerical/economic abort criteria in the eventual run specification.

## 8. Open decision register

| ID | Decision | Present status |
|---|---|---|
| D1 | Replace old transfer; independently toggle compulsory minimum goods spending | **Confirmed user update, 2026-11-17; supersedes permanent floor removal.** |
| D2 | Threshold: SMIC times household family coefficient | **Reuse existing net-SMIC and family scale.** Recovered manuscript basis: 2014 EUR1,128.70/month, weights 1/.5/.3, CPI indexation; existing reader supplies the dated model input. Full SMIC is the new income threshold; half-SMIC remains only the historical spending norm. |
| D3 | Financial-asset gate and assessment income | **Confirmed: liquid + illiquid financial assets, before debt subtraction, strictly below one quarter of household minimum income.** After-tax cash income confirmed. Opening asset date is proposed; exact cash-return components and negative-income handling must be mapped. Do not count financial-asset holdings as income. |
| D4 | Current-income assessment and payment freeze | **Confirmed: freeze payment after post-labour assessment, before consumption.** Assessment window: 3 months including the current period, converted to model frequency (§4.5). No late actual-income true-up/clawback. **Next period (confirmed 2026-11-21): any saved part counts only through the financial-asset stock. The previous payment is not added to assessed income.** |
| D5 | Housing shortfalls | **Direction (2026-11-21, user leaning, not final):** missed rent → move to the existing social housing (house ID −1) → return to market rent once affordable. The subsidy is in kind: the social-rent place. No cash housing subsidy. Consumer-before-mortgage seniority stays. Still open: arrears booking, move timing, re-entry criterion, and whether owners in mortgage distress are included. |
| D6 | Benefit withdrawal and MPC receipt | **Confirmed: euro-for-euro withdrawal; MPC receipt exempt and paid on top.** No taper selection needed. |
| D7 | Behavioural-income bound and learning treatment | **Confirmed: replace the old economic input floor.** **Guard contract per §10.5 agreed 2026-11-21.** Define a separate positive-domain safeguard and test visibility of income changes; map top-up into learning once. |
| D8 | Fiscal closure | **Confirmed 2026-11-21:** keep the existing tax/debt closure and record the incremental cost. No budget-neutral retuning. |
| D9 | Initial policy activation and residual | **Accepted 2026-11-21.** Audit correction: recommend a common completed-period checkpoint with the lazy residual already initialised, then clone policy arms. First-period activation needs a separately validated common-bootstrap contract; it cannot be assumed to preserve epsilon. |

Next: (1) research the RSA resource base and map it to model fields (§4.5); (2) verify that households can move into social housing during a run and specify the D5 transition; (3) choose the guard's numeric value under the §10.5 contract; (4) increment 1. Do not reopen the confirmed SMIC/family, asset horizon, withdrawal or MPC-exemption choices. This draft reuses an existing model calibration and makes no new claim about current statutory benefit entitlement.

## 9. Revision history and planning verification

- Draft 2, 2026-11-17: user requested independent minimum-consumption toggling, selected SMIC times the family coefficient, and restricted eligibility to households with financial assets below the minimum-income cutoff. Updated equations, configuration separation, tests and decision register. Added plain-language explanations of housing/payment priority, benefit withdrawal and the numerical income floor. Asset definition and cutoff horizon asked explicitly; no answer assumed.

- Draft 3, 2026-11-17: user confirmed total financial assets (liquid plus illiquid, before subtracting debt) and a cutoff of one quarter of household minimum income. Fixed quarterly cutoff remains three months even if the model timestep is monthly. Updated formula, ledger and boundary checks; no runtime implementation.

- Draft 4, 2026-11-17: recovered paper/vault SMIC, family weights and intended income/payment sequence; traced consumption-notebooks DH0002 mapping. User accepts after-tax cash income, confirms euro-for-euro withdrawal and an exempt MPC receipt on top, and directs replacement of the old behavioural-income bound. Assets-as-income distinction recorded as a recommendation; no asset-drawdown rule introduced.

- Draft 6, 2026-11-17: user confirms the post-labour/pre-consumption payment freeze and next-period eligibility effect. Scoped missed rent → temporary social housing → affordable matched rental, with housing subsidies as the alternative; discovered existing social-rent and occupancy mechanisms. Interperiod income-versus-saved-assets interpretation asked explicitly.

- Vault save, 2026-11-17: user requested saving the plan in the vault. Retained the complete draft and recorded the saved-assets recommendation as proposed, not approved. The original temporary worktree is unavailable; this vault record is the re-entry source.

### Index note

Fresh codebase-memory index `inet-minimum-income-design` was built on this worktree (9,250 nodes, 49,133 edges). On the user-requested discovery audit, a dedicated GitNexus index was successfully rebuilt and registered as `inet-minimum-income` using `analyze /private/tmp/inet-minimum-income --skip-git --name inet-minimum-income --index-only --force --workers 2`. It contains 11,565 nodes, 23,018 edges, 599 clusters and 300 flows. The earlier failed registration is superseded. Source and instruction files were not injected or changed by indexing.

GitNexus query/context/process/cypher and upstream impact checks were used, with exact-source confirmation from the codebase-memory graph. Empty/low graph results are not proof of no risk: the dynamic `self.financial_feasibility.settle_consumption_floor` call is present in source but absent from GitNexus's production caller list. `reconcile_behavioural_income`, `monetary_income_moments` and `current_income_belief_learning_inputs` are HIGH-impact graph targets. Re-run impact analysis at implementation time on the actual symbols being edited.

Five existing test modules around floor units, behavioural interest, prepared inputs, adapter histories and support settlement passed: **182 passed, 2 pre-existing SyntaxWarnings, 15.99 seconds** in the required `.venv`. These characterise the old behaviour; they do not validate a new policy. No runtime source edits, refits or simulations were performed. Evidence is mirrored in vault `outputs/audits/2026-11-17-minimum-income-plan/` and the companion branch `AUDIT.md`. `detect_changes(scope=all, worktree=...)` found zero changed code symbols; the new Markdown files are untracked, so they were also checked separately with git status and whitespace/link checks.


## 10. Audited implementation contract and recommended structure

This section supersedes vague “wire the transfer” and “replace the floor” wording above. Names below are proposed interfaces, not existing runtime APIs. Confirmed economic choices remain unchanged; recommendations are not disguised user decisions.

### 10.1 Separate three mechanisms

Use one typed configuration block under Country with (a) transfer mode, (b) the independent goods-spending-floor flag and (c) explicit income-input policy for the legacy comparison versus the numerical replacement. Reject misspelled/unsupported options instead of relying on Pydantic extra-field handling. Add the field to `CountryConfiguration`, resolve `paper_parameter_file`/`paper_parameter_ref` through the existing loader, and record effective settings in experiment metadata. Keep existing configurations bit-compatible by default during implementation.

Provide a pure `minimum_income` calculation module and a Country-owned per-period assessment record. The calculation accepts already-prepared arrays and returns eligibility, assessed income, threshold, entitlement and diagnostic flags; it does not alter deposits, consumption, government totals or RNG state. Reuse the existing SMIC source selector, CU producer, CPI convention and period/population conversion. Do not overload the existing goods-floor block with full-SMIC income semantics or a new collection of hardcoded country constants. No `none` transfer mode is needed for the requested comparison unless a later experiment requires it.

The minimum-income controller must not depend on whether CACF's aligned/learning path is enabled. Country owns timing and receipt composition; the consumption rule consumes the resulting income. Do not reuse the consumption adapter's opening snapshot as the sole eligibility owner: that adapter is only activated on some consumption modes. A small immutable eligibility snapshot can read the same opening arrays, tagged with the existing period/layout identity.

### 10.2 Financial-asset and money-unit contract

The raw formula `LFA + IFA` needs one accounting qualification: model liquid balances may represent an overdraft. Debt must not lower a **gross** financial-asset test. Recommended implementation is `gross_assets = maximum(LFA, 0) + IFA`, validating that IFA is finite and nonnegative. Keep the original signed LFA for budget and net-wealth calculations; do not mutate household wealth. Example in ordinary quarterly euros: LFA=-500, IFA=3,500, single-adult threshold=3,386.10. Raw netting gives 3,000 and incorrectly admits the household; gross assets are 3,500 and it is ineligible. This is an interpretation of the confirmed gross-asset rule, not a new means-test threshold.

The base source is already population-expanded by `Country._select_net_smic_base`; downstream helpers must not multiply it by scale again. The income threshold uses model-period months; the asset cutoff always uses three months. Freeze eligibility assets before current-period transfer, borrowing, property transactions, liquidation and portfolio changes. Opening-date assessment is the recommended technical contract, not a more specific timing statement found in the paper. Equality is ineligible. Compare at the same observed planning CPI in both passes; do not let an end-of-quarter CPI revision silently change an already authorised claim.

### 10.3 Explicit cash-income and receipt sequence

Use an explicit assessment helper (proposed `assess_minimum_income_cash_income`) with named monetary components, not `household_gross_cash_income`, raw balance-sheet assets, behavioural total income or `financing_income` wholesale.

| Component | Early/current assessment source | Late actual source / restriction |
|---|---|---|
| Net wages | existing expected employee component early, known `nominal_employee_income` aggregate after labour | realised `income_employee`; do not apply income tax twice |
| Pension / UB / regular transfers | existing expected components, then `compute_realised_household_social_income_components` as available | realised public-pension/UB/other series once; no renormalisation of the D62 budget |
| Rental income | `income_rental`, already after tax in `compute_rental_income` | keep cash rent separate from imputed housing services |
| Cash dividends | `compute_expected_dividend_income` before settlement | `income_dividend_distributions` after dividend settlement; do not count reinvested IFA return as cash |
| Deposit interest receipts | positive receipt side of the existing signed `compute_interest_paid_on_deposits` convention on the declared opening basis | realised deposit-interest receipt; not borrower interest or principal |
| New income top-up / old Stage 5 | exclude from eligibility income | book only the selected transfer mode |
| MPC experimental receipt | exclude from assessment; include once in planning/financing and final cash income | do not subtract it again if eligibility was assembled from components that already exclude the staged hook |

Property/financial cash returns remain the recommended interpretation of after-tax cash income; no deemed return or asset drawdown is introduced. Payment freezes on the authoritative post-labour assessment (10.4); late receipt surprises are diagnostics, not entitlement revisions.

Recommended owner order:

1. Capture the eligibility snapshot at the beginning of the period independently of CACF's adapter.
2. Update regular benefits and base expected income components without the new top-up.
3. Compute `T_expected` from that component snapshot, and add it once to expected social income/total and the first `financing_income` before `_set_household_target_demand`.
4. After labour, rebuild base expected components, assemble known receipts, replace—not increment—the authoritative entitlement, and add that same claim to expected social income, current financing and the income supplied to target replanning. `refresh_known_household_receipts` currently subtracts/rebuilds social components and discards the support slot; it must explicitly carry the new claim across this operation.
5. Settle ordinary financing and payments with the declared available transfer. Do not rebook loans or replay asset sales on the second pass.
6. At late income settlement, book `income_minimum_income` once, update household social-income totals, calculate final cash/wealth through the existing saving route, and record a separate government `total_minimum_income`. Extend `CentralGovernment.compute_deficit`'s explicit component sum; writing only `total_household_social_transfers` is insufficient. Do not credit household deposits a second time.

Avoid extending the existing four-position social-income tuple casually or reusing its Stage 5 slot to hide the new component. Prefer a named internal component record, or a small separate top-up accessor with explicit changes at every sum/consumer. Preserve any compatibility wrapper deliberately. Test every planned and realised consumer, including `compute_realised_household_social_transfers`, not only the recording function.

### 10.4 Confirmed payment contract: freeze before consumption

**User decision:** authorise the payment after the post-labour assessment, before consumption, and freeze it for that period. This supersedes the earlier unresolved final-income-versus-assessment choice.

The early payment is a planning forecast, not an additional cash transfer. Recompute and replace that forecast after labour using the agreed current cash-income assessment, freeze `T_authorised`, use it in financing and consumption, and book exactly `T_authorised` in household income and government expenditure. The final settlement accessor must not recompute the gap from realised spending or late actual income.

Record `E_actual - E_assessed` separately. Unexpected dividends or other late receipts affect cash/saving and next period's state, but do not cause a within-period benefit clawback, additional guarantee true-up, or retroactive goods finance. The guarantee is against the authorised assessment, not an exact floor on every final realised income outcome. A receipt shortfall can still trigger the separately specified housing/payment treatment.

The user requests a next-period eligibility effect but answered 'not sure' when asked to distinguish saved assets from counting the whole prior payment as income. **Recommended rule, not yet a confirmed user decision:** count any saved portion through next period's ordinary financial-asset stock; do not add the previous top-up again to current assessed cash income. Example: with a 1,500 minimum and 1,000 other income, pay 500; saving 200 raises next period's tested assets by 200, while spending the entire payment creates no direct saved-asset increment. Subject to other income/assets, support continues while the asset gate passes. Counting the full previous top-up as extra income instead can generate 500, 0, 500, 0 payments despite constant other income. Preserve the current-period payment freeze and ordinary future asset assessment; do not create permanently exempt shadow wealth. The request to save this plan is not treated as approval of an unresolved eligibility rule.

### 10.5 Replace every active economic income floor

The active FRA path uses prepared inputs; `_geometric_average_income` is a fallback when those inputs are absent. Its apparent local importance must not drive the implementation.

| Layer | Source to change / inspect | Current floor operation |
|---|---|---|
| Initial and per-pass behavioural income | `Country._record_behavioural_income`, `reconcile_behavioural_income` | `max(raw, subsistence)` and the initial `income_bridge_current` floor |
| Monetary PI / uncertainty bridge | `Country._set_household_target_demand`, `Households.current_income_belief_learning_inputs`, `monetary_income_moments` | current and projected monetary incomes floored at S before log moments; affects uncertainty as well as PI |
| Prepared income histories | `ConsumptionInputAdapter.collect_inputs`, `IncomeHistories`, `prepare_inputs` | reads the complete `subsistence_consumption` history and takes `log(max(B_t, S_t))` before smoothing |
| Non-prepared compatibility path | `CreditAugmentedConsumption._geometric_average_income` | no-positive-history fallback to subsistence income |
| Learning observations and saved diagnostics | `compute_non_property_income`, realised learning update, input/bridge flags, output extractors | ensure the new top-up is observed once; preserve raw observations and separately named numerical-guard flags |

Introduce a dedicated numerical-guard history in declared income units, not a zeroed goods-floor series or a second copy of M. Migrate the immutable history records, unit/period/layout checks, collection, smoothing and monetary bridge consistently; the goods-spending toggle must not change the numerical guard. Preserve the identity between the smoothed B denominator and monetary PI rebase across both passes. Old history rows before policy activation retain their historical treatment; do not rewrite them silently.

A tiny epsilon alone is not evidence of safety. The old fallback explicitly guards against near-zero wealth-ratio explosions (issue #90). Keep raw values, flag all activations, verify annualised ratios and growth-guard/clipping incidence, and test zero/negative/no-positive-window cases, the asset-ineligible cash-poor case, common money rescaling and CPI scaling. No new numeric epsilon is selected by this audit. Its contract is a technical implementation prerequisite, not a policy calibration exercise.

### 10.6 Spending toggle and housing shortfalls

Keep `settle_authoritative_household_payments` as the owner of consumer/mortgage service, early repayment, relief, loan stocks and distress. Extract/select a pure allocation policy behind that owner. Existing goods-floor diagnostics should not decide minimum-income eligibility.

For minimum-income mode, the allocator takes resources that already include the authorised top-up, an effective goods floor (`F` when on, zero when off), rent due, debt service, desired goods and investment. It must not calculate another residual transfer. Preserve the established consumer-before-mortgage suspension order and last priority for investment in the funded domain.

There is a real infeasible boundary: an asset-ineligible household, or one with rent above its available cash, may fail to fund rent or an enabled floor. Return explicit unpaid-rent and unmet-floor quantities from the pure policy; do not use `maximum(remaining_cash, 0)` to conceal a missing payment. Their booking/tenancy consequences remain a required policy contract because the inherited model assumes rent is paid and credits landlords. If a rent shortfall is permitted, tenant outflow and landlord after-tax receipts must reconcile; a diagnostic alone is not rent-arrears accounting. If it is forbidden, define the permitted funded remedy—do not silently reinstate Stage 5.

Requested run configurations are the unchanged legacy reference and minimum-income with spending on/off. Any other combination must have defined semantics before it is advertised as supported. Default legacy behaviour remains intact during implementation; no production default for the new switch is inferred.

### 10.7 Common-state comparisons, MPC exemption and outputs

A common cache is insufficient because `CreditAugmentedConsumption.compute_target_consumption` lazily fills `_household_epsilon` from the first evaluated inputs. Changing the numerical-floor policy before that point changes the anchor. Recommended validation uses a common completed-period checkpoint with epsilon/layout initialised, then clones into policy arms without `reset`/`set_run_seed`; policy activation and any additional policy-specific warm-up are registered explicitly. No hand assignment to private epsilon arrays and no premature `_epsilon` call. Verify existing histories, residual, opening assets, RNGs and household identity before the intervention. A first-period alternative needs a native bootstrap design rather than an assertion that the cache fixes epsilon.

The MPC receipt is exempt from the current assessment and paid on top. A saved windfall may increase next period's financial assets and cross the confirmed asset cutoff, withdrawing later support even though the original receipt was income-exempt. The plan must report this separate channel. Current-receipt exemption does not imply permanently exempt shadow wealth; full future shielding would be an additional policy requiring an explicit decision.

Add actual/assessed income, gross eligibility assets/cutoff/mask, planned/authorised/settled top-up, fiscal counterpart, floor switch/amount, numerical-guard flags, rent/floor/payment shortfalls, and policy-activation provenance to the selected-output path and MPC window metadata. Update supported versus floor-bound regime labels and do not assume new recipients are necessarily floor-bound. Preserve all-household impact and separately identified cumulative samples.

### 10.8 Revision baseline and integration gate

At audit time local `main` has advanced to `517cda3` while this branch remains based on `a340e124`; `origin/main` was separately at `4af7f4e`. The observed local-main diff touches housing output bookkeeping and selected MPC/IRF output helpers (`selected_output.py`, `mpc.py`, `irf.py`, `mpc_analysis.py`), not the core support/income sources audited here. Before implementation, reconcile the desired main baseline in this isolated branch and refresh the graph/source pins. Do not overwrite the newer selected-output path with the older runner. No rebase or merge is performed by this audit.


## 11. Housing extension: missed rent and temporary social housing

### 11.1 User direction and existing mechanisms

The proposed primary route to develop is: a household cannot pay market rent, misses the payment, moves to social housing, and returns to a market rental when it can afford one. Housing subsidies are the alternative to investigate. This is permission to flesh out these designs, not a selection of a statutory subsidy schedule or a grant that automatically covers any rent shortfall.

Source discovery on the pinned branch:

- `Households.compute_rent` (`households.py:3395–3458`) represents social housing through `Corresponding Inhabited House ID == -1`. It computes a rent payable to government, not free housing. The call supplies `Number of Adults` as the social-rent size operand despite the callee's `current_household_size` name; verify the intended demographic convention before reuse.
- `DefaultSocialHousing.compute_social_housing_rent` (`social_housing.py:87–110`) sets social rent as a configured fraction of the unemployment-benefit level times that size operand. It does not use the new SMIC-family guarantee automatically.
- `HousingMarket.process_housing_market_clearing` (`housing_market.py:494–704`) already vacates homes and can move sellers into the social-housing state, and records successful rental moves. These state transitions can inform a shared move helper; they are not an existing missed-rent eviction mechanism.
- `Households.process_housing_market_clearing` calculates the period's rent before goods/payment outcomes. A missed-rent transition discovered later therefore needs explicit scheduling; do not retroactively replace the rent due with cheaper social rent for the same elapsed quarter.
- The existing representation is a fallback state, not proof of a finite social-housing stock, a waiting list or an entitlement allocation process. Reusing that fallback is a declared model simplification, not a claim that real-world social housing is unlimited.

### 11.2 Proposed transition and accounting contract

Recommended timing for the candidate route is to record the missed current rent and queue the move for the next period's housing stage, before its new rent accrues. This avoids undoing already-cleared housing transactions and current-quarter goods plans. It is a recommendation, not an agreed eviction lag.

The increment must define and test:

1. Separate rent due, rent actually paid and unpaid rent. A household cannot be credited with paying rent it missed; its landlord cannot receive that missing cash as rental income. Distinguish any receivable/accrual accounting from cash available for consumption.
2. Whether unpaid rent remains a tenant liability/landlord receivable after the move, is written off, or is paid by government. Moving does not forgive it automatically. Define rent-arrears service relative to existing consumer/mortgage debt; do not add rent debt to a bank loan balance.
3. On entry, clear old property occupancy, update household property/tenure links and both sides' renter lists, and make the old dwelling eligible for the agreed relisting process. Do not transfer ownership or alter mortgage principal as a side effect of a tenant's move.
4. Start social rent at the move's effective date and book actual government rent receipts. Check that the social rent itself is affordable under the frozen income guarantee; otherwise another explicit shortfall rule is needed.
5. Re-entry uses the existing rental search/matching path, subject to a specified affordability test. Affording a rental bid does not itself create an available dwelling: leave social housing only when a feasible market rental actually matches and completes. The affordability base must state treatment of rent arrears, debt service, goods needs and the optional spending floor.
6. Specify whether the first missed rent triggers the move, how partial payment is treated, and how a household already in social housing is handled. Do not introduce an arbitrary multi-quarter delay or a new calibrated affordability percentage without a source/decision.

Report transitions into/out of social housing, duration, rent due/paid/arrears, landlord cash-income losses, government social-rent receipts, vacant/relisted properties and any unmet goods floor. Apply graph impact analysis before edits to rent, occupancy, landlord-income and government-revenue symbols; they are more than a consumption-module change.

### 11.3 Housing-subsidy alternative

Before proposing a numeric schedule, compare an explicitly bounded housing allowance against the transition route. Declare eligibility, eligible rent, benefit cap, payment recipient, interaction with the SMIC top-up and the asset test, and fiscal booking. State whether the allowance counts as cash income in minimum-income assessment; otherwise a one-for-one withdrawal can cancel its intended housing protection.

Keep housing subsidy expenditure separate from minimum-income expenditure and from government social-rent receipts. Do not choose the subsidy level to preserve legacy consumption or to improve MPCs. Empirical/statutory calibration would need verified external sources and a dated policy definition; none is selected here.

### 11.4 Implementation ordering

Keep the confirmed payment-freeze/entitlement work modular. Add a separately specified housing-shortfall increment covering both the current missed-payment cash ledger and subsequent occupancy changes; do not implement only an eviction flag. Reuse the existing social-housing representation and rental matching where their invariants fit, rather than inventing a parallel housing market. Decide the route, arrears treatment and transition dates before enabling income-only support in economic runs that can encounter an uncovered rent shortfall.
