# Minimum income replacing Stage 5 subsistence support: design and implementation plan

Revision: draft 8, 2026-11-21. Consolidated rewrite. Draft 7 and earlier drafts are in the git history of this file.
Branch: `feature/minimum-income` (draft PR #213). Base: INET `origin/main` `38e9be92`.
Code verified on that base with GitNexus (index `inet-minimum-income`, built 2026-11-21) and grep. Symbols, not line numbers, are authoritative; line numbers are given for `38e9be92` only.

---

## 1. Objective and scope

Replace the expenditure-contingent Stage 5 "necessity support" with an RSA-style, income-tested cash top-up. The consumption rule then chooses spending; the guarantee insures income, not a level of spending.

In scope:
- a new transfer, the minimum income top-up T;
- an independent on/off switch for the compulsory goods floor;
- replacing the subsistence floor on behavioural income with a technical numerical guard;
- a housing route for renters who cannot pay rent (existing social housing).

Out of scope: changes to pensions, unemployment benefit or other transfers; HFCS opening allocation; wage setting; credit contracts; fiscal closure; consumption-rule estimates.

Why it matters for the paper: today a zero MPC at the bottom can come from compulsory spending or from the artificial income floor. After the change, a zero response has one structural cause: euro-for-euro withdrawal of the top-up. Do not choose any parameter to change the shape of the MPC profile.

## 2. Decisions (all user-confirmed, 2026-11-17 to 2026-11-21)

| ID | Decision |
|---|---|
| D1 | Stage 5 support is removed in the new mode. The compulsory goods floor becomes an independent switch (on or off). The legacy mode stays available for comparison. The two support mechanisms can never be active together. |
| D2 | M = 2014 RSA single-person amount, €499.31/month (décret 2013-1263, from 2014-01-01), × household consumption units (existing weights 1 / 0.5 / 0.3). Indexed like other transfers by the active benefit setter (§3.1) (CPI lag: §3.1). |
| D3 | Assessed income Y uses the previous period's realised values (t−1), the quarterly counterpart of RSA's m−4..m−2 window. Monthly model: mean of t−4..t−2. |
| D4 | Y = net wages + unemployment benefit + pensions + other transfers + rent received after tax + notional asset income. Notional asset income = 3%/yr of gross financial assets (liquid + illiquid, debt not subtracted). It replaces actual deposit interest and dividends. Housing wealth does not count. |
| D5 | T = max(M − Y, 0), euro-for-euro withdrawal. Computed once at the start of the period and fixed for it. No true-up or clawback within the period. |
| D6 | Any saved part of T affects the next period only through the financial-asset stock (via the notional 3%). The previous T is not added to Y. |
| D7 | The MPC experiment receipt is excluded from Y and paid on top. |
| D8 | Renter with unpaid rent → moves into existing social housing in the next period. The landlord absorbs the unpaid rent. Exit through the existing market bid rule. Social rent rule unchanged (0.25 × scalar UB × adults). |
| D9 | Existing fiscal closure kept; the cost of T is recorded separately. No budget-neutral retuning. |
| D10 | Policy comparisons clone one shared, fully initialised checkpoint (lazy residual already created). The shared checkpoint must be built in legacy_stage5 mode; policy arms switch mode by reset. A minimum_income-built country cannot be reset to another guard basis. |
| D11 | Numerical guard G = 1% of M_household per period, separate from the goods floor. |

Stated departures from RSA practice, for the paper: notional 3% applies to all financial assets (RSA: only to assets that pay nothing, CE 14 June 2017 n°401637); no forfait logement; no neutralisation of lost income; no earnings disregard; M uses the model's CU scale, not the RSA schedule. Level of M equals the 2014 RSA.

## 3. Policy definition

All amounts per household per period, nominal, in model monetary units (population-expanded, like `net_smic_base`).

### 3.1 Guaranteed level M

```
M_t = rsa_base_monthly × scale × time_unit × CU_h × I_t × (CPI_{t−1} / CPI_0)
I_t = total_other_benefits_t / total_other_benefits_0        (real level ratio produced by the active setter)
```

- M uses the last realised CPI because period-t CPI is computed after goods clearing; M therefore lags other transfers by one period of inflation (close to RSA practice of revaluing on past inflation).
- `scale` is the same population expansion applied once to the SMIC in `Country._select_net_smic_base`.
- `I_t` follows any setter. Under `regular_transfer_indexation: average_wage`, `CentralGovernment.update_benefits` sets `total_other_benefits = initial × _contract_wage_index_ratio()`, and `_firm_contract_rates` is a real rate, so `I_t` is the real contract-wage index. Under the constant setter, `I_t = 1` and M is CPI-indexed only.
- `CU_h` = `Households.states["Consumption Units"]` (`_current_consumption_units`), updated with household composition.
- Base year: the 2014 value requires the model's `initial_year == 2014`. Validate this; any other start year must supply its own RSA value.

### 3.2 Assessed income Y and top-up T

At the start of period t (before credit planning), with t−1 realised series still `current`:

```
Y_h = income_employee + income_unemployment_benefits + income_public_pension
      + income_other_social_transfers + income_rental
      + r_n × (liquid_financial_assets + illiquid_financial_assets)
r_n = 0.03 × time_unit / 12
T_h = max(M_h,t − Y_h, 0)
```

- Use the component series, not `income_social_transfers`, which includes last period's T and Stage 5. Use component series, not `income`, which includes the MPC shock (`_apply_household_income_shock`). This gives D6 and D7 by construction.
- Asset stocks are the closing t−1 stocks (`Households.update_wealth` runs after income booking in `update_realised_metrics`), so a saved part of T enters exactly once.
- `income_employee` is net of income tax (tax is levied at government level, `CentralGovernment.compute_taxes`). `income_rental` is after tax (`compute_rental_income`).
- Monthly model: replace each flow with its mean over t−4..t−2. Stocks: closing t−2 value. (Monthly is not the production frequency. Implement it, and test it.)
- First period (t = 1): t−1 values are the opening HFCS values already stored as `current`.

### 3.3 Housing shortfall (D8)

- New mode, support off: the allocator pays rent before the goods floor and before debt service. Any rent it cannot pay is `unpaid_rent`.
- Tenant: pays `rent − unpaid_rent`.
- Landlord: rental income and the rental-income tax base fall by `unpaid_rent` for that dwelling.
- Next period's housing step: a private renter (tenure 3, house ID ≠ −1) with `unpaid_rent > 0` vacates. Property inhabitant → −1; remove the household from the landlord's `corr_renters`; household `Corresponding Inhabited House ID = -1` and social tenure status. This reuses the `sellers_vacate` mechanics in `HousingMarket`.
- Social tenants who cannot pay the social rent: the government writes off the shortfall (recorded separately). No further move.
- Exit: unchanged. Social-housing households search every period (`property.py`, `ind_dec`).
- Owners in mortgage distress: no change (forced-sale route already ends in social housing through `sellers_vacate`).

### 3.4 Numerical guard (D11)

```
G_h,t = 0.01 × M_h,t
```

Replaces the subsistence series in every behavioural-income floor (§4.5). Raw income is kept. A flag `income_guard_bound = (raw < G)` is recorded. The goods-floor switch does not affect G.

## 4. Current code (verified on `38e9be92`)

### 4.1 Period sequence (`macromodel/simulation.py`)

| Step | `Country` method | Relevant content |
|---|---|---|
| 1 | `update_pre_credit_planning_metrics` → `update_pre_credit_household_finance_metrics` (1731) | `_update_benefit_planning_metrics`, `_set_household_income_expectations(replace_current=False)`, `_set_household_target_demand(False)` (which calls `_update_subsistence_consumption` and `_record_behavioural_income`) |
| 2 | `prepare_housing_market_clearing`, `clear_housing_market`, credit market, `reconcile_post_grant_feasible_plan`, `process_housing_market_clearing` (2782: rent computed, `sellers_vacate`) | housing moves happen here |
| 3 | `clear_labour_market`, `update_post_labour_planning_metrics` (1818–1826) | `_set_household_income_expectations(True)`, `refresh_known_household_receipts`, `_set_household_target_demand(True)`, `settle_authoritative_household_payments` |
| 4 | `prepare_goods_market_clearing` (≈3281–3305) | `compute_stage5_subsistence_support` from the post-relief funding gap → `set_stage5_subsistence_support_by_household` |
| 5 | `update_realised_metrics` (≈4440–4561) | final `income_rental`; `income_employee`; `settle_household_social_transfers`; dividends/financial income; `reconcile_realised_household_support`; `income`; `_record_behavioural_income(expected=False)`; `update_wealth`; `compute_taxes`; `compute_deficit` |

### 4.2 Stage 5 support (to be disabled in the new mode)

| Site | Role |
|---|---|
| `Country.settle_authoritative_household_payments` (2927) → `Households.apply_consumption_floor_to_post_grant_plan` (1532) → `HouseholdFinancialFeasibility.settle_consumption_floor` (274) | Allocation: `S = max(F + R − (A − E), 0)`, then floor and rent, debt service, goods above floor, investment. Also owns consumer/mortgage suspension and distress. Keep the owner; change only the allocation policy. The graph misses the last call (it goes through `self.financial_feasibility`); grep confirms it is live. |
| `compute_stage5_subsistence_support` (251), called in `prepare_goods_market_clearing` | Sets support from the remaining shortfall. Zero in the new mode. |
| `Country.reconcile_realised_household_support` (698) | Ex-post: support recomputed from realised goods up to the floor plus rent. No-op in the new mode. |
| `clear_stage5_subsistence_support` (1831) | Clears per period. Keep. |

### 4.3 Social income booking and fiscal cost

| Site | Content | Change |
|---|---|---|
| `Country.compute_realised_household_social_income_components` (754) | returns (pension, UB, other, Stage 5) | add T as a named component (not in the Stage 5 slot) |
| `Country.settle_household_social_transfers` (784) | appends each component; `income_social_transfers` = their sum | append `income_minimum_income`; include in the sum |
| `Country.refresh_known_household_receipts` | rebuilds known receipts and discards the Stage 5 slot | carry T explicitly |
| `Country._set_household_income_expectations` | builds expected social income, both passes | add T once per pass (replace, not append, on pass 2) |
| `CentralGovernment.compute_deficit` (522) | sums UB + pensions + other + `total_necessity_support` | add `total_minimum_income` |
| `Households.compute_non_property_income` (1903) | `income_employee + income_social_transfers` (learning input) | T observed once through `income_social_transfers`; no extra change |

### 4.4 Rent and assets

| Site | Content | Change |
|---|---|---|
| `Households.compute_rental_income` (1686) | landlord income = (1 − τ) × full rent of every occupied rented dwelling | subtract that dwelling's `unpaid_rent` |
| `Country.update_realised_metrics` → `compute_taxes(current_total_rent_paid=rent[tenure == 3].sum())` (4677) | rental-income tax on rent due | use rent actually paid |
| `Households.update_wealth` (3972) | tenant outflow `cash_rent = current_consumption_cash_rent()` (the `rent` series) | outflow = rent paid |
| `HousingMarket` clearing, `sellers_vacate` block | sets seller house ID to −1 and social tenure | add the unpaid-rent vacate path (§3.3) |
| `households_ts`: `liquid_financial_assets`, `illiquid_financial_assets` | gross stocks | read for the notional 3% |
| `DefaultSocialHousing.compute_social_housing_rent`; `Households.compute_rent` (3418) | social rent = 0.25 × scalar UB × adults; booked as `Rent paid to Government` | unchanged (D8) |

### 4.5 Behavioural-income floors (all must switch from S to G)

| Site | Current operation |
|---|---|
| `Country._record_behavioural_income` (2026, 2045) → `reconcile_behavioural_income` (`util/behavioural_income.py`) | `max(raw, S)`; initial `income_bridge_current = max(n, S)` and `income_bridge_floor_used` |
| `Country._set_household_target_demand` (2195) | passes `income_floor = S` to the monetary PI / uncertainty moments (`monetary_income_moments`) |
| `Country._set_household_target_demand` (2415) | passes `subsistence_income = S` to the consumption function (`_geometric_average_income` fallback, `consumption.py`) |
| `ConsumptionInputAdapter.collect_inputs` (497) → `IncomeHistories.subsistence_nominal_history` → `prepare_inputs` (326–329) | `log(max(B, S))` before the 4-quarter geometric mean (window fixed at 4 for quarterly, line 202) |
| `borrowing_rates.py` (25–33) | `max(B, S)` in the prior real-income estimate. **Not in earlier drafts.** |

Not affected: `reconcile_realised_household_support` (719) and `prepare_goods_market_clearing` (3281) read S as the goods floor; they follow the goods-floor switch, not G.

### 4.6 Configuration today

`CountryConfiguration.subsistence_consumption` (dict), validated by `_read_subsistence_floor_parameters` (keys `smic_share`, `cu_weights`, `housing_deduction_monthly`). Household CU weights are read separately as `Households.cu_weights`.

## 5. Design

### 5.1 Configuration

New block `CountryConfiguration.household_support`:

```yaml
household_support:
  mode: legacy_stage5          # legacy_stage5 | minimum_income
  goods_floor: true            # independent switch; legacy mode requires true
  minimum_income:
    rsa_base_monthly: 499.31   # EUR, base year 2014, ordinary-household units
    base_year: 2014            # must equal initial_year
    notional_asset_return_annual: 0.03
    guard_share_of_m: 0.01
```

Validation: unknown keys rejected; `mode == legacy_stage5` requires `goods_floor: true` (the legacy semantics); all numbers finite and positive; `base_year == initial_year`. Default: `legacy_stage5`, so existing runs are unchanged.

Supported run configurations: legacy; minimum income + floor on; minimum income + floor off.

### 5.2 New pure functions (`macromodel/agents/households/func/minimum_income.py`)

```python
def minimum_income_level(*, rsa_base_monthly_model_units, time_unit, consumption_units,
                         transfer_index, cpi_ratio) -> np.ndarray
def assessed_income(*, employee, unemployment, pension, other_transfers, rental,
                    liquid_assets, illiquid_assets, notional_rate_period) -> np.ndarray
def minimum_income_top_up(*, level, assessed) -> np.ndarray        # max(level - assessed, 0)
def income_guard(*, level, share) -> np.ndarray
```

No state, no RNG. Country assembles inputs and stores outputs.

### 5.3 Orchestration in `Country`

1. `update_pre_credit_household_finance_metrics`: call `_set_minimum_income_entitlement()` after `_update_benefit_planning_metrics` (so `I_t` is current) and **before** `_set_household_income_expectations`, which overwrites `income_rental` with period-t planned values (country.py 1941, 1963). It computes M, Y, T from t−1 series and stores `minimum_income_level`, `minimum_income_assessed`, `minimum_income_entitlement` (household ts) once per period. Idempotent: a second call in the same period raises.
2. `_set_household_income_expectations` (both passes) and `refresh_known_household_receipts`: add the stored T to expected social income and `financing_income` once. Pass 2 replaces, it does not add.
3. `settle_authoritative_household_payments`: in `minimum_income` mode, the allocator gets resources that already include T, the effective floor `F × goods_floor`, and `support_enabled=False` (§5.4).
4. `prepare_goods_market_clearing` and `reconcile_realised_household_support`: Stage 5 support is zero in `minimum_income` mode.
5. `update_realised_metrics`: `settle_household_social_transfers` books T as `income_minimum_income`; government series `total_minimum_income`; `compute_deficit` includes it; landlord income, rental tax and tenant outflow use rent paid.
6. Housing step of t+1: unpaid-rent vacate path.

### 5.4 Allocator change (`settle_consumption_floor`)

Add `support_enabled: bool = True`. When false:

```
S  = 0
Rp = min(R, A)                     unpaid_rent = R − Rp
Fp = min(F, A − Rp)                unmet_floor = F − Fp        (F = 0 when the floor is off)
then debt service, goods above floor, investment, from A − Rp − Fp, exactly as now
```

Return `unpaid_rent` and `unmet_floor` as new plan fields. No `maximum(remaining, 0)` that hides a missed payment. Consumer-before-mortgage suspension order unchanged. Order rent → floor → debt service is a design choice (rent first protects tenancy); legacy order is unchanged when `support_enabled=True`.

### 5.5 Guard

Introduce `income_guard` (economy ts, per household, same units as income). Replace the S argument with G at the five sites in §4.5. `IncomeHistories.subsistence_nominal_history` → `guard_nominal_history`, with the same shape/unit/label checks. Rows before activation keep their S values (history is not rewritten). Legacy mode keeps S everywhere.

## 6. Accounting invariants (tests)

1. Σ_h T_h = government `total_minimum_income`; it enters `compute_deficit` once.
2. `income_social_transfers` = pension + UB + other + T (+ Stage 5 in legacy mode only). In `minimum_income` mode, Stage 5 = 0 every period.
3. T is independent of desired goods, actual spending, credit, liquidation and realised income of period t.
4. T booked once; a second settlement in the same period raises.
5. Rent: Σ tenant rent paid = Σ landlord gross rent received; rental tax base = rent paid; unpaid rent recorded per landlord and in aggregate.
6. Social rent shortfall written off by government, recorded separately.
7. Floor off: paid goods ≤ desired goods for every household. Floor on: paid goods above desired only through the funded floor; `unmet_floor` visible.
8. Household deposits credited once (T flows through income → `update_wealth`).
9. MPC receipt: absent from Y, present once in planning, financing and final income.
10. Reset/clone: new per-period carriers cleared on `reset`; no extra RNG draw.

## 7. Increments and acceptance gates

Run GitNexus `impact` on each symbol before changing it. Each increment is one PR stacked on #213. Gate for every increment: in the default mode `legacy_stage5`, every pre-existing output series is numerically identical to `main` on a seed-1 smoke run (new series may be added).

| # | Deliverable | Acceptance gate |
|---|---|---|
| 1 | Config block + validation; pure functions (§5.2) with unit tests | Worked examples: zero income, income above M, assets at €0 / €10k / €200k, CU 1 / 1.8 / 2.1, monthly vs quarterly, I_t ≠ 1, base-year mismatch raises. Legacy byte-identical. |
| 2 | Entitlement orchestration (§5.3 steps 1–2) and booking (step 5, social income and deficit only) | Invariants 1, 2, 4, 8, 9. T identical across the two planning passes. |
| 3 | Allocator `support_enabled` + Stage 5 off in new mode (steps 3–4) + goods-floor switch | Invariants 3, 7. All three run configurations complete a seed-1 run. |
| 4 | Guard (§5.5) | All five sites switched in new mode; legacy unchanged; flag recorded; zero / negative / no-positive-window cases tested. |
| 5 | Housing: unpaid rent, landlord write-off, vacate path, social-rent write-off | Invariants 5, 6. A forced-shortfall test household moves to social housing in t+1 and can leave later. |
| 6 | Outputs and diagnostics (§8) | Extractors and notebook read new series. |

## 8. Validation (after increment 6; multi-seed, ≤ 4 parallel runs)

Registered as a vault experiment spec before running. Arms: legacy; minimum income + floor on; minimum income + floor off. Shared checkpoint (D10). The shared checkpoint must be built in legacy_stage5 mode; policy arms switch mode by reset. A minimum_income-built country cannot be reset to another guard basis. Compare from activation + 4 quarters (guard window, §10 R5).

Report, by income quintile: share receiving T; mean T / M; fiscal cost (% GDP) vs the data on RSA spending; MPC profile; share of household-periods with guard binding (target < 0.5%) together with wealth-ratio clipping incidence; unpaid-rent events; social-housing entries; re-entries within 4 quarters of leaving; unmet-floor incidence (floor-on arm). No parameter is tuned to these outputs.

## 9. Open items

1. VERIFY the R132-1 wording on Légifrance (currently quoted from AN written answers and the CE decision). Only affects how the departure is described.
2. Rent-before-floor order in §5.4: confirm.
3. Monthly frequency: the monthly history window length is not fixed by the quarterly check; confirm when monthly runs are needed. Monthly early periods: window truncated at the opening row (t=2 uses row 0, t=3 rows 0–1).

## 10. Review of this plan (2026-11-21)

Method: GitNexus `context`/`impact` on every symbol in §4, cross-checked with grep, and source reading at each call site.

| # | Finding | Severity | Resolution in this draft |
|---|---|---|---|
| R1 | `borrowing_rates.py` floors income at S. Draft 7 missed it. | important | Added to §4.5 and increment 4. |
| R2 | The graph shows no production caller of `settle_consumption_floor`; the call goes through the `self.financial_feasibility` attribute. Live via `Households.apply_consumption_floor_to_post_grant_plan`. | note | Recorded in §4.2. Do not trust graph impact alone for attribute calls; grep each site. |
| R3 | Landlord income (`compute_rental_income`), rental tax (`compute_taxes`) and tenant outflow (`update_wealth`) all use rent due. Unpaid rent needs three changes, not a diagnostic. | important | §4.4, §5.3 step 5, invariant 5. |
| R4 | `I_t` must use the setter's real level ratio; `_firm_contract_rates` is real and CPI is applied at payment, so `M = base × I × CPI ratio` does not count inflation twice. | checked | §3.1. |
| R5 | The behavioural-income window is 4 quarters. One guarded quarter at 1% M lowers the geometric mean to about 0.32× for a household normally near M. Activation mixes S- and G-floored rows for up to 3 quarters. | important | Compare from activation + 4 quarters (§8); report guard binding with ratio clipping. |
| R6 | t−1 assessment removes the two-step estimate/replace contract of draft 7: T is known at period start and the same in both passes. | simplification | §3.2, §5.3. |
| R7 | Assessing from component series automatically excludes last period's T, Stage 5 and the MPC shock (D6, D7). Using `income` or `income_social_transfers` would break both. | important | §3.2; invariant 9. |
| R8 | `net_smic_base` is population-expanded once. RSA base must use the same `scale`; applying it twice or not at all is a factor-of-`scale` error. | important | §3.1; increment 1 test. |
| R9 | Social tenants can also fail to pay social rent; draft 7 was silent. | gap | §3.3: government write-off, recorded. |
| R11 | `income_rental` is overwritten with period-t planned values inside `_set_household_income_expectations` (1941, 1963). Reading it after that call would assess period-t rent, not t−1. | important | §5.3 step 1 fixes the call position. |
| R12 | Liquid and illiquid stocks are appended only in `Households.update_wealth` (closing stocks). Housing purchases in step 2 change cash within the period but after the assessment. | checked | §3.2. |
| R13 | The allocator also covers social tenants (their social rent is in `rent`). The vacate path must apply only to private tenants, or social tenants would be "evicted" into the state they are already in. | important | §3.3 restricts it to tenure 3, house ID ≠ −1. |
| R14 | "Byte-identical" is too strong when new series are added to outputs. | nit | §7 gate reworded. |
| R10 | Draft 7 named `feature/minimum-income-design` and a stale base; mixed superseded decisions (SMIC level, asset cut-off, current-period window) with current ones. | hygiene | Rewritten; superseded items removed from the body. |

## Sources

- DREES, *Minima sociaux et prestations sociales* 2025, fiche 09 (resource base, reference period) and fiche 23 (RSA).
- Décret 2013-1263 (RSA €499.31 from 2014-01-01); décret 2014-1127 (€509.30 from September 2014).
- CASF R132-1 (notional 3% on capital), as quoted in AN written questions 25719 and 26952; Conseil d'État, 14 June 2017, n°401637.
- Vault: `wiki/experiments/2026-11-17-hfcs-legacy-mpc.md`, `wiki/debugging/2026-11-11-floor-bound-self-funded-households.md`, `wiki/debugging/2026-06-10-household-disposable-income-tax-treatment.md`.
