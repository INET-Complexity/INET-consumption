# Fiscal, firm-finance, credit and inflation audit — 12 November 2026

**Reviewed 12 November 2026:** findings **8 and 9 fixed in `dfde7397`**;
the 144 focused tests passed again in an isolated copy of that revision.
Finding **7 resolved for the frozen `dfde7397` baseline at φ=0.05**:
four seeds pass provenance and stability checks; see
[validation report](phi005-validation/report.md). The FRA default and
`investment_adjustment` preset are both selected at 0.05 in the publication
branch. The separate capital-accounting changes on local `main` are outside
this validation and push. Findings 1–6 remain open. Economic numbers in the
original audit below retain the φ=0.25 basis.

The largest confirmed accounting defect is a firm capital-purchase tax recorded
as government revenue without a matching firm cash debit. The lending collapse
is real and is reproduced by borrower DSCR capacity, despite abundant reported
bank capital capacity. Low inflation and persistent fiscal deficits are also
present, but their economic remedies require specification choices rather than
an arbitrary parameter adjustment. The government-spending shock semantics
still need correction or explicit disclosure before paper IRFs (finding 3).

## Original audit evidence and limits

- Inspected the current working source on `main`, HEAD `dd034a54`, preserving
  existing edits. No model source/config/notebook was changed by this audit.
- Replayed saved `run_model/data/output_data/FRA_benchmark.h5`: seed 40,
  50 quarters, 825 firms. **The embedded configuration uses φ=0.25.** The
  accompanying manifest describes a separate `investment_adjustment` scenario
  with φ=0.05. Current YAML and preset also use 0.05. These results must not be
  presented as validation of the current 0.05 baseline.
- Checked all 50 quarters, not just the terminal observation. Exported only
  aggregate evidence to `quarterly_evidence.csv` and `summary.json`.
- Reconstructed saved CFADS to within €0.000005 and firm closing deposits to
  within €0.000031. Government debt increments equal reported deficits.
- The original deterministic probes reproduced government-shock persistence
  and the pre-fix monetary-floor bypass. The live replay now verifies the
  repaired floor. No fresh stochastic simulations were run during the original
  audit; the later finding-7 validation is recorded separately. One original
  seed establishes accounting/rule behaviour, not robust macro magnitudes.
- The chain “government deficit → firm cash accumulation → lending collapse”
  is **not established**. The magnitudes coexist; a complete sector transaction
  matrix and paired interventions are still needed to attribute the flows.

## Ranked findings

Criticality means threat to accounting validity or paper interpretation.
Difficulty includes integration and validation, not just lines of code.

| Rank | Criticality | Finding | Status | Fix difficulty |
|---|---|---|---|---|
| 1 | Critical | Capital-purchase tax has government revenue but no firm payment | Confirmed accounting defect | Medium |
| 2 | High | DSCR cash flow omits direct-TFP cash expense and embeds investment spending | Omission confirmed; capex treatment is a design issue | Easy omission fix; medium–hard redesign |
| 3 | High | One-quarter government shock permanently raises the recursive spending target | Confirmed semantics; inconsistent with a temporary-level interpretation | Medium |
| 4 | High | Persistent primary deficit lacks a debt-stabilising fiscal response | Confirmed configured mechanism; economic closure choice | Medium–hard |
| 5 | High | Cash-rich nonborrowers coexist with DSCR-limited borrowers; zero initial firm debt and low payout shape the transition | Confirmed mechanism/configuration; realism unvalidated | Hard |
| 6 | High | Low nominal cost growth and zero incumbent productivity indexation accompany low inflation | Observed mechanism; causal shares not identified | Medium–hard |
| 7 | Medium | Audited benchmark still contains the high-φ investment cycle | Resolved for frozen φ=0.05 baseline; see validation report | Easy to select/rerun; medium to validate |
| 8 | Medium | Monetary shock bypasses the configured policy-rate floor | Fixed in `dfde7397` | Easy |
| 9 | Medium | Firm total-credit-exposure history has two entries per quarter | Fixed in `dfde7397` | Easy |

## 1. Government collects an unpaid firm capital tax

**Source:** `macromodel/agents/central_government/central_government.py:363`
and `:479`; `macromodel/agents/firms/firms.py:3711`;
`macromodel/country/country.py:4334`.

Government records `0.2485669837 × realised firm capital purchases` as capital
formation tax and includes it in revenue. The same tax is added to GFCF for GDP
accounting. Neither the firm cash settlement nor its credit-budget commitments
include the corresponding capital tax. Searching the tax state's consumers
finds government revenue, initial reconciliation and GDP accounting, not a
firm payment. The saved tax equals this base exactly, and closing firm cash
reconciles without paying it.

Magnitude: **€974.53bn over 50 quarters**, averaging **3.29% of GDP**;
€17.50bn in quarter 50. Government deficit is understated relative to a
collection-only-on-payment ledger, while firms retain cash that a genuine tax
would remove. Booking the same tax on both GDP sides can preserve a GDP
identity without making the transaction funded.

**Remedy:** decide whether this is an actual purchase tax or an accounting
valuation adjustment. For an actual tax, include it consistently in ex-ante
purchase affordability, credit demand, post-credit expenditure allocation,
realised settlement and the acquisition-cost accounting convention. Government
revenue must equal the settled payer amount. Removing it from revenue instead
would increase the reported deficit; it is not a fiscal improvement.

Illustrative ledger contract, after introducing explicit per-firm tax carriers:

```python
capital_tax_due = capital_tax_rate * realised_capital_purchases
capital_purchase_cash_cost = realised_capital_purchases + capital_tax_due
assert np.isclose(capital_tax_paid.sum(), government_capital_tax_received)
```

Do not apply a closing-deposit subtraction alone: that would create unplanned
cash shortfalls. Validate a closed payer/payee transaction and both GDP sides,
then rerun the baseline. This is the first repair to make.

## 2. DSCR binds, but its cash measure is inconsistent

**Source:** `macromodel/markets/credit_market/func/clearing.py:71`, `:271`,
`:1900`; `macromodel/agents/firms/firms.py:3711` and `:3750`.

All material unmet LT credit across all 50 quarters carries DSCR binding code
3. At quarter 50, firms request **€76.22bn** and receive **€1.63bn**. Reported
bank firm capacity is **€4,166.72bn**. That capacity is CAR-based potential
capacity, not an offer to every borrower. There is no evidence here that
aggregate bank capital scarcity is causing the collapse.

CFADS is a four-quarter mean of positive sales minus wages, **all goods
purchases**, production tax and corporate tax. It omits the separate direct-TFP
cash expense, although deposits and distributable cash deduct it. At quarter
50, recognizing that expense lowers aggregate lagged CFADS from **€65.00bn to
€41.16bn**. A frozen-history DSCR replay reduces demand-clipped LT capacity
from **€1.63bn to €0.56bn**. At quarter 10 the equivalent figures are €22.77bn
and €6.94bn. These are mechanical capacities, not simulated new equilibria.

The omission therefore makes underwriting too generous. Fixing it alone
**worsens** the observed rationing. Separately, subtracting the entire previous
capital-purchase bill makes investment reduce future capacity to finance
investment. Existing debt service then consumes the residual room. Whether
that is intended conservative free-cash-flow underwriting or inappropriate
double discipline of expansion is an economic design question.

**Remedy:** register the intended cash concept first. If direct TFP remains a
cash expense, an expenditure-consistent period measure includes:

```python
period_cfads = (
    cash_sales
    - wages
    - goods_purchases
    - production_taxes
    - corporate_taxes
    - direct_tfp_cash_expense
)
```

Separate maintenance and expansion capital spending if the intended DSCR
concept requires it; do not silently remove all capex or simply lower minimum
DSCR to obtain more loans. The missing capital tax in finding 1 must also be
handled consistently with the chosen concept.

GitNexus impact: **HIGH**, three direct callers and six upstream symbols
within three hops, including credit clearing and excess-demand finance-capacity
diagnostics. A short code change has broad economic effects. Validate those
paths together and rebaseline all paper results afterward.

## 3. The fiscal IRF shock is an innovation, not a temporary level change

**Source:** `macromodel/utils/prehooks/irf_shocks.py:58`;
`macromodel/agents/government_entities/func/consumption.py:378`;
`macromodel/agents/government_entities/government_entities.py:224`.

The wrapper multiplies the desired target while active. The next target uses
the previous **shocked** target. With zero growth/inflation, a one-quarter 10%
shock produces **100 → 110 → 110 → 110**, not 100 → 110 → 100. Applying it for
four quarters compounds the direct factor to 1.1⁴ before the hook turns off.
The active notebook specifies `gov_cons_10pct`, duration 1.

**Remedy:** specify whether the paper experiment is a spending-growth
innovation or a temporary spending-level deviation. For the former, retain
the rule but label and normalise the IRF accordingly. For the latter, maintain
a separate unshocked recursive target and apply the intervention to the
period's spending order, with a multi-period sequential regression. Do not
reset the whole shocked economy to the baseline path: endogenous feedbacks
should remain. Existing tests with a fixed previous target miss this feedback.

## 4. The deficit is primarily a primary-balance problem

**Source:** `macromodel/agents/central_government/central_government.py:510`
and `:556`; `macromodel/agents/government_entities/func/consumption.py:378`;
`run_model/config/country_config_FRA.yaml:476`.

Average reported deficit is **5.21% of GDP**, of which **5.10pp is primary**;
interest is only **0.108pp**. Quarter 30 reaches an 11.34% deficit. Debt rises
from **95.72% to 146.50% of annual GDP**. Government debt increments reconcile
with deficits; the debt accumulator itself is not the defect.

Mean revenue is 43.11% of GDP. Government purchases account for 28.32%, public
pensions 15.47%, other transfers 3.09%, unemployment benefits 1.09%, necessity
support 0.245%, and interest 0.108%. These are model ratios, not validated
empirical targets. Spending follows forecast growth/inflation, benefits follow
their configured rule, and the inspected fiscal rules contain no debt/primary
balance feedback. Low sovereign interest is suppressing, not driving, the
reported deficit. Necessity support is not its dominant component.

**Remedy:** reconcile initial spending/revenue concepts and funded tax flows,
then choose an explicit fiscal closure appropriate to the paper experiment.
First establish whether the baseline needs a correction in fiscal levels or
a stabilising rule; do not use a new tax parameter to conceal accounting gaps.
This needs research judgment and multi-seed validation, not a one-line fix.

## 5. Aggregate wealth growth masks the distribution of finance

**Source:** `run_model/config/data_config_FRA.yaml:20`;
`run_model/config/consumption_paper_parameters.yaml:84`;
`macromodel/country/country.py:3511`;
`macromodel/agents/firms/firms.py:1740`.

Firms start with **zero debt** by explicit data configuration. Deposits rise
from **€474.69bn to €2,173.54bn**; assets from €4,078.34bn to €5,800.86bn.
Capital stock value changes only from €3,074.58bn to €3,034.50bn. This is mainly
cash accumulation, not an aggregate physical-capital boom.

In quarter 50, nonrequesters hold **€1,819.42bn**, about **83.7%** of closing
firm deposits. Requesters still hold €354.11bn, but their opening cash is
committed to wages, taxes, debt service and working capital before investment.
Raw deposits do not equal free investment cash. Ordinary ST demand is near
zero partly because operating finance uses a separate revolving facility;
ST requests are largely repair/refinance buckets, not all operating needs.

The configured firm payout is 10% of eligible cash surplus. Cumulative payouts
are €151.43bn. The cash bridge is:

`€2,879.38bn net operating/purchase cash + €715.24bn credit − €452.37bn principal
− €37.42bn net interest − €1,254.56bn direct TFP − €151.43bn distributions
= €1,698.85bn deposit growth`.

There is no unexplained residual in that bridge. It does not include the
unpaid capital tax in finding 1. Cumulative public deficits are €1,504.97bn;
bank reserves grow €1,820.21bn, but these similarities are not a causal or
complete sector-balance reconciliation.

**Remedy:** after the accounting repairs, discipline initial firm debt and its
bank counterpart, cash/investment/payout behaviour, and borrower-level finance
against data. Do not interpret turning off zero-initial-debt as a complete
calibration. Validate a transition and borrower distributions, not only sector
totals. This is the hardest economic task in the audit.

## 6. Low inflation reflects weak nominal cost growth in this baseline

**Source:** `run_model/config/country_config_FRA.yaml:257` and `:360`;
`macromodel/agents/firms/func/prices.py:920`;
`macromodel/agents/firms/func/wage_setter.py:626`.

Mean YoY fixed-basket CPI inflation is **0.288%** in quarters 5–50, versus the
configured 2% target. The policy rate is zero in **37/50 quarters**.
Using fixed initial firm-output weights, from quarter 1 to 50:

- Smoothed technical MC rises **3.72% in total**, labour MC 4.37%, material MC
  3.51%, and prices 3.60%.
- Effective markup changes **−0.12%**; it is not a large sustained markup
  collapse in this diagnostic.
- TFP rises **6.87%**, while the stored real wage-offer level rises only
  **0.54%**. These measures are diagnostics, not a complete labour-share
  decomposition.

The active rule sets a price level from technical costs, bounded markups and
an AC floor. It explicitly ignores general expected PPI inflation as a direct
price multiplier. Incumbent productivity indexation pass-through is 0.0;
**CPI wage indexation remains active at 1.0**. Thus it would be wrong to say
all wages or all indexation are frozen.

**Remedy:** after accounting and credit fixes, test a small, registered wage/
productivity pass-through comparison with real wages, labour share,
unemployment and prices together. Do not add automatic price drift merely to
hit 2%. This audit identifies weak nominal cost growth, not its causal shares
or a validated coefficient that solves it. Earlier inflation experiments are
on different configurations and cannot select today's baseline automatically.

## 7. The inspected benchmark is still the high-φ arm

**Resolved for the frozen baseline:** [four-seed validation](phi005-validation/report.md).
The following describes the original audited artifact, which remains preserved.

**Source:** HDF5 root `configuration` attribute;
`run_model/src/notebook_workflow.py:738`;
`macromodel/agents/firms/func/target_capital_inputs.py:175`.

The manifest's scenario override does not describe the separate benchmark
object. The benchmark HDF5 uses 0.25; current source uses 0.05. This is a
provenance gap, not evidence that override application is broken. The current
workflow supports separate benchmark overrides and checks cache fingerprints.

The capital target responds to current target production with no explicit
memory of credit refusal. Existing four-seed grid evidence in the vault finds
detrended GDP SD 3.9% at φ=0.25 versus 1.5% at 0.05. This audit did not rerun
that grid. Baseline cycles can alter IRF dispersion and shock-phase dependence.

**Implemented and reviewed:** explicit φ=0.05 selection, uniquely named output
paths, copied population/YAML inputs, effective configurations and hashes,
and independent seed-40 HDF5 verification. All four effective configurations
use 0.05; mean detrended GDP SD is 1.43% (range 1.11–1.68%). The original
φ=0.25 HDF5 was preserved. Input/configuration hashes and the frozen source
were checked again during review. This validates provenance and the registered
stability gate, not fiscal/credit/inflation acceptance or the separate newer
capital-accounting specification.

## 8. The easing-shock wrapper bypasses the zero lower bound

**Fixed in `dfde7397`:** additive rate-path shocks now respect the configured
annual floor in model-period units. Positive-shock and unbounded behaviour
are preserved. The following is the original pre-fix finding.

**Source:** `macromodel/utils/prehooks/irf_shocks.py:102`;
`macromodel/agents/central_bank/func/policy_rate.py:269`.

`_PolicyRateShockProxy` computes the bounded baseline rate, then adds its
period shock. In a deterministic quarterly example, baseline is zero and a
−100bp annual shock produces **−0.0025 per quarter**, despite `lower_bound=0`.
Passing the same annual innovation through the native Taylor rule gives zero.
The current notebook's +100bp shock does not trigger this negative-rate case.

**Implemented and reviewed:** retain an additive rate-path intervention on the
bounded baseline rate, then enforce the configured annual floor converted to
model-period units. This preserves existing positive-shock behaviour rather
than introducing a shadow Taylor innovation. Tests cover both shock signs,
monthly/quarterly frequency, zero/positive/negative floors, no floor, positional
and keyword calls, and expiry. Easing at a zero floor now yields zero. Existing
saved IRFs are not rewritten by this code fix.

## 9. Firm exposure history is appended twice

**Fixed in `dfde7397`:** removed the redundant realised-metrics append while
retaining the staged settlement observation and its final override. A
three-period regression preserves the initial observation and checks every
subsequent value; the fresh FRA HDF5 also has 51 rows for 51 dates.

**Source:** `macromodel/agents/firms/firms.py:1998`;
`macromodel/country/country.py:3982`;
`run_model/src/visual_helpers.py:1619`.

Settlement staging appends a zero, then realised metrics append the actual
exposure. The saved history has **101 rows for 51 model dates**. Current-value
access at the end of the step is correct; time-indexed histories can be shifted,
truncated or interleaved with zeros. This does not explain the actual lending
collapse, which is independently verified from borrower arrays.

**Implemented and reviewed:** keep the staged settlement observation and final
override; remove only the extra append in realised metrics. The three-period
regression checks every value, history length, nonzero exposure and preservation
of the initial observation. The new FRA HDF5 independently has 51 rows for 51
dates. Historical malformed files remain unchanged; affected old charts still
need regeneration or reconstruction from borrower-level fields.

## Review and validation of findings 7–9

- **Review verdict: no blocking defects found in the fixes for 8/9 or the
  recorded finding-7 evidence.** GitNexus impact is LOW for the shock proxy,
  realised-metrics change and preset selection. Dynamic dispatch limits the
  graph; review also checked the actual staging/override path and raw diffs.
- **144 tests passed again** against isolated `dfde7397`: IRF shocks, policy
  rules, firm exposure history and firm accounting. Three existing pandas
  deprecation warnings; no failures. The full repository suite was not rerun.
- Finding 7: smoke plus four seeds × 50 quarters; all provenance/stability
  gates pass. The review independently checked copied population/YAML hashes,
  all effective-configuration hashes, embedded HDF5 settings and exposure
  shape. No duplicate full simulation campaign was needed for this review.
- Original fiscal/DSCR audit checks: **33 passed**. Aggregate replay and live
  deterministic probes pass; Ruff and diff checks pass.
- Implementation and evidence were already committed in `dfde7397` and
  `037ef111`. The review update and φ default selection accompany them on
  `fix/macro-audit-7-9`, excluding the separate capital-wear commit `c52e597f`.
- Remaining limits: the government shock persistence in finding 3 is not
  repaired; old IRFs/history files are not migrated; newer capital accounting
  requires a compatible rebuilt population and separate validation. Other
  working-copy model/notebook edits are preserved outside this commit scope.

Recommended repair order: funded capital-tax accounting → expenditure-consistent
CFADS and chosen investment underwriting concept → explicit fiscal shock
semantics → baseline fiscal/firm-stock reconciliation → multi-seed inflation
and credit validation → frozen paper IRFs. Findings 7–9 do not close the
economic issues in findings 1–6. A passing unit suite alone is not economic
acceptance.
