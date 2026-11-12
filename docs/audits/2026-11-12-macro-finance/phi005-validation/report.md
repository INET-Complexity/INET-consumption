# Finding 7 — resolved for the frozen φ = 0.05 baseline

12 November 2026. Four fresh FRA runs, seeds 40–43, 50 quarters each,
on frozen source `dfde7397` with an explicit φ=0.05 override. This is an
exploratory baseline validation, not paper acceptance of the economic model.

| Seed | Detrended GDP SD | Peak unemployment | Mean YoY CPI inflation | Mean deficit/GDP | Terminal debt/annual GDP | LT credit fulfillment, whole run |
|---|---:|---:|---:|---:|---:|---:|
| 40 | 1.50% | 10.70% | 0.208% | 5.55% | 154.77% | 50.00% |
| 41 | 1.68% | 12.28% | 0.150% | 5.69% | 162.06% | 53.29% |
| 42 | 1.11% | 10.23% | 0.168% | 6.12% | 159.60% | 49.51% |
| 43 | 1.42% | 11.53% | 0.169% | 5.42% | 153.38% | 51.84% |

GDP volatility uses linear-detrended log real GDP over quarters 1–50 with
population SD. CPI averages cover quarters 5–50. Deficits cover quarters 1–50;
credit fulfillment is total granted divided by total requested over the run.
No confidence intervals are claimed from four exploratory seeds.

## Registered gates

- **Provenance PASS:** effective configurations explicitly use φ=0.05 for all
  four seeds. Seed-40 HDF5 independently confirms the same configuration.
  Old `FRA_benchmark.h5` remains untouched at φ=0.25. Each artifact has a unique
  path and a saved effective configuration, rather than inheriting the label
  of a separate scenario in a shared manifest.
- **Finite-output/history PASS:** positive finite GDP and CPI; seed-40 exposure
  history has exactly 51 observations, confirming fix 9 in this FRA run.
- **Stability PASS:** GDP SD below 3% in every seed; mean **1.43%**, below the
  registered 2% threshold; peak unemployment below 20%; zero bank insolvency.
- **Smoke PASS:** two-quarter seed-40 run completed successfully; no missing-data,
  API, NaN or exception signatures in the passing smoke log. Its first attempt
  completed the simulation but failed JSON export of a NumPy boolean; the
  driver was corrected and the smoke rerun before full execution.

The earlier grid's rounded mean GDP SD of 1.5% at φ=0.05 and 3.9% at φ=0.25 is
historical context, not a new paired estimate from this validation. Do not
attribute every macro difference from the original audited seed to φ.

## Important boundary: concurrent capital-accounting changes

During setup, the working copy changed from `surplus_pool/eurostat_cfc` to
`production_cost/none`, including edits to firm code and the data configuration.
The first unisolated launch failed the population/configuration compatibility
check before simulation. It was not used as evidence.

This validation therefore uses `git archive dfde7397`, its YAMLs, and a copied,
hashed version of the original cached population. Only φ changes to 0.05.
It does **not** validate the newer capital-accounting specification. That needs
a compatible rebuilt population and its own baseline validation. All concurrent
working-copy edits were preserved.

## Economic verdict

The provenance problem and the registered stability check are resolved for the
frozen baseline. Fiscal/credit/inflation acceptance remains **INCONCLUSIVE**.
Deficits remain 5.4–6.1% of GDP, debt reaches 153–162% of annual GDP, and annual
inflation averages only 0.15–0.21%. Late-run LT fulfillment is just **13.7–16.3%**
over quarters 41–50, versus roughly 50–53% over the whole run. Findings 1–6 remain open. No accounting fix was
smuggled into this φ comparison.

## Artifacts and reproduction

- `metrics.csv`: per-seed values; `summary.json`: gate verdicts.
- `provenance.json`: revision, population/configuration hashes and environment path.
- Full outputs: `run_model/data/output_data/phi005-validation-20261112-dfde7397-isolated/`.
  This includes frozen YAMLs/population, environment snapshot, all macro panels,
  effective configurations and the seed-40 HDF5.
- Smoke log: `/tmp/inet-phi005-isolated-smoke.log`; full log: `/tmp/inet-phi005-full.log`.
- Sequential full simulation time: about **54 seconds**, after import/setup.

From the repository root, create an isolated archive and use a new output path:

```sh
mkdir -p /private/tmp/inet-phi005-dfde7397
git archive dfde7397 | tar -x -C /private/tmp/inet-phi005-dfde7397
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python docs/audits/2026-11-12-macro-finance/phi005-validation/run.py \
  --source-root /private/tmp/inet-phi005-dfde7397 --output /private/tmp/new-phi005-run --phase smoke
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python docs/audits/2026-11-12-macro-finance/phi005-validation/run.py \
  --source-root /private/tmp/inet-phi005-dfde7397 --output /private/tmp/new-phi005-run --phase full
```

The runner refuses to overwrite an existing phase directory. It reads the
original cache only to make the frozen copy; subsequent seeds reuse that copy.
