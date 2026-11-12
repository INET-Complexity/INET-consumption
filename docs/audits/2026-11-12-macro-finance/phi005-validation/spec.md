# Finding 7: fresh φ = 0.05 benchmark validation

Registered 12 November 2026, before execution. This is baseline validation,
not a parameter search or acceptance of the unresolved fiscal/credit mechanisms.

Pre-smoke amendment: concurrent edits switched capital accounting to
`production_cost/none` while the cached population remains
`surplus_pool/eurostat_cfc`. The first launch stopped at the configuration
compatibility check before simulation. Isolate source/configuration from
`git archive dfde7397` and apply only φ=0.05; retain the cached population.
This isolates finding 7 from the separate capital-accounting work. It does
not validate the concurrently revised working configuration. Use a new
`phi005-validation-20261112-dfde7397-isolated` output directory.

- Use current source after fixes 8/9, the existing `data_benchmark.pkl` population
  (build seed 38), FRA quarterly frequency, and the frozen country YAML with
  explicit φ=0.05. No consumption-intercept preset or other scenario override.
- Freeze the input YAMLs and their hashes, input-cache hash, git revision/diff,
  interpreter/dependency versions and effective simulation configuration.
- Smoke: seed 40, 2 quarters; require finite positive GDP/CPI, successful exit,
  φ=0.05 in saved HDF5 configuration, one exposure observation per model date,
  and no missing-data/API/NaN error signatures. Expected under two minutes.
- Full: seeds 40–43, 50 quarters each, sequential, one BLAS thread. Save aggregate
  data per seed and seed-40 HDF5 under a unique directory. Expected 5–20 minutes;
  stop on exceptions, nonfinite/nonpositive GDP/CPI, missing data, or >30 minutes.
- Provenance gate: effective φ exactly 0.05 for every seed; the seed-40 embedded
  HDF5 configuration agrees; previous benchmark paths remain unchanged.
- Stability gate: every seed's linear-detrended log-real-GDP SD over quarters
  1–50 below 3%; mean below 2%; peak unemployment below 20%; no bank insolvency.
  These are diagnostic thresholds, not an empirical calibration target.
- Report each seed's GDP volatility, CPI inflation (quarters 5–50), deficit/GDP,
  terminal debt/annual GDP, LT credit fulfillment and unemployment. Compare the
  saved earlier φ-grid only as historical context, disclosing revisions.
- Verdict: finding 7 is resolved if provenance passes. Economic stability is a
  separate PASS/FAIL and does not close findings 1–6 or establish paper readiness.
