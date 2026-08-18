# Run-model notebooks

The notebook workflow is split by responsibility:

- `run_model.ipynb` — thin single-simulation runner and optional Monte Carlo call.
- `run_model_diagnostics.ipynb` — standard macro, benchmark, sector, and firm-credit diagnostics.
- `run_model_exploration.ipynb` — exploratory firm, household, reader, income, and balance-sheet analysis.
- `run_sensitivity.ipynb` — parameter sensitivity experiments.
- `run_mpc.ipynb` — household MPC experiments.
- `run_irf.ipynb` — macro impulse-response experiments.

`run_model_legacy_2026-08-18.ipynb` is a preserved historical snapshot. It is
clearly marked inside the notebook and must not be used for new work.

Reusable calculations and experiment implementations belong under `src/`.
Top-level Python files in this directory are compatibility commands only.
