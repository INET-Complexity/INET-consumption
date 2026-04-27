---
name: macromodel
description: "Skill for the Macromodel area of macro-main. 27 symbols across 7 files."
---

# Macromodel

27 symbols | 7 files | Cohesion: 91%

## When to Use

- Working with code in `macromodel/`
- Understanding how main, from_datawrapper, check_compatibility work
- Modifying macromodel-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `macromodel/timeseries.py` | historic, get_keys, write_to_h5, reset, __eq__ (+6) |
| `macromodel/simulation.py` | from_datawrapper, check_compatibility, get_compatibility_mismatches, run_prehooks, run_posthooks (+4) |
| `run_model/run_model.py` | _resolve_run_model_path, main |
| `run_model/src/monte_carlo.py` | _run_single_seed, _run_seed_batch |
| `macromodel/timestep.py` | step |
| `macromodel/country/regional_aggregator.py` | sync_central_banks |
| `macromodel/agents/firms/firm_ts.py` | FirmTimeSeries |

## Entry Points

Start here when exploring this area:

- **`main`** (Function) — `run_model/run_model.py:39`
- **`from_datawrapper`** (Function) — `macromodel/simulation.py:70`
- **`check_compatibility`** (Function) — `macromodel/simulation.py:512`
- **`get_compatibility_mismatches`** (Function) — `macromodel/simulation.py:530`
- **`historic`** (Function) — `macromodel/timeseries.py:115`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `TimeSeries` | Class | `macromodel/timeseries.py` | 17 |
| `FirmTimeSeries` | Class | `macromodel/agents/firms/firm_ts.py` | 11 |
| `main` | Function | `run_model/run_model.py` | 39 |
| `from_datawrapper` | Function | `macromodel/simulation.py` | 70 |
| `check_compatibility` | Function | `macromodel/simulation.py` | 512 |
| `get_compatibility_mismatches` | Function | `macromodel/simulation.py` | 530 |
| `historic` | Function | `macromodel/timeseries.py` | 115 |
| `get_keys` | Function | `macromodel/timeseries.py` | 126 |
| `write_to_h5` | Function | `macromodel/timeseries.py` | 134 |
| `reset` | Function | `macromodel/timeseries.py` | 235 |
| `get_aggregate` | Function | `macromodel/timeseries.py` | 256 |
| `step` | Function | `macromodel/timestep.py` | 40 |
| `run_prehooks` | Function | `macromodel/simulation.py` | 253 |
| `run_posthooks` | Function | `macromodel/simulation.py` | 270 |
| `iterate` | Function | `macromodel/simulation.py` | 284 |
| `sync_central_banks` | Function | `macromodel/country/regional_aggregator.py` | 13 |
| `write_field_to_h5` | Function | `macromodel/timeseries.py` | 154 |
| `write_2d_field_to_h5` | Function | `macromodel/timeseries.py` | 170 |
| `write_3d_field_to_h5` | Function | `macromodel/timeseries.py` | 186 |
| `create_multiindex_dataframe` | Function | `macromodel/timeseries.py` | 210 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Simple_run → Get_compatibility_mismatches` | cross_community | 5 |
| `Core_run → Get_compatibility_mismatches` | cross_community | 5 |
| `Write_to_h5 → Create_multiindex_dataframe` | cross_community | 4 |
| `Simple_run → Set_seed` | cross_community | 4 |
| `Simple_run → Run_prehooks` | cross_community | 4 |
| `Simple_run → Sync_central_banks` | cross_community | 4 |
| `Simple_run → Run_posthooks` | cross_community | 4 |
| `Simple_run → Step` | cross_community | 4 |
| `Simulator → Run_prehooks` | cross_community | 4 |
| `Simulator → Sync_central_banks` | cross_community | 4 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Sampler | 1 calls |

## How to Explore

1. `gitnexus_context({name: "main"})` — see callers and callees
2. `gitnexus_query({query: "macromodel"})` — find related execution flows
3. Read key files listed above for implementation details
