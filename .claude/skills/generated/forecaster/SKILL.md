---
name: forecaster
description: "Skill for the Forecaster area of macro-main. 15 symbols across 2 files."
---

# Forecaster

15 symbols | 2 files | Cohesion: 100%

## When to Use

- Working with code in `macromodel/`
- Understanding how check_len, forecast, forecast work
- Modifying forecaster-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `macromodel/forecaster/forecaster.py` | check_len, forecast, forecast, forecast, get_noise_variance (+6) |
| `tests/test_macromodel/unit/test_forecaster/test_forecaster.py` | test__check_len, test__forecast, test__forecast_increasing, test__forecast |

## Entry Points

Start here when exploring this area:

- **`check_len`** (Function) — `macromodel/forecaster/forecaster.py:34`
- **`forecast`** (Function) — `macromodel/forecaster/forecaster.py:109`
- **`forecast`** (Function) — `macromodel/forecaster/forecaster.py:155`
- **`forecast`** (Function) — `macromodel/forecaster/forecaster.py:179`
- **`get_noise_variance`** (Function) — `macromodel/forecaster/forecaster.py:214`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `Forecaster` | Class | `macromodel/forecaster/forecaster.py` | 47 |
| `ConstantForecaster` | Class | `macromodel/forecaster/forecaster.py` | 70 |
| `OLSForecaster` | Class | `macromodel/forecaster/forecaster.py` | 102 |
| `ImplementedAutoregForecaster` | Class | `macromodel/forecaster/forecaster.py` | 129 |
| `ManualAutoregForecaster` | Class | `macromodel/forecaster/forecaster.py` | 171 |
| `check_len` | Function | `macromodel/forecaster/forecaster.py` | 34 |
| `forecast` | Function | `macromodel/forecaster/forecaster.py` | 109 |
| `forecast` | Function | `macromodel/forecaster/forecaster.py` | 155 |
| `forecast` | Function | `macromodel/forecaster/forecaster.py` | 179 |
| `get_noise_variance` | Function | `macromodel/forecaster/forecaster.py` | 214 |
| `rfvar3` | Function | `macromodel/forecaster/forecaster.py` | 229 |
| `test__check_len` | Function | `tests/test_macromodel/unit/test_forecaster/test_forecaster.py` | 10 |
| `test__forecast` | Function | `tests/test_macromodel/unit/test_forecaster/test_forecaster.py` | 17 |
| `test__forecast_increasing` | Function | `tests/test_macromodel/unit/test_forecaster/test_forecaster.py` | 22 |
| `test__forecast` | Function | `tests/test_macromodel/unit/test_forecaster/test_forecaster.py` | 29 |

## How to Explore

1. `gitnexus_context({name: "check_len"})` — see callers and callees
2. `gitnexus_query({query: "forecaster"})` — find related execution flows
3. Read key files listed above for implementation details
