---
name: synthetic-central-government
description: "Skill for the Synthetic_central_government area of macro-main. 11 symbols across 4 files."
---

# Synthetic_central_government

11 symbols | 4 files | Cohesion: 85%

## When to Use

- Working with code in `macro_data/`
- Understanding how get_central_gov_debt, from_readers, build_unemployment_model work
- Modifying synthetic_central_government-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `macro_data/processing/synthetic_central_government/default_synthetic_central_government.py` | from_readers, build_unemployment_model, build_other_benefits_model, DefaultSyntheticCGovernment, __init__ |
| `macro_data/processing/synthetic_central_government/synthetic_central_government.py` | SyntheticCentralGovernment, __init__, update_fields, set_revenue |
| `macro_data/readers/economic_data/world_bank_reader.py` | get_central_gov_debt |
| `tests/test_macro_data/unit/test_processing/test_synthetic_central_government/test_synthetic_central_government.py` | test__create |

## Entry Points

Start here when exploring this area:

- **`get_central_gov_debt`** (Function) — `macro_data/readers/economic_data/world_bank_reader.py:169`
- **`from_readers`** (Function) — `macro_data/processing/synthetic_central_government/default_synthetic_central_government.py:94`
- **`build_unemployment_model`** (Function) — `macro_data/processing/synthetic_central_government/default_synthetic_central_government.py:198`
- **`build_other_benefits_model`** (Function) — `macro_data/processing/synthetic_central_government/default_synthetic_central_government.py:231`
- **`test__create`** (Function) — `tests/test_macro_data/unit/test_processing/test_synthetic_central_government/test_synthetic_central_government.py:14`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `SyntheticCentralGovernment` | Class | `macro_data/processing/synthetic_central_government/synthetic_central_government.py` | 40 |
| `DefaultSyntheticCGovernment` | Class | `macro_data/processing/synthetic_central_government/default_synthetic_central_government.py` | 38 |
| `get_central_gov_debt` | Function | `macro_data/readers/economic_data/world_bank_reader.py` | 169 |
| `from_readers` | Function | `macro_data/processing/synthetic_central_government/default_synthetic_central_government.py` | 94 |
| `build_unemployment_model` | Function | `macro_data/processing/synthetic_central_government/default_synthetic_central_government.py` | 198 |
| `build_other_benefits_model` | Function | `macro_data/processing/synthetic_central_government/default_synthetic_central_government.py` | 231 |
| `test__create` | Function | `tests/test_macro_data/unit/test_processing/test_synthetic_central_government/test_synthetic_central_government.py` | 14 |
| `update_fields` | Function | `macro_data/processing/synthetic_central_government/synthetic_central_government.py` | 106 |
| `set_revenue` | Function | `macro_data/processing/synthetic_central_government/synthetic_central_government.py` | 184 |
| `__init__` | Function | `macro_data/processing/synthetic_central_government/synthetic_central_government.py` | 79 |
| `__init__` | Function | `macro_data/processing/synthetic_central_government/default_synthetic_central_government.py` | 68 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `From_readers → Get_historic_gdp` | cross_community | 4 |
| `From_readers → Get_log_inflation` | cross_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Readers | 4 calls |

## How to Explore

1. `gitnexus_context({name: "get_central_gov_debt"})` — see callers and callees
2. `gitnexus_query({query: "synthetic_central_government"})` — find related execution flows
3. Read key files listed above for implementation details
