---
name: io-tables
description: "Skill for the Io_tables area of macro-main. 14 symbols across 2 files."
---

# Io_tables

14 symbols | 2 files | Cohesion: 100%

## When to Use

- Working with code in `macro_data/`
- Understanding how column_allc, capital_formation, capital_weights work
- Modifying io_tables-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `macro_data/readers/io_tables/wiod_reader.py` | column_allc, capital_formation, capital_weights, hh_consumption, hh_consumption_weights (+8) |
| `macro_data/readers/io_tables/util.py` | aggregate_df |

## Entry Points

Start here when exploring this area:

- **`column_allc`** (Function) — `macro_data/readers/io_tables/wiod_reader.py:238`
- **`capital_formation`** (Function) — `macro_data/readers/io_tables/wiod_reader.py:258`
- **`capital_weights`** (Function) — `macro_data/readers/io_tables/wiod_reader.py:274`
- **`hh_consumption`** (Function) — `macro_data/readers/io_tables/wiod_reader.py:291`
- **`hh_consumption_weights`** (Function) — `macro_data/readers/io_tables/wiod_reader.py:307`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `column_allc` | Function | `macro_data/readers/io_tables/wiod_reader.py` | 238 |
| `capital_formation` | Function | `macro_data/readers/io_tables/wiod_reader.py` | 258 |
| `capital_weights` | Function | `macro_data/readers/io_tables/wiod_reader.py` | 274 |
| `hh_consumption` | Function | `macro_data/readers/io_tables/wiod_reader.py` | 291 |
| `hh_consumption_weights` | Function | `macro_data/readers/io_tables/wiod_reader.py` | 307 |
| `govt_consumption` | Function | `macro_data/readers/io_tables/wiod_reader.py` | 324 |
| `govt_cons_weights` | Function | `macro_data/readers/io_tables/wiod_reader.py` | 340 |
| `from_csv` | Function | `macro_data/readers/io_tables/wiod_reader.py` | 89 |
| `read_csv` | Function | `macro_data/readers/io_tables/wiod_reader.py` | 115 |
| `agg_from_csv` | Function | `macro_data/readers/io_tables/wiod_reader.py` | 149 |
| `aggregate_io` | Function | `macro_data/readers/io_tables/wiod_reader.py` | 188 |
| `aggregate_df` | Function | `macro_data/readers/io_tables/util.py` | 3 |
| `intermediate_inputs` | Function | `macro_data/readers/io_tables/wiod_reader.py` | 357 |
| `intermediate_input_weights` | Function | `macro_data/readers/io_tables/wiod_reader.py` | 384 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Agg_from_csv → Aggregate_df` | intra_community | 3 |
| `Capital_weights → Column_allc` | intra_community | 3 |
| `Hh_consumption_weights → Column_allc` | intra_community | 3 |
| `Govt_cons_weights → Column_allc` | intra_community | 3 |

## How to Explore

1. `gitnexus_context({name: "column_allc"})` — see callers and callees
2. `gitnexus_query({query: "io_tables"})` — find related execution flows
3. Read key files listed above for implementation details
