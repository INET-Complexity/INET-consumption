---
name: population-data
description: "Skill for the Population_data area of macro-main. 14 symbols across 3 files."
---

# Population_data

14 symbols | 3 files | Cohesion: 93%

## When to Use

- Working with code in `macro_data/`
- Understanding how from_raw_data, test__compustat_firms_uses_configured_quarter, test__compustat_firms_converts_active_quarterly_flows_to_monthly work
- Modifying population_data-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `tests/test_macro_data/unit/test_readers/test_compustat_readers.py` | _write_firm_files, test__compustat_firms_uses_configured_quarter, test__compustat_firms_converts_active_quarterly_flows_to_monthly, test__compustat_firms_converts_active_quarterly_flows_to_bimonthly, _write_bank_file (+1) |
| `macro_data/readers/population_data/compustat_firms_reader.py` | _active_quarterly_flow_source_columns, _validate_time_unit, _convert_active_quarterly_flows_to_time_unit, from_raw_data |
| `macro_data/readers/population_data/compustat_banks_reader.py` | _active_quarterly_flow_source_columns, _validate_time_unit, _convert_active_quarterly_flows_to_time_unit, from_raw_data |

## Entry Points

Start here when exploring this area:

- **`from_raw_data`** (Function) — `macro_data/readers/population_data/compustat_firms_reader.py:183`
- **`test__compustat_firms_uses_configured_quarter`** (Function) — `tests/test_macro_data/unit/test_readers/test_compustat_readers.py:107`
- **`test__compustat_firms_converts_active_quarterly_flows_to_monthly`** (Function) — `tests/test_macro_data/unit/test_readers/test_compustat_readers.py:124`
- **`test__compustat_firms_converts_active_quarterly_flows_to_bimonthly`** (Function) — `tests/test_macro_data/unit/test_readers/test_compustat_readers.py:142`
- **`from_raw_data`** (Function) — `macro_data/readers/population_data/compustat_banks_reader.py:174`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `from_raw_data` | Function | `macro_data/readers/population_data/compustat_firms_reader.py` | 183 |
| `test__compustat_firms_uses_configured_quarter` | Function | `tests/test_macro_data/unit/test_readers/test_compustat_readers.py` | 107 |
| `test__compustat_firms_converts_active_quarterly_flows_to_monthly` | Function | `tests/test_macro_data/unit/test_readers/test_compustat_readers.py` | 124 |
| `test__compustat_firms_converts_active_quarterly_flows_to_bimonthly` | Function | `tests/test_macro_data/unit/test_readers/test_compustat_readers.py` | 142 |
| `from_raw_data` | Function | `macro_data/readers/population_data/compustat_banks_reader.py` | 174 |
| `test__compustat_banks_uses_configured_quarter_without_converting_inactive_flows` | Function | `tests/test_macro_data/unit/test_readers/test_compustat_readers.py` | 159 |
| `_active_quarterly_flow_source_columns` | Function | `macro_data/readers/population_data/compustat_firms_reader.py` | 122 |
| `_validate_time_unit` | Function | `macro_data/readers/population_data/compustat_firms_reader.py` | 130 |
| `_convert_active_quarterly_flows_to_time_unit` | Function | `macro_data/readers/population_data/compustat_firms_reader.py` | 135 |
| `_write_firm_files` | Function | `tests/test_macro_data/unit/test_readers/test_compustat_readers.py` | 8 |
| `_active_quarterly_flow_source_columns` | Function | `macro_data/readers/population_data/compustat_banks_reader.py` | 113 |
| `_validate_time_unit` | Function | `macro_data/readers/population_data/compustat_banks_reader.py` | 121 |
| `_convert_active_quarterly_flows_to_time_unit` | Function | `macro_data/readers/population_data/compustat_banks_reader.py` | 126 |
| `_write_bank_file` | Function | `tests/test_macro_data/unit/test_readers/test_compustat_readers.py` | 66 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `From_raw_data → _validate_time_unit` | intra_community | 3 |

## How to Explore

1. `gitnexus_context({name: "from_raw_data"})` — see callers and callees
2. `gitnexus_query({query: "population_data"})` — find related execution flows
3. Read key files listed above for implementation details
