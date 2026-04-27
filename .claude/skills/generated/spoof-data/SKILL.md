---
name: spoof-data
description: "Skill for the Spoof_data area of macro-main. 46 symbols across 7 files."
---

# Spoof_data

46 symbols | 7 files | Cohesion: 98%

## When to Use

- Working with code in `spoof_data/`
- Understanding how load_data, get_preserve_columns, spoof_categorical_column work
- Modifying spoof_data-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `spoof_data/generate_spoofed_hfcs.py` | load_data, get_preserve_columns, spoof_categorical_column, spoof_numerical_column, spoof_age_income_correlated (+10) |
| `spoof_data/explore_hfcs_data.py` | load_data, analyze_column_types, print_summary, print_key_columns, identify_spoofing_strategy (+4) |
| `spoof_data/generate_spoofed_compustat.py` | generate_fake_company_name, spoof_numerical_column, spoof_firms_annual, spoof_firms_quarterly, spoof_banks (+3) |
| `spoof_data/validate_compustat_coherence.py` | validate_firms_data, validate_banks_data, validate_cross_file_consistency, main |
| `spoof_data/explore_compustat_data.py` | analyze_dataframe, analyze_relationships, generate_spoofing_strategy, main |
| `spoof_data/validate_hfcs_coherence.py` | load_data, run_all_checks, main |
| `spoof_data/compare_original_vs_spoofed.py` | compare_distributions, compare_col, main |

## Entry Points

Start here when exploring this area:

- **`load_data`** (Function) — `spoof_data/generate_spoofed_hfcs.py:57`
- **`get_preserve_columns`** (Function) — `spoof_data/generate_spoofed_hfcs.py:76`
- **`spoof_categorical_column`** (Function) — `spoof_data/generate_spoofed_hfcs.py:93`
- **`spoof_numerical_column`** (Function) — `spoof_data/generate_spoofed_hfcs.py:136`
- **`spoof_age_income_correlated`** (Function) — `spoof_data/generate_spoofed_hfcs.py:244`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `load_data` | Function | `spoof_data/generate_spoofed_hfcs.py` | 57 |
| `get_preserve_columns` | Function | `spoof_data/generate_spoofed_hfcs.py` | 76 |
| `spoof_categorical_column` | Function | `spoof_data/generate_spoofed_hfcs.py` | 93 |
| `spoof_numerical_column` | Function | `spoof_data/generate_spoofed_hfcs.py` | 136 |
| `spoof_age_income_correlated` | Function | `spoof_data/generate_spoofed_hfcs.py` | 244 |
| `spoof_flag_columns` | Function | `spoof_data/generate_spoofed_hfcs.py` | 288 |
| `spoof_p1` | Function | `spoof_data/generate_spoofed_hfcs.py` | 318 |
| `spoof_h1` | Function | `spoof_data/generate_spoofed_hfcs.py` | 362 |
| `spoof_d1` | Function | `spoof_data/generate_spoofed_hfcs.py` | 415 |
| `spoof_main_residence_value` | Function | `spoof_data/generate_spoofed_hfcs.py` | 473 |
| `generate_spoofed_data` | Function | `spoof_data/generate_spoofed_hfcs.py` | 518 |
| `load_data` | Function | `spoof_data/explore_hfcs_data.py` | 38 |
| `analyze_column_types` | Function | `spoof_data/explore_hfcs_data.py` | 46 |
| `print_summary` | Function | `spoof_data/explore_hfcs_data.py` | 134 |
| `print_key_columns` | Function | `spoof_data/explore_hfcs_data.py` | 145 |
| `identify_spoofing_strategy` | Function | `spoof_data/explore_hfcs_data.py` | 189 |
| `check_special_relationships` | Function | `spoof_data/explore_hfcs_data.py` | 245 |
| `save_results` | Function | `spoof_data/explore_hfcs_data.py` | 263 |
| `run_exploration` | Function | `spoof_data/explore_hfcs_data.py` | 283 |
| `main` | Function | `spoof_data/explore_hfcs_data.py` | 306 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Main → Get_preserve_columns` | cross_community | 4 |
| `Main → Spoof_age_income_correlated` | cross_community | 4 |
| `Main → Spoof_categorical_column` | cross_community | 4 |
| `Main → Spoof_numerical_column` | cross_community | 4 |
| `Main → Generate_fake_company_name` | intra_community | 4 |
| `Main → Spoof_numerical_column` | intra_community | 4 |
| `Main → Load_data` | cross_community | 3 |
| `Main → Compare_column` | intra_community | 3 |
| `Main → Load_data` | intra_community | 3 |
| `Generate_spoofed_data → Spoof_categorical_column` | intra_community | 3 |

## How to Explore

1. `gitnexus_context({name: "load_data"})` — see callers and callees
2. `gitnexus_query({query: "spoof_data"})` — find related execution flows
3. Read key files listed above for implementation details
