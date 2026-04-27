---
name: economic-data
description: "Skill for the Economic_data area of macro-main. 40 symbols across 15 files."
---

# Economic_data

40 symbols | 15 files | Cohesion: 84%

## When to Use

- Working with code in `macro_data/`
- Understanding how prune_index, prune, prune work
- Modifying economic_data-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `macro_data/readers/economic_data/world_bank_reader.py` | prune, get_population, get_participation_rate, get_unemployment_rate, get_tau_vat (+4) |
| `macro_data/readers/economic_data/imf_reader.py` | prune, get_labour_stats, get_value, number_of_commercial_banks, number_of_commercial_depositors (+3) |
| `macro_data/readers/economic_data/exchange_rates.py` | prune, to_usd, from_usd, from_eur_to_lcu |
| `macro_data/readers/economic_data/policy_rates.py` | prune, __init__, country_code_switch |
| `macro_data/readers/economic_data/ecb_reader.py` | country_code_switch, preprocess_df, __init__ |
| `macro_data/readers/exogenous_data.py` | prepare_labour_stats, from_data_readers |
| `macro_data/readers/population_data/hfcs_reader.py` | from_csv, read_csv |
| `macro_data/readers/economic_data/ons_reader.py` | __init__, get_files_with_codes |
| `macro_data/readers/util/prune_util.py` | prune_index |
| `macro_data/readers/socioeconomic_data/wiod_sea_data.py` | prune |

## Entry Points

Start here when exploring this area:

- **`prune_index`** (Function) — `macro_data/readers/util/prune_util.py:9`
- **`prune`** (Function) — `macro_data/readers/socioeconomic_data/wiod_sea_data.py:212`
- **`prune`** (Function) — `macro_data/readers/economic_data/world_bank_reader.py:494`
- **`prune`** (Function) — `macro_data/readers/economic_data/policy_rates.py:177`
- **`prune`** (Function) — `macro_data/readers/economic_data/imf_reader.py:472`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `prune_index` | Function | `macro_data/readers/util/prune_util.py` | 9 |
| `prune` | Function | `macro_data/readers/socioeconomic_data/wiod_sea_data.py` | 212 |
| `prune` | Function | `macro_data/readers/economic_data/world_bank_reader.py` | 494 |
| `prune` | Function | `macro_data/readers/economic_data/policy_rates.py` | 177 |
| `prune` | Function | `macro_data/readers/economic_data/imf_reader.py` | 472 |
| `prune` | Function | `macro_data/readers/economic_data/exchange_rates.py` | 204 |
| `test_prune_index` | Function | `tests/test_macro_data/unit/test_readers/test_util/test_prune_util.py` | 5 |
| `build_social_benefits_reader_df` | Function | `run_model/src/helpers.py` | 153 |
| `prepare_labour_stats` | Function | `macro_data/readers/exogenous_data.py` | 161 |
| `get_population` | Function | `macro_data/readers/economic_data/world_bank_reader.py` | 202 |
| `get_participation_rate` | Function | `macro_data/readers/economic_data/world_bank_reader.py` | 223 |
| `get_unemployment_rate` | Function | `macro_data/readers/economic_data/world_bank_reader.py` | 398 |
| `get_labour_stats` | Function | `macro_data/readers/economic_data/imf_reader.py` | 410 |
| `from_data_readers` | Function | `macro_data/readers/exogenous_data.py` | 20 |
| `exogenous_data` | Function | `tests/test_macro_data/unit/conftest.py` | 200 |
| `get_tau_vat` | Function | `macro_data/readers/economic_data/world_bank_reader.py` | 252 |
| `get_inflation` | Function | `macro_data/readers/economic_data/world_bank_reader.py` | 427 |
| `get_tau_exp` | Function | `macro_data/readers/economic_data/world_bank_reader.py` | 484 |
| `test__exogenous` | Function | `tests/test_macro_data/unit/test_readers/test_exogenous.py` | 55 |
| `get_value` | Function | `macro_data/readers/economic_data/imf_reader.py` | 136 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `From_raw_data → To_usd` | cross_community | 5 |
| `From_raw_data → From_usd` | cross_community | 5 |
| `Create_all_exogenous_data → From_usd` | cross_community | 5 |
| `From_config → Get_na_growth_rates` | cross_community | 4 |
| `From_config → From_usd` | cross_community | 4 |
| `Build_social_benefits_reader_df → Get_historic_gdp` | cross_community | 4 |
| `From_readers → From_usd` | cross_community | 4 |
| `From_readers → From_usd` | cross_community | 4 |
| `From_config → Get_inflation` | cross_community | 3 |
| `From_config → Get_inflation` | cross_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Readers | 6 calls |
| Test_readers | 3 calls |

## How to Explore

1. `gitnexus_context({name: "prune_index"})` — see callers and callees
2. `gitnexus_query({query: "economic_data"})` — find related execution flows
3. Read key files listed above for implementation details
