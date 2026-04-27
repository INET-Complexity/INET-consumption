---
name: readers
description: "Skill for the Readers area of macro-main. 55 symbols across 16 files."
---

# Readers

55 symbols | 16 files | Cohesion: 74%

## When to Use

- Working with code in `macro_data/`
- Understanding how reconcile_value_added, default_paths, from_raw_data work
- Modifying readers-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `macro_data/readers/default_readers.py` | default_paths, _time_unit_from_yearly_factor, from_raw_data, get_investment_year, get_investment_fractions (+8) |
| `macro_data/readers/icio_sea_matching.py` | reconcile_value_added, _reconcile_value_added, match_iot_with_sea, get_sea, _match_country_iot_with_sea (+4) |
| `tests/test_macro_data/unit/conftest.py` | readers, readers_disagg_can, all_readers, all_industries_readers, gen_multic_readers |
| `macro_data/readers/exogenous_data.py` | get_calibration_data, normalised_growth, compile_national_accounts_data, get_growth, prepare_inflation |
| `macro_data/readers/economic_data/world_bank_reader.py` | get_lcu_exports, get_historic_gdp, get_current_scaled_gdp, get_log_inflation |
| `macro_data/readers/socioeconomic_data/wiod_sea_data.py` | agg_from_csv, set_values_in_usd, get_values_in_usd |
| `macro_data/readers/economic_data/imf_reader.py` | from_data, get_na_growth_rates, get_inflation |
| `macro_data/readers/economic_data/exchange_rates.py` | from_csv, exchange_rates_dict, from_usd_to_lcu |
| `macro_data/readers/criticality_data/goods_criticality_reader.py` | from_csv, aggregate |
| `run_model/src/helpers.py` | _prepare_social_benefits_arx_regression_data, fit_social_benefits_arx_diagnostics |

## Entry Points

Start here when exploring this area:

- **`reconcile_value_added`** (Function) — `macro_data/readers/icio_sea_matching.py:161`
- **`default_paths`** (Function) — `macro_data/readers/default_readers.py:112`
- **`from_raw_data`** (Function) — `macro_data/readers/default_readers.py:208`
- **`get_investment_year`** (Function) — `macro_data/readers/default_readers.py:254`
- **`get_investment_fractions`** (Function) — `macro_data/readers/default_readers.py:510`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `reconcile_value_added` | Function | `macro_data/readers/icio_sea_matching.py` | 161 |
| `default_paths` | Function | `macro_data/readers/default_readers.py` | 112 |
| `from_raw_data` | Function | `macro_data/readers/default_readers.py` | 208 |
| `get_investment_year` | Function | `macro_data/readers/default_readers.py` | 254 |
| `get_investment_fractions` | Function | `macro_data/readers/default_readers.py` | 510 |
| `prune_icio_dict` | Function | `macro_data/readers/default_readers.py` | 763 |
| `readers` | Function | `tests/test_macro_data/unit/conftest.py` | 50 |
| `readers_disagg_can` | Function | `tests/test_macro_data/unit/conftest.py` | 66 |
| `all_readers` | Function | `tests/test_macro_data/unit/conftest.py` | 125 |
| `all_industries_readers` | Function | `tests/test_macro_data/unit/conftest.py` | 141 |
| `gen_multic_readers` | Function | `tests/test_macro_data/unit/conftest.py` | 157 |
| `agg_from_csv` | Function | `macro_data/readers/socioeconomic_data/wiod_sea_data.py` | 46 |
| `read_price_data` | Function | `macro_data/readers/emissions/emissions_reader.py` | 82 |
| `from_data` | Function | `macro_data/readers/economic_data/imf_reader.py` | 90 |
| `from_csv` | Function | `macro_data/readers/economic_data/exchange_rates.py` | 88 |
| `exchange_rates_dict` | Function | `macro_data/readers/economic_data/exchange_rates.py` | 105 |
| `from_csv` | Function | `macro_data/readers/criticality_data/goods_criticality_reader.py` | 10 |
| `aggregate` | Function | `macro_data/readers/criticality_data/goods_criticality_reader.py` | 16 |
| `test__prune_icio_dict` | Function | `tests/test_macro_data/unit/test_readers/test_default_readers.py` | 9 |
| `get_govt_debt_lcu` | Function | `macro_data/readers/default_readers.py` | 662 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `From_raw_data → To_usd` | cross_community | 5 |
| `From_raw_data → From_usd` | cross_community | 5 |
| `Create_all_exogenous_data → From_usd` | cross_community | 5 |
| `From_config → Get_na_growth_rates` | cross_community | 4 |
| `From_config → From_usd` | cross_community | 4 |
| `Build_social_benefits_reader_df → Get_historic_gdp` | cross_community | 4 |
| `Fit_social_benefits_arx_diagnostics → Get_historic_gdp` | cross_community | 4 |
| `From_readers → Get_historic_gdp` | cross_community | 4 |
| `From_readers → From_usd` | cross_community | 4 |
| `From_readers → From_usd` | cross_community | 4 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Economic_data | 4 calls |
| Population_data | 2 calls |

## How to Explore

1. `gitnexus_context({name: "reconcile_value_added"})` — see callers and callees
2. `gitnexus_query({query: "readers"})` — find related execution flows
3. Read key files listed above for implementation details
