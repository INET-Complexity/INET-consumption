---
name: unit
description: "Skill for the Unit area of macro-main. 35 symbols across 11 files."
---

# Unit

35 symbols | 11 files | Cohesion: 91%

## When to Use

- Working with code in `tests/`
- Understanding how read_country_conf, create_country_configurations, default_data_configuration work
- Modifying unit-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `tests/test_macro_data/unit/test_data_wrapper.py` | test__create, test__create_all_industries, test__create_us_can, test_create_us_can_all_industries, test__create_can_disagg (+4) |
| `tests/test_macrocalib/unit/conftest.py` | instantiate_datawrapper, configuration_updater, update_country_configuration, observer, country_data_array |
| `tests/test_macromodel/unit/conftest.py` | instantiate_datawrapper, instantiate_allind_datawrapper, instantiate_can_disagg_datawrapper, instantiate_can_provincial_datawrapper |
| `tests/test_macro_data/unit/conftest.py` | exogenous_industry_data, all_exogenous_data, readers_provincial_can, canada_disagg_config |
| `macro_data/configuration_utils.py` | read_country_conf, create_country_configurations, default_data_configuration |
| `tests/test_macro_data/unit/test_configuration_utils.py` | test__create_fra, test__create_can_error, test__create_can |
| `macro_data/readers/util/industry_extraction.py` | compile_exogenous_industry_data, get_row_industry_data |
| `tests/test_macro_data/unit/test_readers/test_default_readers.py` | test__get_benefits_inflation, test__create_exogenous_data |
| `macro_data/readers/exogenous_data.py` | create_all_exogenous_data |
| `macro_data/configuration/region.py` | from_code |

## Entry Points

Start here when exploring this area:

- **`read_country_conf`** (Function) — `macro_data/configuration_utils.py:49`
- **`create_country_configurations`** (Function) — `macro_data/configuration_utils.py:77`
- **`default_data_configuration`** (Function) — `macro_data/configuration_utils.py:161`
- **`instantiate_datawrapper`** (Function) — `tests/test_macromodel/unit/conftest.py:421`
- **`instantiate_allind_datawrapper`** (Function) — `tests/test_macromodel/unit/conftest.py:428`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `read_country_conf` | Function | `macro_data/configuration_utils.py` | 49 |
| `create_country_configurations` | Function | `macro_data/configuration_utils.py` | 77 |
| `default_data_configuration` | Function | `macro_data/configuration_utils.py` | 161 |
| `instantiate_datawrapper` | Function | `tests/test_macromodel/unit/conftest.py` | 421 |
| `instantiate_allind_datawrapper` | Function | `tests/test_macromodel/unit/conftest.py` | 428 |
| `instantiate_can_disagg_datawrapper` | Function | `tests/test_macromodel/unit/conftest.py` | 435 |
| `instantiate_datawrapper` | Function | `tests/test_macrocalib/unit/conftest.py` | 19 |
| `test__create_fra` | Function | `tests/test_macro_data/unit/test_configuration_utils.py` | 4 |
| `test__create_can_error` | Function | `tests/test_macro_data/unit/test_configuration_utils.py` | 8 |
| `test__create_can` | Function | `tests/test_macro_data/unit/test_configuration_utils.py` | 21 |
| `test__create` | Function | `tests/test_macro_data/unit/test_data_wrapper.py` | 15 |
| `test__create_all_industries` | Function | `tests/test_macro_data/unit/test_data_wrapper.py` | 61 |
| `test__create_us_can` | Function | `tests/test_macro_data/unit/test_data_wrapper.py` | 135 |
| `test_create_us_can_all_industries` | Function | `tests/test_macro_data/unit/test_data_wrapper.py` | 211 |
| `test__create_can_disagg` | Function | `tests/test_macro_data/unit/test_data_wrapper.py` | 314 |
| `test__create_can_provincial` | Function | `tests/test_macro_data/unit/test_data_wrapper.py` | 388 |
| `check_country_credit` | Function | `tests/test_macro_data/unit/test_data_wrapper.py` | 406 |
| `check_country_gdp` | Function | `tests/test_macro_data/unit/test_data_wrapper.py` | 428 |
| `check_country_rent_consistency` | Function | `tests/test_macro_data/unit/test_data_wrapper.py` | 437 |
| `create_all_exogenous_data` | Function | `macro_data/readers/exogenous_data.py` | 209 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Create_all_exogenous_data → From_usd` | cross_community | 5 |
| `Create_all_exogenous_data → Get_values_in_usd` | cross_community | 4 |
| `Create_all_exogenous_data → Get_inflation` | cross_community | 3 |
| `Create_all_exogenous_data → Get_log_inflation` | cross_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Readers | 5 calls |

## How to Explore

1. `gitnexus_context({name: "read_country_conf"})` — see callers and callees
2. `gitnexus_query({query: "unit"})` — find related execution flows
3. Read key files listed above for implementation details
