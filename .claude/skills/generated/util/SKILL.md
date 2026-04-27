---
name: util
description: "Skill for the Util area of macro-main. 25 symbols across 14 files."
---

# Util

25 symbols | 14 files | Cohesion: 82%

## When to Use

- Working with code in `macromodel/`
- Understanding how get_histogram, fillna, update_property_value work
- Modifying util-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `macromodel/util/loader.py` | get_country_agent_field_dataframe, load_1d_field_as_dataframe, load_2d_field_as_dataframe, load_3d_field_as_dataframe |
| `macromodel/util/get_histogram.py` | get_histogram, _degenerate_histogram, fillna |
| `macromodel/agents/firms/firm_ts.py` | from_data, create_firms_timeseries, get_n_firms_by_industry |
| `macro_data/readers/util/industry_extraction.py` | compile_industry_data, get_industry_vectors, fill_trade_data |
| `macromodel/markets/housing_market/housing_market.py` | update_property_value, reset |
| `tests/test_macro_data/unit/conftest.py` | industry_data, multic_industry_data |
| `macromodel/agents/households/households_ts.py` | create_households_timeseries |
| `macromodel/util/function_mapping.py` | update_functions |
| `macromodel/markets/labour_market/labour_market.py` | reset |
| `macromodel/markets/goods_market/goods_market.py` | reset |

## Entry Points

Start here when exploring this area:

- **`get_histogram`** (Function) — `macromodel/util/get_histogram.py:20`
- **`fillna`** (Function) — `macromodel/util/get_histogram.py:88`
- **`update_property_value`** (Function) — `macromodel/markets/housing_market/housing_market.py:276`
- **`create_households_timeseries`** (Function) — `macromodel/agents/households/households_ts.py:29`
- **`from_data`** (Function) — `macromodel/agents/firms/firm_ts.py:118`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `get_histogram` | Function | `macromodel/util/get_histogram.py` | 20 |
| `fillna` | Function | `macromodel/util/get_histogram.py` | 88 |
| `update_property_value` | Function | `macromodel/markets/housing_market/housing_market.py` | 276 |
| `create_households_timeseries` | Function | `macromodel/agents/households/households_ts.py` | 29 |
| `from_data` | Function | `macromodel/agents/firms/firm_ts.py` | 118 |
| `create_firms_timeseries` | Function | `macromodel/agents/firms/firm_ts.py` | 380 |
| `get_n_firms_by_industry` | Function | `macromodel/agents/firms/firm_ts.py` | 519 |
| `update_functions` | Function | `macromodel/util/function_mapping.py` | 126 |
| `reset` | Function | `macromodel/markets/labour_market/labour_market.py` | 180 |
| `reset` | Function | `macromodel/markets/housing_market/housing_market.py` | 192 |
| `reset` | Function | `macromodel/markets/goods_market/goods_market.py` | 197 |
| `reset` | Function | `macromodel/markets/credit_market/credit_market.py` | 173 |
| `reset` | Function | `macromodel/agents/banks/banks.py` | 195 |
| `industry_data` | Function | `tests/test_macro_data/unit/conftest.py` | 193 |
| `multic_industry_data` | Function | `tests/test_macro_data/unit/conftest.py` | 211 |
| `compile_industry_data` | Function | `macro_data/readers/util/industry_extraction.py` | 12 |
| `get_industry_vectors` | Function | `macro_data/readers/util/industry_extraction.py` | 79 |
| `fill_trade_data` | Function | `macro_data/readers/util/industry_extraction.py` | 161 |
| `get_country_agent_field_dataframe` | Function | `macromodel/util/loader.py` | 55 |
| `load_1d_field_as_dataframe` | Function | `macromodel/util/loader.py` | 128 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `From_pickled_market → Fillna` | cross_community | 4 |
| `From_pickled_market → _degenerate_histogram` | cross_community | 4 |
| `From_pickled_agent → Fillna` | cross_community | 4 |
| `From_pickled_agent → _degenerate_histogram` | cross_community | 4 |
| `From_pickled_agent → Fillna` | cross_community | 4 |
| `From_pickled_agent → _degenerate_histogram` | cross_community | 4 |
| `From_data → Fillna` | cross_community | 4 |
| `From_data → _degenerate_histogram` | cross_community | 4 |
| `Compute_observed_fraction_value_price → Fillna` | cross_community | 3 |
| `Compute_observed_fraction_value_price → _degenerate_histogram` | cross_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Readers | 3 calls |
| Rest_of_the_world | 1 calls |

## How to Explore

1. `gitnexus_context({name: "get_histogram"})` — see callers and callees
2. `gitnexus_query({query: "util"})` — find related execution flows
3. Read key files listed above for implementation details
