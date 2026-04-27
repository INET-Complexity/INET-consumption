---
name: synthetic-banks
description: "Skill for the Synthetic_banks area of macro-main. 41 symbols across 6 files."
---

# Synthetic_banks

41 symbols | 6 files | Cohesion: 92%

## When to Use

- Working with code in `macro_data/`
- Understanding how get_country_data, default_rate_df, get_policy_rates work
- Modifying synthetic_banks-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `macro_data/processing/synthetic_banks/synthetic_banks.py` | initialise_rates_profits_liabilities, set_initial_interest_rates, set_interest_received_from_loans, set_interest_received_from_deposits, set_profits (+11) |
| `macro_data/processing/synthetic_banks/default_synthetic_banks.py` | from_readers, from_readers_compustat, _periods_per_year, _convert_annual_rate_to_period, _convert_annual_rate_series_to_period (+7) |
| `macro_data/processing/synthetic_banks/rates_utils.py` | rates_dataframe, default_rate_values, fit_mortgage_models, fit_household_models, fit_firm_models (+1) |
| `macro_data/readers/economic_data/policy_rates.py` | default_rate_df, get_policy_rates, singapore_rates, costa_rica_rates, get_dates |
| `macro_data/readers/population_data/compustat_banks_reader.py` | get_country_data |
| `tests/test_macro_data/unit/test_processing/test_synthetic_banks/test_synthetic_banks.py` | test__create |

## Entry Points

Start here when exploring this area:

- **`get_country_data`** (Function) — `macro_data/readers/population_data/compustat_banks_reader.py:265`
- **`default_rate_df`** (Function) — `macro_data/readers/economic_data/policy_rates.py:13`
- **`get_policy_rates`** (Function) — `macro_data/readers/economic_data/policy_rates.py:62`
- **`singapore_rates`** (Function) — `macro_data/readers/economic_data/policy_rates.py:110`
- **`costa_rica_rates`** (Function) — `macro_data/readers/economic_data/policy_rates.py:134`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `SyntheticBanks` | Class | `macro_data/processing/synthetic_banks/synthetic_banks.py` | 36 |
| `DefaultSyntheticBanks` | Class | `macro_data/processing/synthetic_banks/default_synthetic_banks.py` | 44 |
| `get_country_data` | Function | `macro_data/readers/population_data/compustat_banks_reader.py` | 265 |
| `default_rate_df` | Function | `macro_data/readers/economic_data/policy_rates.py` | 13 |
| `get_policy_rates` | Function | `macro_data/readers/economic_data/policy_rates.py` | 62 |
| `singapore_rates` | Function | `macro_data/readers/economic_data/policy_rates.py` | 110 |
| `costa_rica_rates` | Function | `macro_data/readers/economic_data/policy_rates.py` | 134 |
| `get_dates` | Function | `macro_data/readers/economic_data/policy_rates.py` | 164 |
| `rates_dataframe` | Function | `macro_data/processing/synthetic_banks/rates_utils.py` | 29 |
| `from_readers` | Function | `macro_data/processing/synthetic_banks/default_synthetic_banks.py` | 151 |
| `from_readers_compustat` | Function | `macro_data/processing/synthetic_banks/default_synthetic_banks.py` | 278 |
| `initialise_rates` | Function | `macro_data/processing/synthetic_banks/default_synthetic_banks.py` | 471 |
| `test__create` | Function | `tests/test_macro_data/unit/test_processing/test_synthetic_banks/test_synthetic_banks.py` | 15 |
| `initialise_rates_profits_liabilities` | Function | `macro_data/processing/synthetic_banks/synthetic_banks.py` | 287 |
| `set_initial_interest_rates` | Function | `macro_data/processing/synthetic_banks/synthetic_banks.py` | 329 |
| `set_interest_received_from_loans` | Function | `macro_data/processing/synthetic_banks/synthetic_banks.py` | 371 |
| `set_interest_received_from_deposits` | Function | `macro_data/processing/synthetic_banks/synthetic_banks.py` | 388 |
| `set_profits` | Function | `macro_data/processing/synthetic_banks/synthetic_banks.py` | 411 |
| `set_corporate_taxes_paid` | Function | `macro_data/processing/synthetic_banks/synthetic_banks.py` | 424 |
| `set_market_share` | Function | `macro_data/processing/synthetic_banks/synthetic_banks.py` | 432 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `From_readers_compustat → _periods_per_year` | intra_community | 4 |
| `From_readers → From_usd` | cross_community | 4 |
| `From_readers → _periods_per_year` | intra_community | 4 |
| `From_readers → Get_country_data` | intra_community | 3 |
| `From_readers → Get_dates` | intra_community | 3 |
| `From_readers → Costa_rica_rates` | intra_community | 3 |
| `From_readers → Singapore_rates` | intra_community | 3 |
| `From_readers → Default_rate_df` | intra_community | 3 |
| `From_readers → Get_dates` | cross_community | 3 |
| `From_readers → Costa_rica_rates` | cross_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Readers | 1 calls |

## How to Explore

1. `gitnexus_context({name: "get_country_data"})` — see callers and callees
2. `gitnexus_query({query: "synthetic_banks"})` — find related execution flows
3. Read key files listed above for implementation details
