---
name: synthetic-matching
description: "Skill for the Synthetic_matching area of macro-main. 16 symbols across 6 files."
---

# Synthetic_matching

16 symbols | 6 files | Cohesion: 91%

## When to Use

- Working with code in `macro_data/`
- Understanding how remove_outliers, set_housing_df, create_owners_df work
- Modifying synthetic_matching-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `macro_data/processing/synthetic_matching/matching_households_with_houses.py` | set_housing_df, create_owners_df, create_rental_df, housing_info_from_population, set_social_housing_renters (+1) |
| `macro_data/processing/synthetic_matching/matching_individuals_with_firms.py` | preprocess, find_optimal_matching, match_individuals_with_firms_country |
| `macro_data/processing/synthetic_population/synthetic_population.py` | set_individual_labour_inputs, set_income |
| `macro_data/processing/synthetic_matching/matching_households_with_banks.py` | match_households_with_banks_optimal, rescale |
| `macro_data/processing/synthetic_matching/matching_firms_with_banks.py` | match_firms_with_banks_optimal, rescale |
| `macro_data/util/clean_data.py` | remove_outliers |

## Entry Points

Start here when exploring this area:

- **`remove_outliers`** (Function) — `macro_data/util/clean_data.py:40`
- **`set_housing_df`** (Function) — `macro_data/processing/synthetic_matching/matching_households_with_houses.py:63`
- **`create_owners_df`** (Function) — `macro_data/processing/synthetic_matching/matching_households_with_houses.py:172`
- **`create_rental_df`** (Function) — `macro_data/processing/synthetic_matching/matching_households_with_houses.py:221`
- **`housing_info_from_population`** (Function) — `macro_data/processing/synthetic_matching/matching_households_with_houses.py:295`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `remove_outliers` | Function | `macro_data/util/clean_data.py` | 40 |
| `set_housing_df` | Function | `macro_data/processing/synthetic_matching/matching_households_with_houses.py` | 63 |
| `create_owners_df` | Function | `macro_data/processing/synthetic_matching/matching_households_with_houses.py` | 172 |
| `create_rental_df` | Function | `macro_data/processing/synthetic_matching/matching_households_with_houses.py` | 221 |
| `housing_info_from_population` | Function | `macro_data/processing/synthetic_matching/matching_households_with_houses.py` | 295 |
| `set_social_housing_renters` | Function | `macro_data/processing/synthetic_matching/matching_households_with_houses.py` | 342 |
| `match_renters_to_properties` | Function | `macro_data/processing/synthetic_matching/matching_households_with_houses.py` | 373 |
| `set_individual_labour_inputs` | Function | `macro_data/processing/synthetic_population/synthetic_population.py` | 211 |
| `set_income` | Function | `macro_data/processing/synthetic_population/synthetic_population.py` | 408 |
| `preprocess` | Function | `macro_data/processing/synthetic_matching/matching_individuals_with_firms.py` | 47 |
| `find_optimal_matching` | Function | `macro_data/processing/synthetic_matching/matching_individuals_with_firms.py` | 137 |
| `match_individuals_with_firms_country` | Function | `macro_data/processing/synthetic_matching/matching_individuals_with_firms.py` | 209 |
| `match_households_with_banks_optimal` | Function | `macro_data/processing/synthetic_matching/matching_households_with_banks.py` | 78 |
| `rescale` | Function | `macro_data/processing/synthetic_matching/matching_households_with_banks.py` | 163 |
| `match_firms_with_banks_optimal` | Function | `macro_data/processing/synthetic_matching/matching_firms_with_banks.py` | 76 |
| `rescale` | Function | `macro_data/processing/synthetic_matching/matching_firms_with_banks.py` | 161 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Match_individuals_with_firms_country → Set_income` | intra_community | 3 |
| `Set_housing_df → Set_social_housing_renters` | intra_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Synthetic_population | 1 calls |

## How to Explore

1. `gitnexus_context({name: "remove_outliers"})` — see callers and callees
2. `gitnexus_query({query: "synthetic_matching"})` — find related execution flows
3. Read key files listed above for implementation details
