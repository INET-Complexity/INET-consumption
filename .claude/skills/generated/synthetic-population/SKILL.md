---
name: synthetic-population
description: "Skill for the Synthetic_population area of macro-main. 31 symbols across 6 files."
---

# Synthetic_population

31 symbols | 6 files | Cohesion: 80%

## When to Use

- Working with code in `macro_data/`
- Understanding how process_individual_data, fill_missing_gender, fill_individual_labour_status work
- Modifying synthetic_population-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `macro_data/processing/synthetic_population/hfcs_individual_tools.py` | process_individual_data, fill_missing_gender, fill_individual_labour_status, fill_individual_nace, reassign_industry (+10) |
| `macro_data/processing/synthetic_population/hfcs_household_tools.py` | set_household_housing_data, fix_rent, fix_property_values, fill_rent, rescale_monthly_hfcs_cash_flows (+2) |
| `macro_data/processing/synthetic_population/synthetic_population.py` | normalise_household_investment, get_current_hh_investment_by_industry, add_emissions, default_target_investment, SyntheticPopulation |
| `macro_data/processing/synthetic_population/utils.py` | ensure_minimum_workers_in_industries, count_employees |
| `macro_data/util/imputation.py` | apply_iterative_imputer |
| `macro_data/processing/synthetic_population/hfcs_synthetic_population.py` | SyntheticHFCSPopulation |

## Entry Points

Start here when exploring this area:

- **`process_individual_data`** (Function) — `macro_data/processing/synthetic_population/hfcs_individual_tools.py:9`
- **`fill_missing_gender`** (Function) — `macro_data/processing/synthetic_population/hfcs_individual_tools.py:97`
- **`fill_individual_labour_status`** (Function) — `macro_data/processing/synthetic_population/hfcs_individual_tools.py:162`
- **`fill_individual_nace`** (Function) — `macro_data/processing/synthetic_population/hfcs_individual_tools.py:375`
- **`reassign_industry`** (Function) — `macro_data/processing/synthetic_population/hfcs_individual_tools.py:476`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `SyntheticPopulation` | Class | `macro_data/processing/synthetic_population/synthetic_population.py` | 101 |
| `SyntheticHFCSPopulation` | Class | `macro_data/processing/synthetic_population/hfcs_synthetic_population.py` | 98 |
| `process_individual_data` | Function | `macro_data/processing/synthetic_population/hfcs_individual_tools.py` | 9 |
| `fill_missing_gender` | Function | `macro_data/processing/synthetic_population/hfcs_individual_tools.py` | 97 |
| `fill_individual_labour_status` | Function | `macro_data/processing/synthetic_population/hfcs_individual_tools.py` | 162 |
| `fill_individual_nace` | Function | `macro_data/processing/synthetic_population/hfcs_individual_tools.py` | 375 |
| `reassign_industry` | Function | `macro_data/processing/synthetic_population/hfcs_individual_tools.py` | 476 |
| `select_employed_in_industry` | Function | `macro_data/processing/synthetic_population/hfcs_individual_tools.py` | 537 |
| `set_individual_unemployed_income` | Function | `macro_data/processing/synthetic_population/hfcs_individual_tools.py` | 609 |
| `set_individual_activity_status` | Function | `macro_data/processing/synthetic_population/hfcs_individual_tools.py` | 183 |
| `decrease_unemployment_rate` | Function | `macro_data/processing/synthetic_population/hfcs_individual_tools.py` | 231 |
| `increase_unemployment_rate` | Function | `macro_data/processing/synthetic_population/hfcs_individual_tools.py` | 266 |
| `decrease_participation_rate` | Function | `macro_data/processing/synthetic_population/hfcs_individual_tools.py` | 299 |
| `increase_participation_rate` | Function | `macro_data/processing/synthetic_population/hfcs_individual_tools.py` | 338 |
| `set_household_housing_data` | Function | `macro_data/processing/synthetic_population/hfcs_household_tools.py` | 102 |
| `fix_rent` | Function | `macro_data/processing/synthetic_population/hfcs_household_tools.py` | 169 |
| `fix_property_values` | Function | `macro_data/processing/synthetic_population/hfcs_household_tools.py` | 203 |
| `fill_rent` | Function | `macro_data/processing/synthetic_population/hfcs_household_tools.py` | 231 |
| `rescale_monthly_hfcs_cash_flows` | Function | `macro_data/processing/synthetic_population/hfcs_household_tools.py` | 280 |
| `apply_iterative_imputer` | Function | `macro_data/util/imputation.py` | 44 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Process_individual_data → Apply_iterative_imputer` | cross_community | 3 |
| `Set_household_housing_data → Apply_iterative_imputer` | cross_community | 3 |
| `Add_emissions → Default_target_investment` | intra_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Synthetic_matching | 1 calls |

## How to Explore

1. `gitnexus_context({name: "process_individual_data"})` — see callers and callees
2. `gitnexus_query({query: "synthetic_population"})` — find related execution flows
3. Read key files listed above for implementation details
