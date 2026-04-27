---
name: synthetic-firms
description: "Skill for the Synthetic_firms area of macro-main. 26 symbols across 6 files."
---

# Synthetic_firms

26 symbols | 6 files | Cohesion: 86%

## When to Use

- Working with code in `macro_data/`
- Understanding how get_firm_data, get_firm_size_zetas, function_parameters_dependent_initialisation work
- Modifying synthetic_firms-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `macro_data/processing/synthetic_firms/default_synthetic_firms.py` | from_readers, reset_function_parameters, set_taxes_paid_on_production, set_interest_paid, set_firm_profits (+6) |
| `macro_data/processing/synthetic_firms/firm_tools.py` | function_parameters_dependent_initialisation, draw_industry_firm_sizes, distribute_industry_employee_remainder, add_number_employees_compustat, add_number_employees_random (+4) |
| `tests/test_macro_data/unit/test_processing/test_synthetic_firms/test_synthetic_firms.py` | test__create, test__create_multic |
| `macro_data/processing/synthetic_firms/synthetic_firms.py` | SyntheticFirms, __init__ |
| `macro_data/readers/population_data/compustat_firms_reader.py` | get_firm_data |
| `macro_data/readers/economic_data/ons_reader.py` | get_firm_size_zetas |

## Entry Points

Start here when exploring this area:

- **`get_firm_data`** (Function) — `macro_data/readers/population_data/compustat_firms_reader.py:292`
- **`get_firm_size_zetas`** (Function) — `macro_data/readers/economic_data/ons_reader.py:102`
- **`function_parameters_dependent_initialisation`** (Function) — `macro_data/processing/synthetic_firms/firm_tools.py:440`
- **`from_readers`** (Function) — `macro_data/processing/synthetic_firms/default_synthetic_firms.py:135`
- **`reset_function_parameters`** (Function) — `macro_data/processing/synthetic_firms/default_synthetic_firms.py:295`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `SyntheticFirms` | Class | `macro_data/processing/synthetic_firms/synthetic_firms.py` | 42 |
| `DefaultSyntheticFirms` | Class | `macro_data/processing/synthetic_firms/default_synthetic_firms.py` | 53 |
| `get_firm_data` | Function | `macro_data/readers/population_data/compustat_firms_reader.py` | 292 |
| `get_firm_size_zetas` | Function | `macro_data/readers/economic_data/ons_reader.py` | 102 |
| `function_parameters_dependent_initialisation` | Function | `macro_data/processing/synthetic_firms/firm_tools.py` | 440 |
| `from_readers` | Function | `macro_data/processing/synthetic_firms/default_synthetic_firms.py` | 135 |
| `reset_function_parameters` | Function | `macro_data/processing/synthetic_firms/default_synthetic_firms.py` | 295 |
| `test__create` | Function | `tests/test_macro_data/unit/test_processing/test_synthetic_firms/test_synthetic_firms.py` | 16 |
| `test__create_multic` | Function | `tests/test_macro_data/unit/test_processing/test_synthetic_firms/test_synthetic_firms.py` | 107 |
| `set_taxes_paid_on_production` | Function | `macro_data/processing/synthetic_firms/default_synthetic_firms.py` | 328 |
| `set_interest_paid` | Function | `macro_data/processing/synthetic_firms/default_synthetic_firms.py` | 335 |
| `set_firm_profits` | Function | `macro_data/processing/synthetic_firms/default_synthetic_firms.py` | 359 |
| `set_unit_costs` | Function | `macro_data/processing/synthetic_firms/default_synthetic_firms.py` | 394 |
| `set_corporate_taxes_paid` | Function | `macro_data/processing/synthetic_firms/default_synthetic_firms.py` | 423 |
| `set_firm_debt_installments` | Function | `macro_data/processing/synthetic_firms/default_synthetic_firms.py` | 426 |
| `set_additional_initial_conditions` | Function | `macro_data/processing/synthetic_firms/default_synthetic_firms.py` | 434 |
| `draw_industry_firm_sizes` | Function | `macro_data/processing/synthetic_firms/firm_tools.py` | 37 |
| `distribute_industry_employee_remainder` | Function | `macro_data/processing/synthetic_firms/firm_tools.py` | 95 |
| `add_number_employees_compustat` | Function | `macro_data/processing/synthetic_firms/firm_tools.py` | 122 |
| `add_number_employees_random` | Function | `macro_data/processing/synthetic_firms/firm_tools.py` | 179 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `From_readers → Draw_industry_firm_sizes` | cross_community | 4 |
| `From_readers → Distribute_industry_employee_remainder` | cross_community | 4 |
| `From_readers → From_usd` | cross_community | 3 |
| `From_readers → Add_wages` | cross_community | 3 |
| `From_readers → Add_production` | cross_community | 3 |
| `Initialise_basic_firm_fields_compustat → Distribute_industry_employee_remainder` | cross_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Readers | 1 calls |

## How to Explore

1. `gitnexus_context({name: "get_firm_data"})` — see callers and callees
2. `gitnexus_query({query: "synthetic_firms"})` — find related execution flows
3. Read key files listed above for implementation details
