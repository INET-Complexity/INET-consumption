---
name: credit-market
description: "Skill for the Credit_market area of macro-main. 14 symbols across 5 files."
---

# Credit_market

14 symbols | 5 files | Cohesion: 96%

## When to Use

- Working with code in `macromodel/`
- Understanding how compute_outstanding_loans_by_bank, compute_outstanding_short_term_firm_loans_by_bank, compute_outstanding_long_term_firm_loans_by_bank work
- Modifying credit_market-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `macromodel/markets/credit_market/credit_market.py` | compute_outstanding_loans_by_bank, compute_outstanding_short_term_firm_loans_by_bank, compute_outstanding_long_term_firm_loans_by_bank, compute_outstanding_household_consumption_loans_by_bank, compute_outstanding_mortgages_by_bank (+4) |
| `macromodel/agents/banks/banks.py` | update_loans, handle_insolvency |
| `tests/test_macromodel/unit/conftest.py` | test_credit_market |
| `macromodel/markets/credit_market/credit_market_ts.py` | create_credit_market_timeseries |
| `macromodel/agents/households/func/insolvency.py` | handle_insolvency |

## Entry Points

Start here when exploring this area:

- **`compute_outstanding_loans_by_bank`** (Function) — `macromodel/markets/credit_market/credit_market.py:449`
- **`compute_outstanding_short_term_firm_loans_by_bank`** (Function) — `macromodel/markets/credit_market/credit_market.py:462`
- **`compute_outstanding_long_term_firm_loans_by_bank`** (Function) — `macromodel/markets/credit_market/credit_market.py:470`
- **`compute_outstanding_household_consumption_loans_by_bank`** (Function) — `macromodel/markets/credit_market/credit_market.py:478`
- **`compute_outstanding_mortgages_by_bank`** (Function) — `macromodel/markets/credit_market/credit_market.py:486`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `compute_outstanding_loans_by_bank` | Function | `macromodel/markets/credit_market/credit_market.py` | 449 |
| `compute_outstanding_short_term_firm_loans_by_bank` | Function | `macromodel/markets/credit_market/credit_market.py` | 462 |
| `compute_outstanding_long_term_firm_loans_by_bank` | Function | `macromodel/markets/credit_market/credit_market.py` | 470 |
| `compute_outstanding_household_consumption_loans_by_bank` | Function | `macromodel/markets/credit_market/credit_market.py` | 478 |
| `compute_outstanding_mortgages_by_bank` | Function | `macromodel/markets/credit_market/credit_market.py` | 486 |
| `update_loans` | Function | `macromodel/agents/banks/banks.py` | 422 |
| `test_credit_market` | Function | `tests/test_macromodel/unit/conftest.py` | 269 |
| `create_credit_market_timeseries` | Function | `macromodel/markets/credit_market/credit_market_ts.py` | 20 |
| `from_pickled_market` | Function | `macromodel/markets/credit_market/credit_market.py` | 111 |
| `from_data` | Function | `macromodel/markets/credit_market/credit_market.py` | 188 |
| `remove_loans_to_households` | Function | `macromodel/markets/credit_market/credit_market.py` | 539 |
| `handle_insolvency` | Function | `macromodel/agents/households/func/insolvency.py` | 65 |
| `remove_loans_by_bank` | Function | `macromodel/markets/credit_market/credit_market.py` | 556 |
| `handle_insolvency` | Function | `macromodel/agents/banks/banks.py` | 526 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Rest_of_the_world | 1 calls |

## How to Explore

1. `gitnexus_context({name: "compute_outstanding_loans_by_bank"})` — see callers and callees
2. `gitnexus_query({query: "credit_market"})` — find related execution flows
3. Read key files listed above for implementation details
