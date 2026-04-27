---
name: synthetic-credit-market
description: "Skill for the Synthetic_credit_market area of macro-main. 12 symbols across 2 files."
---

# Synthetic_credit_market

12 symbols | 2 files | Cohesion: 100%

## When to Use

- Working with code in `macro_data/`
- Understanding how create_from_agents, from_agent_data, from_agent_data work
- Modifying synthetic_credit_market-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `macro_data/processing/synthetic_credit_market/loan_data.py` | from_agent_data, from_agent_data, from_agent_data, from_agent_data, from_agent_data (+6) |
| `macro_data/processing/synthetic_credit_market/synthetic_credit_market.py` | create_from_agents |

## Entry Points

Start here when exploring this area:

- **`create_from_agents`** (Function) — `macro_data/processing/synthetic_credit_market/synthetic_credit_market.py:90`
- **`from_agent_data`** (Function) — `macro_data/processing/synthetic_credit_market/loan_data.py:86`
- **`from_agent_data`** (Function) — `macro_data/processing/synthetic_credit_market/loan_data.py:137`
- **`from_agent_data`** (Function) — `macro_data/processing/synthetic_credit_market/loan_data.py:179`
- **`from_agent_data`** (Function) — `macro_data/processing/synthetic_credit_market/loan_data.py:234`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `LoanData` | Class | `macro_data/processing/synthetic_credit_market/loan_data.py` | 35 |
| `LongtermLoans` | Class | `macro_data/processing/synthetic_credit_market/loan_data.py` | 72 |
| `ShorttermLoans` | Class | `macro_data/processing/synthetic_credit_market/loan_data.py` | 123 |
| `ConsumptionExpansionLoans` | Class | `macro_data/processing/synthetic_credit_market/loan_data.py` | 165 |
| `PaydayLoans` | Class | `macro_data/processing/synthetic_credit_market/loan_data.py` | 220 |
| `MortgageLoans` | Class | `macro_data/processing/synthetic_credit_market/loan_data.py` | 260 |
| `create_from_agents` | Function | `macro_data/processing/synthetic_credit_market/synthetic_credit_market.py` | 90 |
| `from_agent_data` | Function | `macro_data/processing/synthetic_credit_market/loan_data.py` | 86 |
| `from_agent_data` | Function | `macro_data/processing/synthetic_credit_market/loan_data.py` | 137 |
| `from_agent_data` | Function | `macro_data/processing/synthetic_credit_market/loan_data.py` | 179 |
| `from_agent_data` | Function | `macro_data/processing/synthetic_credit_market/loan_data.py` | 234 |
| `from_agent_data` | Function | `macro_data/processing/synthetic_credit_market/loan_data.py` | 274 |

## How to Explore

1. `gitnexus_context({name: "create_from_agents"})` — see callers and callees
2. `gitnexus_query({query: "synthetic_credit_market"})` — find related execution flows
3. Read key files listed above for implementation details
