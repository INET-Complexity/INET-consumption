---
name: agent
description: "Skill for the Agent area of macro-main. 11 symbols across 7 files."
---

# Agent

11 symbols | 7 files | Cohesion: 100%

## When to Use

- Working with code in `macromodel/`
- Understanding how initiate_ts, record, round_pos work
- Modifying agent-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `macromodel/agents/agent/agent.py` | __init__, initiate_ts, record, round_pos, round_pos2 |
| `macromodel/rest_of_the_world/rest_of_the_world.py` | __init__ |
| `macromodel/agents/individuals/individuals.py` | __init__ |
| `macromodel/agents/government_entities/government_entities.py` | __init__ |
| `macromodel/agents/central_government/central_government.py` | __init__ |
| `macromodel/agents/central_bank/central_bank.py` | __init__ |
| `macromodel/agents/banks/banks.py` | __init__ |

## Entry Points

Start here when exploring this area:

- **`initiate_ts`** (Function) — `macromodel/agents/agent/agent.py:84`
- **`record`** (Function) — `macromodel/agents/agent/agent.py:238`
- **`round_pos`** (Function) — `macromodel/agents/agent/agent.py:357`
- **`round_pos2`** (Function) — `macromodel/agents/agent/agent.py:372`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `initiate_ts` | Function | `macromodel/agents/agent/agent.py` | 84 |
| `record` | Function | `macromodel/agents/agent/agent.py` | 238 |
| `round_pos` | Function | `macromodel/agents/agent/agent.py` | 357 |
| `round_pos2` | Function | `macromodel/agents/agent/agent.py` | 372 |
| `__init__` | Function | `macromodel/rest_of_the_world/rest_of_the_world.py` | 69 |
| `__init__` | Function | `macromodel/agents/individuals/individuals.py` | 78 |
| `__init__` | Function | `macromodel/agents/government_entities/government_entities.py` | 56 |
| `__init__` | Function | `macromodel/agents/central_government/central_government.py` | 54 |
| `__init__` | Function | `macromodel/agents/central_bank/central_bank.py` | 56 |
| `__init__` | Function | `macromodel/agents/agent/agent.py` | 42 |
| `__init__` | Function | `macromodel/agents/banks/banks.py` | 63 |

## How to Explore

1. `gitnexus_context({name: "initiate_ts"})` — see callers and callees
2. `gitnexus_query({query: "agent"})` — find related execution flows
3. Read key files listed above for implementation details
