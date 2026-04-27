---
name: rest-of-the-world
description: "Skill for the Rest_of_the_world area of macro-main. 35 symbols across 18 files."
---

# Rest_of_the_world

35 symbols | 18 files | Cohesion: 83%

## When to Use

- Working with code in `macromodel/`
- Understanding how functions_from_model, create_rest_of_the_world_timeseries, from_pickled_row work
- Modifying rest_of_the_world-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `macromodel/rest_of_the_world/rest_of_the_world.py` | from_pickled_row, RestOfTheWorld, prepare_buying_goods, prepare_goods_market_clearing, update_planning_metrics (+1) |
| `macromodel/agents/government_entities/government_entities.py` | from_pickled_agent, GovernmentEntities, prepare_buying_goods, prepare_goods_market_clearing, reset |
| `macromodel/agents/agent/agent.py` | Agent, set_goods_to_buy, set_exchange_rate, gen_reset |
| `macromodel/agents/central_government/central_government.py` | from_pickled_agent, CentralGovernment, reset |
| `macromodel/agents/central_bank/central_bank.py` | from_pickled_agent, CentralBank, reset |
| `macromodel/agents/individuals/individuals.py` | Individuals, reset |
| `macromodel/util/function_mapping.py` | functions_from_model |
| `macromodel/rest_of_the_world/rest_of_the_world_ts.py` | create_rest_of_the_world_timeseries |
| `tests/test_macromodel/unit/conftest.py` | test_goods_market |
| `macromodel/markets/goods_market/goods_market_ts.py` | create_goods_market_timeseries |

## Entry Points

Start here when exploring this area:

- **`functions_from_model`** (Function) — `macromodel/util/function_mapping.py:93`
- **`create_rest_of_the_world_timeseries`** (Function) — `macromodel/rest_of_the_world/rest_of_the_world_ts.py:25`
- **`from_pickled_row`** (Function) — `macromodel/rest_of_the_world/rest_of_the_world.py:126`
- **`test_goods_market`** (Function) — `tests/test_macromodel/unit/conftest.py:336`
- **`create_goods_market_timeseries`** (Function) — `macromodel/markets/goods_market/goods_market_ts.py:13`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `RestOfTheWorld` | Class | `macromodel/rest_of_the_world/rest_of_the_world.py` | 46 |
| `Households` | Class | `macromodel/agents/households/households.py` | 41 |
| `Individuals` | Class | `macromodel/agents/individuals/individuals.py` | 45 |
| `GovernmentEntities` | Class | `macromodel/agents/government_entities/government_entities.py` | 32 |
| `Firms` | Class | `macromodel/agents/firms/firms.py` | 17 |
| `CentralGovernment` | Class | `macromodel/agents/central_government/central_government.py` | 32 |
| `CentralBank` | Class | `macromodel/agents/central_bank/central_bank.py` | 30 |
| `Agent` | Class | `macromodel/agents/agent/agent.py` | 19 |
| `Banks` | Class | `macromodel/agents/banks/banks.py` | 32 |
| `functions_from_model` | Function | `macromodel/util/function_mapping.py` | 93 |
| `create_rest_of_the_world_timeseries` | Function | `macromodel/rest_of_the_world/rest_of_the_world_ts.py` | 25 |
| `from_pickled_row` | Function | `macromodel/rest_of_the_world/rest_of_the_world.py` | 126 |
| `test_goods_market` | Function | `tests/test_macromodel/unit/conftest.py` | 336 |
| `create_goods_market_timeseries` | Function | `macromodel/markets/goods_market/goods_market_ts.py` | 13 |
| `from_data` | Function | `macromodel/markets/goods_market/goods_market.py` | 101 |
| `create_government_entities_timeseries` | Function | `macromodel/agents/government_entities/government_entities_ts.py` | 23 |
| `from_pickled_agent` | Function | `macromodel/agents/government_entities/government_entities.py` | 95 |
| `create_central_government_timeseries` | Function | `macromodel/agents/central_government/central_government_ts.py` | 22 |
| `from_pickled_agent` | Function | `macromodel/agents/central_government/central_government.py` | 85 |
| `create_central_bank_timeseries` | Function | `macromodel/agents/central_bank/central_bank_ts.py` | 19 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Update_planning_metrics → Set_goods_to_buy` | intra_community | 4 |
| `Update_planning_metrics → Set_goods_to_sell` | cross_community | 4 |
| `Update_planning_metrics → Set_prices` | cross_community | 4 |
| `Update_planning_metrics → Set_seller_industries` | cross_community | 4 |
| `Update_planning_metrics → Set_maximum_excess_demand` | cross_community | 4 |
| `Prepare_goods_market_clearing → Set_goods_to_buy` | intra_community | 3 |
| `Prepare_goods_market_clearing → Set_goods_to_sell` | cross_community | 3 |
| `Prepare_goods_market_clearing → Set_prices` | cross_community | 3 |
| `Update_planning_metrics → Set_exchange_rate` | intra_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Util | 5 calls |
| Test_goods_market | 2 calls |

## How to Explore

1. `gitnexus_context({name: "functions_from_model"})` — see callers and callees
2. `gitnexus_query({query: "rest_of_the_world"})` — find related execution flows
3. Read key files listed above for implementation details
