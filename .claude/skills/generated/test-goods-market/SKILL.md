---
name: test-goods-market
description: "Skill for the Test_goods_market area of macro-main. 11 symbols across 4 files."
---

# Test_goods_market

11 symbols | 4 files | Cohesion: 88%

## When to Use

- Working with code in `macromodel/`
- Understanding how prepare_selling_goods, prepare_selling_goods, set_goods_to_sell work
- Modifying test_goods_market-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `tests/test_macromodel/unit/test_markets/test_goods_market/test_goods_market.py` | create_test_transactor, create_test_transactors, check_things_adding_up, check_excess_demand, test__transaction |
| `macromodel/agents/agent/agent.py` | set_goods_to_sell, set_maximum_excess_demand, set_prices, set_seller_industries |
| `macromodel/rest_of_the_world/rest_of_the_world.py` | prepare_selling_goods |
| `macromodel/agents/government_entities/government_entities.py` | prepare_selling_goods |

## Entry Points

Start here when exploring this area:

- **`prepare_selling_goods`** (Function) — `macromodel/rest_of_the_world/rest_of_the_world.py:286`
- **`prepare_selling_goods`** (Function) — `macromodel/agents/government_entities/government_entities.py:249`
- **`set_goods_to_sell`** (Function) — `macromodel/agents/agent/agent.py:133`
- **`set_maximum_excess_demand`** (Function) — `macromodel/agents/agent/agent.py:141`
- **`set_prices`** (Function) — `macromodel/agents/agent/agent.py:149`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `prepare_selling_goods` | Function | `macromodel/rest_of_the_world/rest_of_the_world.py` | 286 |
| `prepare_selling_goods` | Function | `macromodel/agents/government_entities/government_entities.py` | 249 |
| `set_goods_to_sell` | Function | `macromodel/agents/agent/agent.py` | 133 |
| `set_maximum_excess_demand` | Function | `macromodel/agents/agent/agent.py` | 141 |
| `set_prices` | Function | `macromodel/agents/agent/agent.py` | 149 |
| `set_seller_industries` | Function | `macromodel/agents/agent/agent.py` | 157 |
| `create_test_transactor` | Function | `tests/test_macromodel/unit/test_markets/test_goods_market/test_goods_market.py` | 9 |
| `create_test_transactors` | Function | `tests/test_macromodel/unit/test_markets/test_goods_market/test_goods_market.py` | 35 |
| `check_things_adding_up` | Function | `tests/test_macromodel/unit/test_markets/test_goods_market/test_goods_market.py` | 90 |
| `check_excess_demand` | Function | `tests/test_macromodel/unit/test_markets/test_goods_market/test_goods_market.py` | 99 |
| `test__transaction` | Function | `tests/test_macromodel/unit/test_markets/test_goods_market/test_goods_market.py` | 123 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Update_planning_metrics → Set_goods_to_sell` | cross_community | 4 |
| `Update_planning_metrics → Set_prices` | cross_community | 4 |
| `Update_planning_metrics → Set_seller_industries` | cross_community | 4 |
| `Update_planning_metrics → Set_maximum_excess_demand` | cross_community | 4 |
| `Prepare_goods_market_clearing → Set_goods_to_sell` | cross_community | 3 |
| `Prepare_goods_market_clearing → Set_prices` | cross_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Rest_of_the_world | 6 calls |

## How to Explore

1. `gitnexus_context({name: "prepare_selling_goods"})` — see callers and callees
2. `gitnexus_query({query: "test_goods_market"})` — find related execution flows
3. Read key files listed above for implementation details
