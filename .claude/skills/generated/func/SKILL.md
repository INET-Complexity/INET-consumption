---
name: func
description: "Skill for the Func area of macro-main. 306 symbols across 67 files."
---

# Func

306 symbols | 67 files | Cohesion: 98%

## When to Use

- Working with code in `macromodel/`
- Understanding how create_test_params, test_always_returns_zero_investment, test_ignores_all_parameters work
- Modifying func-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `tests/test_macromodel/unit/test_agents/test_firms/func/test_productivity_investment_planner.py` | create_test_params, test_always_returns_zero_investment, test_ignores_all_parameters, test_basic_investment_calculation_with_cost_savings, test_higher_unit_costs_encourage_investment (+15) |
| `tests/test_macromodel/unit/test_agents/test_firms/func/test_productivity_growth.py` | test_base_growth_only, test_investment_driven_growth, test_zero_production_handling, test_stochastic_shocks, test_zero_shock_std (+7) |
| `macromodel/agents/households/func/consumption.py` | compute_target_consumption, _compute_ces_weights, _compute_target_consumption_ces, compute_target_consumption, _compute_target_consumption (+7) |
| `tests/test_macromodel/unit/test_agents/test_firms/func/test_technical_coefficients_growth.py` | test_intermediate_growth_calculation, test_diminishing_returns, test_zero_production_handling, test_negative_investment, test_empty_arrays (+7) |
| `macromodel/economy/func/inflation.py` | InflationForecasting, InflationForecastingConstant, InflationForecastingOLS, InflationImplementedForecastingAutoReg, InflationManualForecastingAutoReg (+7) |
| `macromodel/economy/func/growth.py` | GrowthForecasting, GrowthForecastingConstant, GrowthForecastingOLS, GrowthImplementedForecastingAutoReg, GrowthManualForecastingAutoReg (+7) |
| `macromodel/agents/firms/func/technical_coefficients_growth.py` | compute_intermediate_multiplier_growth, TechnicalCoefficientsGrowth, NoOpTechnicalGrowth, SimpleTechnicalGrowth, __init__ (+6) |
| `macromodel/agents/banks/func/interest_rates.py` | _policy_plus_spread, get_interest_rates_on_short_term_firm_loans, get_interest_rates_on_long_term_firm_loans, get_interest_rates_on_household_consumption_loans, get_interest_rate_on_mortgages (+6) |
| `macromodel/economy/func/house_price_index.py` | HPIForecasting, HPIForecastingConstant, HPIForecastingOLS, HPIImplementedForecastingAutoReg, HPIManualForecastingAutoReg (+5) |
| `macromodel/agents/firms/func/productivity_growth.py` | compute_tfp_growth, compute_tfp_growth, ProductivityGrowth, NoOpTFPGrowth, SimpleTFPGrowth (+4) |

## Entry Points

Start here when exploring this area:

- **`create_test_params`** (Function) — `tests/test_macromodel/unit/test_agents/test_firms/func/test_productivity_investment_planner.py:10`
- **`test_always_returns_zero_investment`** (Function) — `tests/test_macromodel/unit/test_agents/test_firms/func/test_productivity_investment_planner.py:24`
- **`test_ignores_all_parameters`** (Function) — `tests/test_macromodel/unit/test_agents/test_firms/func/test_productivity_investment_planner.py:49`
- **`test_basic_investment_calculation_with_cost_savings`** (Function) — `tests/test_macromodel/unit/test_agents/test_firms/func/test_productivity_investment_planner.py:75`
- **`test_higher_unit_costs_encourage_investment`** (Function) — `tests/test_macromodel/unit/test_agents/test_firms/func/test_productivity_investment_planner.py:108`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `ProductionSetter` | Class | `macromodel/agents/firms/func/production.py` | 5 |
| `PureLeontief` | Class | `macromodel/agents/firms/func/production.py` | 227 |
| `CriticalAndImportantLeontief` | Class | `macromodel/agents/firms/func/production.py` | 378 |
| `CriticalLeontief` | Class | `macromodel/agents/firms/func/production.py` | 498 |
| `Linear` | Class | `macromodel/agents/firms/func/production.py` | 618 |
| `UnconstrainedProduction` | Class | `macromodel/agents/firms/func/production.py` | 731 |
| `BundledLeontief` | Class | `macromodel/agents/firms/func/production.py` | 834 |
| `InflationForecasting` | Class | `macromodel/economy/func/inflation.py` | 40 |
| `InflationForecastingConstant` | Class | `macromodel/economy/func/inflation.py` | 95 |
| `InflationForecastingOLS` | Class | `macromodel/economy/func/inflation.py` | 117 |
| `InflationImplementedForecastingAutoReg` | Class | `macromodel/economy/func/inflation.py` | 138 |
| `InflationManualForecastingAutoReg` | Class | `macromodel/economy/func/inflation.py` | 160 |
| `ExogenousInflationForecasting` | Class | `macromodel/economy/func/inflation.py` | 186 |
| `GrowthForecasting` | Class | `macromodel/economy/func/growth.py` | 39 |
| `GrowthForecastingConstant` | Class | `macromodel/economy/func/growth.py` | 81 |
| `GrowthForecastingOLS` | Class | `macromodel/economy/func/growth.py` | 103 |
| `GrowthImplementedForecastingAutoReg` | Class | `macromodel/economy/func/growth.py` | 119 |
| `GrowthManualForecastingAutoReg` | Class | `macromodel/economy/func/growth.py` | 141 |
| `ExogenousGrowthForecasting` | Class | `macromodel/economy/func/growth.py` | 167 |
| `HPIForecasting` | Class | `macromodel/economy/func/house_price_index.py` | 36 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `N_industry_default → Create_household_bundle` | cross_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Test_firms | 5 calls |

## How to Explore

1. `gitnexus_context({name: "create_test_params"})` — see callers and callees
2. `gitnexus_query({query: "func"})` — find related execution flows
3. Read key files listed above for implementation details
