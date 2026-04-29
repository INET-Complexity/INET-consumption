# Economy API Reference

This section documents the Economy entity in the macromodel package, which manages and tracks aggregate economic metrics, market-level interactions, and macroeconomic indicators for each country.

::: macromodel.economy.economy.Economy
    options:
        members:
            - Economy
            - from_agents
            - reset
            - set_estimates
            - compute_inflation
            - record_global_trade
            - compute_rental_market_aggregates
            - compute_gdp
            - total_exports
            - total_cpi_inflation
            - total_ppi_inflation
            - total_cfpi_inflation
            - unemployment_rate
            - gdp_expenditure
            - gdp_output
        show_root_heading: true
        show_signature_annotations: true
        show_docstring: true
        show_source: false
        show_bases: false
        show_inheritance_diagram: false
        show_if_no_docstring: true
        heading_level: 4
        show_module_name: false
        hide_name: false

## Functions

## Consumer Price Index Configuration

`EconomyConfiguration` separates the consumer-facing CPI concept from the
inflation forecasting rule:

```yaml
economy:
  consumer_price_index:
    source: fixed_basket_cpi
  functions:
    inflation_forecaster:
      name: InflationManualForecastingAutoReg
      path_name: inflation
      parameters:
        lags: 1
        value: 0.0
```

The CPI source can be `transaction_cpi`, `fixed_basket_cpi`, or
`chained_basket_cpi`. The selected source is used consistently for consumer
price levels, period inflation, and annual inflation. The default is
`fixed_basket_cpi`.

### Sentiment

::: macromodel.economy.func.sentiment.SentimentSetter
    options:
        members:
            - SentimentSetter
        show_root_heading: true
        show_signature_annotations: true
        show_docstring: true
        show_source: false
        show_bases: false
        show_inheritance_diagram: false
        show_if_no_docstring: true
        heading_level: 4
        show_module_name: false
        hide_name: false

### Inflation Forecasting

::: macromodel.economy.func.inflation.InflationForecasting
    options:
        members:
            - InflationForecasting
        show_root_heading: true
        show_signature_annotations: true
        show_docstring: true
        show_source: false
        show_bases: false
        show_inheritance_diagram: false
        show_if_no_docstring: true
        heading_level: 4
        show_module_name: false
        hide_name: false

### Growth Forecasting

::: macromodel.economy.func.growth.GrowthForecasting
    options:
        members:
            - GrowthForecasting
        show_root_heading: true
        show_signature_annotations: true
        show_docstring: true
        show_source: false
        show_bases: false
        show_inheritance_diagram: false
        show_if_no_docstring: true
        heading_level: 4
        show_module_name: false
        hide_name: false
