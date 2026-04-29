from typing import Literal

from pydantic import BaseModel, ConfigDict

ConsumerPriceIndexSource = Literal["transaction_cpi", "fixed_basket_cpi", "chained_basket_cpi"]


class Growth(BaseModel):
    """
    The function used for setting how growth is centrally forecasted.
    """

    name: Literal[
        "GrowthForecastingConstant",
        "GrowthForecastingOLS",
        "GrowthManualForecastingAutoReg",
        "GrowthImplementedForecastingAutoReg",
        "ExogenousGrowthForecasting",
    ] = "GrowthManualForecastingAutoReg"
    parameters: dict = {"value": 0.0, "lags": 1}
    path_name: str = "growth"


class HPI(BaseModel):
    """
    The function used for setting how the house price index is centrally forecasted.
    """

    name: Literal[
        "HPIForecastingConstant",
        "HPIManualForecastingAutoReg",
        "HPIImplementedForecastingAutoReg",
        "HPIManualForecastingAutoReg",
    ] = "HPIManualForecastingAutoReg"
    parameters: dict = {"value": 0.0, "lags": 1}
    path_name: str = "house_price_index"


class InflationForecaster(BaseModel):
    """
    The function used for setting how inflation is centrally forecasted.
    """

    name: Literal[
        "InflationForecastingConstant",
        "InflationImplementedForecastingAutoReg",
        "InflationManualForecastingAutoReg",
        "ExogenousInflationForecasting",
    ] = "InflationManualForecastingAutoReg"
    parameters: dict = {"value": 0.0, "lags": 1}
    path_name: str = "inflation"


class ConsumerPriceIndex(BaseModel):
    """
    The consumer-facing CPI concept used for price levels and inflation rates.
    """

    model_config = ConfigDict(extra="forbid")

    source: ConsumerPriceIndexSource = "fixed_basket_cpi"


class Sentiment(BaseModel):
    """
    The function used for setting how sector sentiment is centrally forecasted.
    """

    name: Literal["ConstantSentimentSetter"] = "ConstantSentimentSetter"
    parameters: dict = {"value": 0.0}
    path_name: str = "sentiment"


class EconomyFunctions(BaseModel):
    """
    The functions used for the economy.
    """

    model_config = ConfigDict(extra="forbid")

    growth: Growth = Growth()
    house_price_index: HPI = HPI()
    inflation_forecaster: InflationForecaster = InflationForecaster()
    # sentiment: Sentiment = Sentiment()


class EconomyConfiguration(BaseModel):
    """
    The configuration settings for the economy.
    """

    model_config = ConfigDict(extra="forbid")

    functions: EconomyFunctions = EconomyFunctions()
    consumer_price_index: ConsumerPriceIndex = ConsumerPriceIndex()
