import numpy as np
import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.configuration.region import Region
from macro_data.readers.default_readers import DataReaders
from macro_data.readers.economic_data.exchange_rates import ExchangeRatesReader
from macro_data.readers.economic_data.oecd_economic_data import OECDEconData
from macro_data.readers.io_tables.icio_reader import ICIOReader
from macro_data.readers.socioeconomic_data.wiod_sea_data import WIODSEAReader
from macro_data.readers.util.capital_depreciation import load_eurostat_capital_depreciation_data


def compile_industry_data(
    year: int,
    readers: DataReaders,
    country_names: list[Country],
    single_firm_per_industry: dict[str, bool],
    yearly_factor: float = 4.0,
    capital_replacement_matrix_sources: dict[str, str] | None = None,
    capital_depreciation_accounting_modes: dict[str, str] | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    industry_data = {}
    current_icio_reader = readers.icio[year]
    sea_reader = readers.wiod_sea
    exchange_rates = readers.exchange_rates
    econ_reader = readers.oecd_econ
    if capital_replacement_matrix_sources is None:
        capital_replacement_matrix_sources = {}
    if capital_depreciation_accounting_modes is None:
        capital_depreciation_accounting_modes = {}

    for country_name in country_names:
        # Matrices

        intermediate_inputs_productivity_matrix = current_icio_reader.get_intermediate_inputs_matrix(country_name)
        capital_inputs_productivity_matrix = current_icio_reader.get_capital_inputs_matrix(
            country_name=country_name,
            capital_stock=sea_reader.get_values_in_usd(country_name, "Capital Stock"),
        )
        capital_depreciation_data = None
        replacement_source = capital_replacement_matrix_sources.get(str(country_name), "capital_compensation")
        depreciation_mode = capital_depreciation_accounting_modes.get(str(country_name), "none")
        if depreciation_mode == "eurostat_cfc" and replacement_source != "eurostat_cfc_output":
            raise ValueError(
                "capital_depreciation_accounting_mode='eurostat_cfc' requires "
                "capital_replacement_matrix_source='eurostat_cfc_output'."
            )
        if replacement_source == "eurostat_cfc_output" or depreciation_mode == "eurostat_cfc":
            capital_depreciation_data = load_eurostat_capital_depreciation_data(
                eurostat_path=readers.eurostat.path,
                country_name=country_name,
                year=year,
                industries=list(current_icio_reader.industries),
            )
            capital_depreciation_data["Annual Capital Depreciation Rate"] = capital_depreciation_data[
                "Capital Depreciation Rate"
            ]
            capital_depreciation_data["Capital Depreciation Rate"] = (
                capital_depreciation_data["Capital Depreciation Rate"] / yearly_factor
            )
            capital_depreciation_data["Annual CFC Output Ratio"] = capital_depreciation_data["CFC Output Ratio"]
            capital_depreciation_data["Capital Depreciation Output Ratio"] = capital_depreciation_data[
                "CFC Output Ratio"
            ]
        if replacement_source == "capital_compensation":
            capital_input_use_matrix = current_icio_reader.get_capital_inputs_depreciation(
                country_name=country_name,
                capital_compensation=sea_reader.get_values_in_usd(country_name, "Capital Compensation"),
            )
        elif replacement_source == "eurostat_cfc_output":
            capital_input_use_matrix = current_icio_reader.get_capital_inputs_depreciation_from_cfc_output(
                country_name=country_name,
                cfc_output_ratios=capital_depreciation_data["CFC Output Ratio"],
            )
        else:
            raise ValueError(
                "Unknown capital_replacement_matrix_source "
                f"{replacement_source!r}; expected 'capital_compensation' or 'eurostat_cfc_output'."
            )

        # Exchange rates
        exchange_rate = exchange_rates.from_usd_to_lcu(country_name, sea_reader.year)

        # Industry vectors
        industry_vectors = get_industry_vectors(
            country_name,
            current_icio_reader,
            exchange_rate,
            sea_reader,
            econ_reader,
            single_firm_per_industry[country_name],
            trade_partners=country_names + ["ROW"],
            yearly_factor=yearly_factor,
        )
        if capital_depreciation_data is not None:
            for column in capital_depreciation_data.columns:
                industry_vectors[column] = capital_depreciation_data[column].reindex(industry_vectors.index).values
        if depreciation_mode == "eurostat_cfc":
            labour_lcu = industry_vectors["Labour Output Ratio"].values * industry_vectors["Output in LCU"].values
            production_taxes_lcu = (
                industry_vectors["Production Tax Output Ratio"].values * industry_vectors["Output in LCU"].values
            )
            industry_vectors["Labour Compensation in LCU"] = labour_lcu
            industry_vectors["Labour Compensation in USD"] = labour_lcu / exchange_rate
            industry_vectors["Taxes Less Subsidies in LCU"] = production_taxes_lcu
            industry_vectors["Taxes Less Subsidies in USD"] = production_taxes_lcu / exchange_rate
            industry_vectors["Taxes Less Subsidies Rates"] = industry_vectors["Production Tax Output Ratio"]

        # Record all
        industry_data[country_name] = {
            "intermediate_inputs_productivity_matrix": intermediate_inputs_productivity_matrix,
            "capital_inputs_productivity_matrix": capital_inputs_productivity_matrix,
            "capital_input_use_matrix": capital_input_use_matrix,
            "capital_inputs_depreciation_matrix": capital_input_use_matrix,
            "industry_vectors": industry_vectors,
        }

    # Adding the rest of the world
    exchange_rate_row = exchange_rates.from_usd_to_lcu("ROW", sea_reader.year)
    industry_data["ROW"] = {
        "industry_vectors": {
            "Exports in USD": current_icio_reader.get_exports("ROW"),
            "Exports in LCU": exchange_rate_row * current_icio_reader.get_exports("ROW"),
            "Imports in USD": current_icio_reader.get_imports("ROW"),
            "Imports in LCU": exchange_rate_row * current_icio_reader.get_imports("ROW"),
        }
    }

    fill_trade_data(
        "ROW", current_icio_reader, exchange_rate_row, industry_data["ROW"]["industry_vectors"], country_names
    )

    return industry_data


def get_industry_vectors(
    country_name: Country,
    current_icio_reader: ICIOReader,
    exchange_rate: float,
    sea_reader: WIODSEAReader,
    econ_reader: OECDEconData,
    single_firm_per_industry: bool,
    trade_partners: list[str],
    yearly_factor: float = 4.0,
) -> pd.DataFrame:
    industry_vectors = pd.DataFrame(
        data={
            "Output in USD": current_icio_reader.get_total_output(country_name),
            "Output in LCU": current_icio_reader.get_total_output(country_name) * exchange_rate,
            "Intermediate Inputs Supply": current_icio_reader.get_intermediate_inputs_use(country_name).sum(axis=1),
            "Intermediate Inputs Use in USD": current_icio_reader.get_intermediate_inputs_use(country_name).sum(axis=0),
            "Intermediate Inputs Use in LCU": exchange_rate
            * current_icio_reader.get_intermediate_inputs_use(country_name).sum(axis=0),
            "Intermediate Inputs Domestic Use in USD": current_icio_reader.get_intermediate_inputs_domestic(
                country_name
            ).sum(axis=0),
            "Intermediate Inputs Domestic Use in LCU": exchange_rate
            * current_icio_reader.get_intermediate_inputs_domestic(country_name).sum(axis=0),
            "Firm Capital Inputs in USD": current_icio_reader.get_firm_capital_inputs(country_name),
            "Firm Capital Inputs in LCU": exchange_rate * current_icio_reader.get_firm_capital_inputs(country_name),
            "Household Capital Inputs in USD": current_icio_reader.get_household_capital_inputs(country_name),
            "Household Capital Inputs in LCU": exchange_rate
            * current_icio_reader.get_household_capital_inputs(country_name),
            "Household Investment Weights": current_icio_reader.get_household_capital_inputs(country_name)
            / current_icio_reader.get_household_capital_inputs(country_name).sum(),
            "Value Added in USD": current_icio_reader.get_value_added(country_name),
            "Value Added in LCU": exchange_rate * current_icio_reader.get_value_added(country_name),
            "Taxes Less Subsidies in USD": current_icio_reader.get_taxes_less_subsidies(country_name),
            "Taxes Less Subsidies in LCU": exchange_rate * current_icio_reader.get_taxes_less_subsidies(country_name),
            "Taxes Less Subsidies Rates": current_icio_reader.get_taxes_less_subsidies_rates(country_name),
            "Household Consumption in USD": current_icio_reader.get_hh_consumption(country_name),
            "Household Consumption in LCU": exchange_rate * current_icio_reader.get_hh_consumption(country_name),
            "Domestic Household Consumption in USD": current_icio_reader.get_hh_consumption_domestic(country_name),
            "Domestic Household Consumption in LCU": exchange_rate
            * current_icio_reader.get_hh_consumption_domestic(country_name),
            "Household Consumption Weights": current_icio_reader.get_hh_consumption_weights(country_name),
            "Government Consumption in USD": current_icio_reader.get_govt_consumption(country_name),
            "Government Consumption in LCU": exchange_rate * current_icio_reader.get_govt_consumption(country_name),
            "Government Consumption Weights": current_icio_reader.govt_consumption_weights(country_name),
            "Labour Compensation in USD": sea_reader.get_values_in_usd(country_name, "Labour Compensation")
            / yearly_factor,
            "Labour Compensation in LCU": exchange_rate
            * sea_reader.get_values_in_usd(
                country_name,
                "Labour Compensation",
            )
            / yearly_factor,
            "Capital Stock": sea_reader.get_values_in_usd(country_name, "Capital Stock"),
            "Exports in USD": current_icio_reader.get_exports(country_name),
            "Exports in LCU": exchange_rate * current_icio_reader.get_exports(country_name),
            "Imports in USD": current_icio_reader.get_imports(country_name),
            "Imports in LCU": exchange_rate * current_icio_reader.get_imports(country_name),
            "Average Initial Price": np.full(len(current_icio_reader.industries), exchange_rate),
        },
        index=pd.Index(current_icio_reader.industries, name="Industry"),
    )
    # Record the number of firms by sector
    if single_firm_per_industry:
        industry_vectors["Number of Firms"] = np.ones(len(current_icio_reader.industries), dtype=int)
    else:
        if isinstance(country_name, Region):
            data_country = country_name.parent_country
        else:
            data_country = country_name
        n_firms_by_industry = econ_reader.read_business_demography(
            country=data_country,
            output=pd.Series(industry_vectors["Output in USD"].values),
            year=sea_reader.year,
        )
        n_firms_by_industry[n_firms_by_industry == 0] = 1
        industry_vectors["Number of Firms"] = n_firms_by_industry

    fill_trade_data(country_name, current_icio_reader, exchange_rate, industry_vectors, trade_partners)

    return industry_vectors


def fill_trade_data(
    country_name: str,
    current_icio_reader: ICIOReader,
    exchange_rate: float,
    industry_vectors: pd.DataFrame,
    trade_partners: list[str],
):
    for c in trade_partners:
        if c == country_name:
            continue
        industry_vectors["Exports in USD to " + c] = current_icio_reader.get_trade(country_name, c)
        industry_vectors["Exports in LCU to " + c] = exchange_rate * current_icio_reader.get_trade(country_name, c)
        industry_vectors["Imports in USD from " + c] = current_icio_reader.get_trade(c, country_name)
        industry_vectors["Imports in LCU from " + c] = exchange_rate * current_icio_reader.get_trade(c, country_name)


def compile_exogenous_industry_data(
    readers: DataReaders, country_names: list[str], year_min: int = 2010, year_max: int = 2019
) -> dict[str, pd.DataFrame]:
    icio_readers = readers.icio
    exchange_rates = readers.exchange_rates
    sea_reader = readers.wiod_sea

    # Handle regular countries
    exogenous_industry_data = {
        country: pd.concat(
            [
                get_country_industry_data(country, country_names, exchange_rates, icio_readers, year, sea_reader)
                for year in range(year_min, year_max)
                if year in icio_readers.keys()
            ]
        )
        for country in country_names
    }

    # Adding the rest of the world
    exogenous_industry_data["ROW"] = pd.concat(
        [
            get_row_industry_data(country_names, exchange_rates, icio_readers, year)
            for year in range(year_min, year_max)
            if year in icio_readers.keys()
        ]
    )

    # Interpolate if we have enough data
    for country_name, country_data in exogenous_industry_data.items():
        if country_data.shape[0] > 1:
            country_data = country_data.astype(float).resample("M").interpolate("linear").copy()
            country_data.index = pd.DatetimeIndex([pd.Timestamp(d.year, d.month, 1) for d in country_data.index])
            exogenous_industry_data[country_name] = country_data

    return exogenous_industry_data


def get_row_industry_data(
    country_names: list[str], exchange_rates: ExchangeRatesReader, icio_readers: dict[int, ICIOReader], year: int
) -> pd.DataFrame:
    exchange_rate_row = exchange_rates.from_usd_to_lcu("ROW", year)
    row_industry_data = pd.DataFrame(
        data={
            "Exports in USD": icio_readers[year].get_exports("ROW"),
            "Exports in LCU": exchange_rate_row * icio_readers[year].get_exports("ROW"),
            "Imports in USD": icio_readers[year].get_imports("ROW"),
            "Imports in LCU": exchange_rate_row * icio_readers[year].get_imports("ROW"),
        },
        index=pd.Index(icio_readers[year].industries, name="Industry"),
    )
    for c in country_names:
        row_industry_data["Exports in USD to " + c] = icio_readers[year].get_trade("ROW", c)
        row_industry_data["Exports in LCU to " + c] = exchange_rate_row * icio_readers[year].get_trade("ROW", c)
        row_industry_data["Imports in USD from " + c] = icio_readers[year].get_trade(c, "ROW")
        row_industry_data["Imports in LCU from " + c] = exchange_rate_row * icio_readers[year].get_trade(c, "ROW")

    row_industry_data = row_industry_data.stack().to_frame().T.swaplevel(0, 1, axis=1)

    row_industry_data.index = [pd.to_datetime(year, format="%Y")]

    return row_industry_data


def get_country_industry_data(
    country_name: str,
    country_names: list[str],
    exchange_rates: ExchangeRatesReader,
    icio_readers: dict[int, ICIOReader],
    year: int,
    sea_reader: WIODSEAReader,
) -> pd.DataFrame:
    current_icio_reader = icio_readers[year]
    # Exchange rates
    exchange_rate = exchange_rates.from_usd_to_lcu(country_name, year)
    # Industry vectors
    industry_data = pd.DataFrame(
        data={
            "Output in USD": current_icio_reader.get_total_output(country_name),
            "Output in LCU": current_icio_reader.get_total_output(country_name) * exchange_rate,
            "Intermediate Inputs Supply": current_icio_reader.get_intermediate_inputs_use(country_name).sum(axis=0),
            "Intermediate Inputs Use in USD": current_icio_reader.get_intermediate_inputs_use(country_name).sum(axis=1),
            "Intermediate Inputs Use in LCU": exchange_rate
            * current_icio_reader.get_intermediate_inputs_use(country_name).sum(axis=1),
            "Intermediate Inputs Domestic Use in USD": current_icio_reader.get_intermediate_inputs_domestic(
                country_name
            ).sum(axis=1),
            "Intermediate Inputs Domestic Use in LCU": exchange_rate
            * current_icio_reader.get_intermediate_inputs_domestic(country_name).sum(axis=1),
            "Firm Capital Inputs in USD": current_icio_reader.get_firm_capital_inputs(country_name),
            "Firm Capital Inputs in LCU": exchange_rate * current_icio_reader.get_firm_capital_inputs(country_name),
            "Household Capital Inputs in USD": current_icio_reader.get_household_capital_inputs(country_name),
            "Household Capital Inputs in LCU": exchange_rate
            * current_icio_reader.get_household_capital_inputs(country_name),
            "Household Investment Weights": current_icio_reader.get_household_capital_inputs(country_name)
            / current_icio_reader.get_household_capital_inputs(country_name).sum(),
            "Value Added in USD": current_icio_reader.get_value_added(country_name),
            "Value Added in LCU": exchange_rate * current_icio_reader.get_value_added(country_name),
            "Taxes Less Subsidies in USD": current_icio_reader.get_taxes_less_subsidies(country_name),
            "Taxes Less Subsidies in LCU": exchange_rate * current_icio_reader.get_taxes_less_subsidies(country_name),
            "Taxes Less Subsidies Rates": current_icio_reader.get_taxes_less_subsidies_rates(country_name),
            "Household Consumption in USD": current_icio_reader.get_hh_consumption(country_name),
            "Household Consumption in LCU": exchange_rate * current_icio_reader.get_hh_consumption(country_name),
            "Domestic Household Consumption in USD": current_icio_reader.get_hh_consumption_domestic(country_name),
            "Domestic Household Consumption in LCU": exchange_rate
            * current_icio_reader.get_hh_consumption_domestic(country_name),
            "Household Consumption Weights": current_icio_reader.get_hh_consumption_weights(country_name),
            "Government Consumption in USD": current_icio_reader.get_govt_consumption(country_name),
            "Government Consumption in LCU": exchange_rate * current_icio_reader.get_govt_consumption(country_name),
            "Government Consumption Weights": current_icio_reader.govt_consumption_weights(country_name),
            "Labour Compensation in USD": sea_reader.get_values_in_usd(country_name, "Labour Compensation")
            / current_icio_reader.yearly_factor,
            "Labour Compensation in LCU": exchange_rate
            * sea_reader.get_values_in_usd(
                country_name,
                "Labour Compensation",
            )
            / current_icio_reader.yearly_factor,
            "Capital Stock": sea_reader.get_values_in_usd(country_name, "Capital Stock"),
            "Exports in USD": current_icio_reader.get_exports(country_name),
            "Exports in LCU": exchange_rate * current_icio_reader.get_exports(country_name),
            "Imports in USD": current_icio_reader.get_imports(country_name),
            "Imports in LCU": exchange_rate * current_icio_reader.get_imports(country_name),
            "Average Initial Price": np.full(len(current_icio_reader.industries), exchange_rate),
        },
        index=pd.Index(icio_readers[year].industries, name="Industry"),
    )
    for c in country_names + ["ROW"]:
        if c == country_name:
            continue
        industry_data["Exports in USD to " + c] = icio_readers[year].get_trade(country_name, c)
        industry_data["Exports in LCU to " + c] = exchange_rate * icio_readers[year].get_trade(country_name, c)
        industry_data["Imports in USD from " + c] = icio_readers[year].get_trade(c, country_name)
        industry_data["Imports in LCU from " + c] = exchange_rate * icio_readers[year].get_trade(c, country_name)

    # put everything in one row
    industry_data = industry_data.stack().to_frame().T.swaplevel(0, 1, axis=1)

    industry_data.index = [pd.to_datetime(year, format="%Y")]

    return industry_data
