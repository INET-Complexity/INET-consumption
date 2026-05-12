from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import statsmodels.api as sm

from macro_data.readers.default_readers import DataReaders
from macromodel.configurations import CountryConfiguration


def _default_substitution_bundles(n_industries: int) -> list[int]:
    return list(range(n_industries))


def align_country_configuration_to_data(
    country_cfg: CountryConfiguration,
    n_industries: int,
    n_firms: int | None = None,
) -> CountryConfiguration:
    """
    Align data-dimensioned parts of a CountryConfiguration to the data shape.

    The goal is to preserve YAML-loaded behavioral settings and only repair
    fields whose lengths must match the model dimensions implied by the data.
    """
    cfg = deepcopy(country_cfg)
    planner_n_firms = n_industries if n_firms is None else n_firms

    if len(cfg.firms.parameters.capital_inputs_delay) != n_industries:
        cfg.firms.parameters.capital_inputs_delay = [0] * n_industries

    if len(cfg.firms.parameters.depreciation_rates) != n_industries:
        cfg.firms.parameters.depreciation_rates = [0.0] * n_industries

    if len(cfg.firms.substitution_bundles) != n_industries:
        cfg.firms.substitution_bundles = _default_substitution_bundles(n_industries)

    if len(cfg.households.substitution_bundles) != n_industries:
        cfg.households.substitution_bundles = _default_substitution_bundles(n_industries)

    planner_params = cfg.firms.functions.productivity_investment_planner.parameters
    if planner_params.get("n_firms") != planner_n_firms:
        planner_params["n_firms"] = planner_n_firms

    if cfg.firms.functions.productivity_investment_planner.name == "TargetIntensityTFPInvestmentPlanner":
        growth_phi = cfg.firms.functions.productivity_growth.parameters.get("investment_effectiveness")
        if growth_phi is None:
            raise ValueError(
                "TargetIntensityTFPInvestmentPlanner requires "
                "productivity_growth.parameters['investment_effectiveness']; "
                "realised TFP effectiveness is the canonical phi used by the planner."
            )
        planner_params["investment_effectiveness"] = growth_phi

    return cfg


def unpack_cell(x):
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return x.item()
        return x.tolist()
    if isinstance(x, list) and len(x) == 1:
        return x[0]
    return x


def _prepare_social_benefits_arx_regression_data(
    benefits_inflation_data: pd.DataFrame,
    gdp_growth: pd.Series,
    target_column: str,
    regression_window: int = 48,
    log_floor: float = 1e-8,
) -> pd.DataFrame:
    data = benefits_inflation_data[[target_column, "Data CPI Inflation", "Unemployment Rate"]].copy()
    data["Log Benefits"] = np.log(data[target_column].clip(lower=log_floor))
    data["Delta Log Benefits"] = data["Log Benefits"].diff()
    data["Lagged Inflation"] = data["Data CPI Inflation"].shift(1)
    data["Unemployment Change"] = data["Unemployment Rate"].diff()
    gdp_growth_frame = pd.DataFrame({"GDP Growth": gdp_growth}).sort_index()
    data = pd.merge_asof(data.sort_index(), gdp_growth_frame, left_index=True, right_index=True)
    return data.iloc[-regression_window:].dropna()


def fit_social_benefits_arx_diagnostics(
    readers: DataReaders,
    country_name,
    year: int,
    year_range: int = 10,
    regression_window: int = 48,
    log_floor: float = 1e-8,
) -> dict[str, dict[str, object]]:
    """Fit statsmodels OLS diagnostics for the social-benefits ARX specification.

    Returns a dictionary keyed by target name. Each target contains:
    - ``results``: fitted statsmodels OLS results object
    - ``data``: regression dataframe used for estimation

    Example
    -------
    diagnostics = fit_social_benefits_arx_diagnostics(...)
    diagnostics["unemployment_benefits"]["results"].summary()
    diagnostics["other_benefits"]["data"].head()
    """
    country_exogenous_data = readers.get_exogenous_data(country_name)
    if country_exogenous_data is None:
        return {}

    benefits_inflation_data = readers.get_benefits_inflation_data(
        country_name=country_name,
        year_min=year - year_range,
        year_max=year,
        exogenous_data=country_exogenous_data,
    )
    national_accounts_growth = readers.get_national_accounts_growth(country_name)
    gdp_growth = (
        national_accounts_growth["GDP"] if "GDP" in national_accounts_growth.columns else pd.Series(dtype=float)
    )

    diagnostics: dict[str, dict[str, object]] = {}

    unemployment_regression_data = _prepare_social_benefits_arx_regression_data(
        benefits_inflation_data=benefits_inflation_data,
        gdp_growth=gdp_growth,
        target_column="Unemployment Benefits",
        regression_window=regression_window,
        log_floor=log_floor,
    )
    if not unemployment_regression_data.empty:
        unemployment_exog = sm.add_constant(
            unemployment_regression_data[["Lagged Inflation", "Unemployment Change", "GDP Growth"]],
            has_constant="add",
        )
        unemployment_endog = unemployment_regression_data["Delta Log Benefits"].rename(
            "Delta Log Unemployment Benefits"
        )
        unemployment_results = sm.OLS(unemployment_endog, unemployment_exog).fit()
        diagnostics["unemployment_benefits"] = {
            "results": unemployment_results,
            "data": unemployment_regression_data,
        }

    other_benefits_regression_data = _prepare_social_benefits_arx_regression_data(
        benefits_inflation_data=benefits_inflation_data,
        gdp_growth=gdp_growth,
        target_column="Other Total Benefits",
        regression_window=regression_window,
        log_floor=log_floor,
    )
    if not other_benefits_regression_data.empty:
        other_benefits_exog = sm.add_constant(
            other_benefits_regression_data[["Lagged Inflation", "Unemployment Change", "GDP Growth"]],
            has_constant="add",
        )
        other_benefits_endog = other_benefits_regression_data["Delta Log Benefits"].rename("Delta Log Other Benefits")
        other_benefits_results = sm.OLS(other_benefits_endog, other_benefits_exog).fit()
        diagnostics["other_benefits"] = {
            "results": other_benefits_results,
            "data": other_benefits_regression_data,
        }

    return diagnostics


def build_social_benefits_reader_df(
    readers: DataReaders,
    country_name,
    year: int,
    year_range: int = 10,
) -> pd.DataFrame:
    """Build a datetime-indexed dataframe of reader-side social-benefit drivers.

    The dataframe is aligned to the benefits time series frequency and includes:
    - unemployment benefits
    - other total benefits
    - inflation measures
    - GDP level and GDP growth
    - unemployment level and change
    - vacancy rate
    - house price index growth
    - firm deposits and debt
    """
    country_exogenous_data = readers.get_exogenous_data(country_name)
    if country_exogenous_data is None:
        return pd.DataFrame()

    benefits_data = readers.get_benefits_inflation_data(
        country_name=country_name,
        year_min=year - year_range,
        year_max=year,
        exogenous_data=country_exogenous_data,
    ).sort_index()

    df = benefits_data.copy()
    df.index = pd.to_datetime(df.index)

    inflation = country_exogenous_data["log_inflation"].copy().sort_index()
    inflation.index = pd.to_datetime(inflation.index)
    inflation_columns = [col for col in inflation.columns if col not in df.columns]
    if inflation_columns:
        df = pd.merge_asof(df, inflation[inflation_columns], left_index=True, right_index=True)
    national_accounts_growth = readers.get_national_accounts_growth(country_name)

    gdp = pd.Series(
        {ts: readers.eurostat.get_quarterly_gdp(country_name, ts.year, ts.quarter) for ts in df.index},
        name="GDP",
    )
    df["GDP"] = gdp
    df["GDP Growth"] = df["GDP"].pct_change()
    df["Unemployment Change"] = df["Unemployment Rate"].diff()
    if "Compensation of Employees" in national_accounts_growth.columns:
        wage_growth = national_accounts_growth["Compensation of Employees"].sort_index()
        wage_growth.index = pd.to_datetime(wage_growth.index)
        wage_frame = pd.DataFrame({"Wage Growth": wage_growth})
        wage_frame["Wages"] = (1.0 + wage_frame["Wage Growth"]).cumprod()
        df = pd.merge_asof(df, wage_frame[["Wages", "Wage Growth"]], left_index=True, right_index=True)

    vacancy_rate = country_exogenous_data["vacancy_rate"].copy().sort_index()
    vacancy_rate.index = pd.to_datetime(vacancy_rate.index)
    df = pd.merge_asof(df, vacancy_rate, left_index=True, right_index=True)

    house_price_index = country_exogenous_data["house_price_index"].copy().sort_index()
    house_price_index.index = pd.to_datetime(house_price_index.index)
    df = pd.merge_asof(df, house_price_index, left_index=True, right_index=True)

    firm_balance_sheet = country_exogenous_data["total_firm_deposits_and_debt"].copy().sort_index()
    firm_balance_sheet.index = pd.to_datetime(firm_balance_sheet.index)
    df = pd.merge_asof(df, firm_balance_sheet, left_index=True, right_index=True)

    labour_stats = readers.imf_reader.get_labour_stats(country_name)
    if labour_stats is not None:
        labour_stats = labour_stats.copy().sort_index()
        labour_stats.index = pd.to_datetime(labour_stats.index)
        extra_labour_columns = [col for col in labour_stats.columns if col not in df.columns]
        if extra_labour_columns:
            df = pd.merge_asof(df, labour_stats[extra_labour_columns], left_index=True, right_index=True)

    participation_rate = readers.world_bank.get_participation_rate(country_name).copy().sort_index()
    participation_rate.index = pd.to_datetime(participation_rate.index)
    df = pd.merge_asof(
        df,
        participation_rate.rename(columns={"Participation Rate": "Participation Rate"}),
        left_index=True,
        right_index=True,
    )
    if "Participation Rate" in df.columns:
        df["Participation Rate Change"] = df["Participation Rate"].diff()

    annual_population = pd.Series(
        {ts: readers.world_bank.get_population(country_name, ts.year) for ts in df.index},
        name="Population",
    )
    df["Population"] = annual_population
    df["Population Growth"] = df["Population"].pct_change()

    return df
