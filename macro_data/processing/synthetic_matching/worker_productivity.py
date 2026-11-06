"""Fixed worker labour inputs, built from survey income at initialisation.

Each worker gets one fixed number, the labour input L. Hours are used here only:
they convert income to a full-time equivalent, and they scale the result back.

    rho_i = full-time equivalent income / industry mean, capped at the top
    L_i   = rho_i * hours_i / full-time hours

Workers are then spread over the firms of their industry so each firm's mean L
is close to 1, and L is rescaled to make it exactly 1. Firm labour therefore
equals headcount, and firm wage bills are unchanged. Pay is L times firm wage
per head, so relative survey income survives at runtime.

Unemployed workers have no income, so their rho comes from a regression on the
employed plus a drawn residual. Their hours are full time.
"""

import numpy as np
import pandas as pd

from macro_data.configuration.dataconfiguration import WorkerProductivityConfiguration


def effective_hours(hours: pd.Series, configuration: WorkerProductivityConfiguration) -> pd.Series:
    """Clip weekly hours to the allowed range. Missing or non-positive hours mean full time."""
    hours = pd.to_numeric(hours, errors="coerce")
    hours = hours.where(hours > 0.0, configuration.full_time_hours)
    return hours.clip(lower=configuration.min_hours, upper=configuration.full_time_hours)


def apply_wage_floor(
    individual_data: pd.DataFrame,
    configuration: WorkerProductivityConfiguration,
    scale: int,
    yearly_factor: float,
) -> pd.DataFrame:
    """Raise employed income to the hours-adjusted floor and store effective hours."""
    hours = effective_hours(individual_data["Weekly Hours"], configuration)
    individual_data["Effective Hours"] = hours.values
    floor = configuration.annual_wage_floor * scale / yearly_factor * hours / configuration.full_time_hours
    employed = individual_data["Activity Status"] == 1
    individual_data.loc[employed, "Employee Income"] = np.maximum(
        individual_data.loc[employed, "Employee Income"], floor[employed]
    )
    return individual_data


def balanced_firm_assignment(labour: np.ndarray, firm_ids: np.ndarray, firm_sizes: np.ndarray) -> np.ndarray:
    """Give each firm evenly spaced ranks of ``labour``. Returns the firm of each worker."""
    positions = np.concatenate([(np.arange(size) + 0.5) / size for size in firm_sizes])
    slots = np.repeat(firm_ids, firm_sizes)[np.argsort(positions, kind="stable")]
    assignment = np.empty(len(labour), dtype=int)
    assignment[np.argsort(-labour, kind="stable")] = slots
    return assignment


def _demographics(data: pd.DataFrame, mean_age: float, education_levels: list[int]) -> np.ndarray:
    age = (data["Age"].values.astype(float) - mean_age) / 10.0
    education = data["Education"].values
    gender = (data["Gender"].values == 1).astype(float)
    dummies = [(education == level).astype(float) for level in education_levels[1:]]
    return np.column_stack([np.ones(len(data)), age, age**2, gender, *dummies])


def predict_unemployed_labour(
    employed: pd.DataFrame, employed_rho: np.ndarray, unemployed: pd.DataFrame, cap: float
) -> np.ndarray:
    """Fit log rho on age, age squared, gender, education. Predict with a drawn residual."""
    mean_age = float(employed["Age"].astype(float).mean())
    education_levels = sorted(set(employed["Education"].dropna().astype(int)))
    outcome = np.log(employed_rho)
    regressors = _demographics(employed, mean_age, education_levels)
    coefficients = np.linalg.lstsq(regressors, outcome, rcond=None)[0]
    residuals = outcome - regressors @ coefficients
    draws = np.random.choice(residuals, len(unemployed), replace=True)
    predicted = _demographics(unemployed, mean_age, education_levels) @ coefficients + draws
    return np.minimum(np.exp(predicted), cap)


def match_workers_by_productivity(
    n_industries: int,
    labour_taxrate: float,
    firms,
    population,
    configuration: WorkerProductivityConfiguration,
) -> None:
    """Set labour inputs, firm assignment, pay and benefit weights for one country."""
    data = population.individual_data
    firm_data = firms.firm_data
    hours = data["Effective Hours"].values.astype(float)
    full_time = configuration.full_time_hours
    employed = (data["Activity Status"] == 1).values
    industry = data["Employment Industry"].values

    income = data["Employee Income"].values.astype(float)
    rho = np.zeros(len(data))
    for ind in range(n_industries):
        members = employed & (industry == ind)
        if members.any():
            rho[members] = income[members] / income[members].mean() * full_time / hours[members]
    cap = float(np.quantile(rho[employed], configuration.top_quantile))
    rho = np.minimum(rho, cap)

    labour = np.zeros(len(data))
    pay = np.zeros(len(data))
    assigned_firm = data["Corresponding Firm ID"].values.astype(float).copy()
    for ind in range(n_industries):
        members = np.flatnonzero(employed & (industry == ind))
        firm_ids = np.flatnonzero(firm_data["Industry"] == ind)
        sizes = firm_data.loc[firm_ids, "Number of Employees"].values.astype(int)
        if sizes.sum() != len(members):
            raise ValueError(f"Industry {ind}: {len(members)} workers for {sizes.sum()} firm positions.")
        if len(members) == 0:
            continue
        industry_labour = rho[members] * hours[members] / full_time
        industry_labour = industry_labour / industry_labour.mean()
        keep = sizes > 0
        assignment = balanced_firm_assignment(industry_labour, firm_ids[keep], sizes[keep])
        for firm_id, size in zip(firm_ids[keep], sizes[keep]):
            in_firm = assignment == firm_id
            workers = members[in_firm]
            labour[workers] = industry_labour[in_firm] / industry_labour[in_firm].mean()
            pay[workers] = labour[workers] * labour_taxrate * firm_data.at[firm_id, "Total Wages"] / size
            assigned_firm[workers] = firm_id
            firm_data.at[firm_id, "Employees ID"] = list(workers)

    unemployed = (data["Activity Status"] == 2).values
    if unemployed.any():
        employed_rho = labour[employed] * full_time / hours[employed]
        labour[unemployed] = predict_unemployed_labour(data.loc[employed], employed_rho, data.loc[unemployed], cap)

    benefit_weight = np.zeros(len(data))
    active = employed | unemployed
    if unemployed.any():
        benefit_weight[active] = labour[active] / labour[unemployed].mean()

    data["Labour Inputs"] = labour
    data["Corresponding Firm ID"] = assigned_firm
    data["Employee Income"] = pay
    data["Benefit Weight"] = benefit_weight
    data.loc[unemployed, "Income from Unemployment Benefits"] = (
        data.loc[unemployed, "Unemployment Benefit Entitlement"].values * benefit_weight[unemployed]
    )
    population.set_income()
