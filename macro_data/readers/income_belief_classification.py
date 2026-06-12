"""Per-member classification and household aggregation for Stage 3 income beliefs."""

import numpy as np
import pandas as pd

from macromodel.agents.individuals.individual_properties import ActivityStatus
from macro_data.readers.income_belief_priors import lookup_income_belief_priors

AGE_GROUPS = ["18-34", "35-49", "50-70", "71-85"]
AGE_GROUP_BOUNDS = [18, 35, 50, 71, 86]  # right-open bins; ages clipped to [18, 85]

_EMPLOYMENT_STATUS_MAP = {
    ActivityStatus.EMPLOYED: "employed",
    ActivityStatus.UNEMPLOYED: "unemployed",
    ActivityStatus.NOT_ECONOMICALLY_ACTIVE: "retired",
    ActivityStatus.FIRM_INVESTOR: "retired",
    ActivityStatus.BANK_INVESTOR: "retired",
}

BELIEF_FIELDS = ["mu_1_0", "P_1_0", "sigma2_xi", "sigma2_v", "rho"]
BELIEF_OUTPUT_NAMES = {
    "mu_1_0": "income_belief_mu",
    "P_1_0": "income_belief_p",
    "rho": "income_belief_rho",
    "sigma2_xi": "sigma2_xi",
    "sigma2_v": "sigma2_v",
}


def classify_age_group(age: np.ndarray) -> np.ndarray:
    """Bin ages into the priors table's four age groups, clipping to [18, 85]."""
    clipped = np.clip(np.asarray(age, dtype=float), 18, 85)
    bin_indices = np.digitize(clipped, AGE_GROUP_BOUNDS[1:-1], right=False)
    return np.asarray(AGE_GROUPS)[bin_indices]


def classify_employment_status(activity_status: np.ndarray) -> np.ndarray:
    """Map ``ActivityStatus`` values to 'employed'/'unemployed'/'retired'."""
    return np.array([_EMPLOYMENT_STATUS_MAP[ActivityStatus(value)] for value in np.asarray(activity_status)])


def compute_household_income_beliefs(
    individuals_age: np.ndarray,
    individuals_activity_status: np.ndarray,
    individuals_household_id: np.ndarray,
    individuals_income: np.ndarray,
    n_households: int,
    priors_table: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Classify each individual, look up their priors row, and income-weight-aggregate to households.

    Returns a dict with household-level arrays (shape ``(n_households,)``) for
    ``income_belief_mu``, ``income_belief_p``, ``income_belief_rho``, ``sigma2_xi``, ``sigma2_v``.
    Households with zero total member income fall back to equal weights across members.
    """
    employment_status = classify_employment_status(individuals_activity_status)
    age_group = classify_age_group(individuals_age)
    household_id = np.asarray(individuals_household_id, dtype=int)
    income = np.asarray(individuals_income, dtype=float)

    member_values = {field: np.empty(len(household_id), dtype=float) for field in BELIEF_FIELDS}
    for status, group in priors_table.index:
        mask = (employment_status == status) & (age_group == group)
        if not mask.any():
            continue
        priors = lookup_income_belief_priors(status, group, priors_table)
        for field in BELIEF_FIELDS:
            member_values[field][mask] = priors[field]

    household_total_income = np.bincount(household_id, weights=income, minlength=n_households)
    member_count = np.bincount(household_id, minlength=n_households)
    total_income_per_member = household_total_income[household_id]
    has_income = total_income_per_member > 0
    weights = np.where(
        has_income,
        income / np.where(has_income, total_income_per_member, 1.0),
        1.0 / member_count[household_id],
    )

    result = {}
    for field in BELIEF_FIELDS:
        weighted_sum = np.bincount(household_id, weights=weights * member_values[field], minlength=n_households)
        result[BELIEF_OUTPUT_NAMES[field]] = weighted_sum
    return result
