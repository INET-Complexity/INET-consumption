"""Income-belief calibration priors (Stage 3 learning block)."""

from pathlib import Path

import pandas as pd

DEFAULT_INCOME_BELIEF_PRIORS_CSV = Path(__file__).resolve().parent / "data" / "CES" / "FR_priors_table5.csv"

_FALLBACK_COLUMNS = ["mean_income_uncertainty_var_prior", "sigma2_xi", "sigma2_v"]


def load_income_belief_priors_table(path: str | Path = DEFAULT_INCOME_BELIEF_PRIORS_CSV) -> pd.DataFrame:
    """Load the employment_status x age_group income-belief priors table.

    The ``unemployed``/``71-85`` row is missing ``mean_income_uncertainty_var_prior``,
    ``sigma2_xi`` and ``sigma2_v``; these are filled from the ``unemployed``/``50-70`` row.
    """
    table = pd.read_csv(path)
    fallback_row = table.loc[(table["employment_status"] == "unemployed") & (table["age_group"] == "50-70")].iloc[0]
    target_index = table.index[(table["employment_status"] == "unemployed") & (table["age_group"] == "71-85")]
    table.loc[target_index, _FALLBACK_COLUMNS] = fallback_row[_FALLBACK_COLUMNS].to_numpy()
    return table.set_index(["employment_status", "age_group"])


def lookup_income_belief_priors(employment_status: str, age_group: str, table: pd.DataFrame) -> dict:
    """Return ``{rho, mu_1_0, P_1_0, sigma2_xi, sigma2_v}`` for one (employment_status, age_group)."""
    row = table.loc[(employment_status, age_group)]
    return {
        "rho": float(row["rho"]),
        "mu_1_0": float(row["mean_income_exp_dev"]),
        "P_1_0": float(row["mean_income_uncertainty_var_prior"]),
        "sigma2_xi": float(row["sigma2_xi"]),
        "sigma2_v": float(row["sigma2_v"]),
    }
