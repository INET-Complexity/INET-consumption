"""Common permanent-income forecast inputs for Stage 3."""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAW_VARIABLE_ORDER = [
    "const",
    "trend",
    "splittrend_pdv",
    "covid19",
    "log_y",
    "d4_log_y",
    "log_work_age_pop_share",
    "cci_fs_ma4_l1",
    "rr_ma4_l1",
    "rr_ma4_l5",
    "rr_ma4_l9",
    "t_bill_d4",
    "t_bill_d4_l1",
    "stock_log_ma4_l1",
    "u_ma4_l1",
    "u_ma4_l5",
    "oil_log_ma4_l1",
    "oil_log_ma4_l5",
    "fx_log_ma4_l1",
    "fx_log_ma4_l5",
]

SIMULATION_VARIABLE_NAME_MAP = {
    "const": "constant",
    "trend": "time_trend",
    "splittrend_pdv": "split_trend_from_2009q4_discounted_present_value",
    "covid19": "covid19",
    "log_y": "log_real_pc_income",
    "d4_log_y": "d4_log_real_pc_income",
    "log_work_age_pop_share": "log_working_age_population_share",
    "cci_fs_ma4_l1": "survey_expectations_ma4_l1",
    "rr_ma4_l1": "real_interest_rate_ma4_l1",
    "rr_ma4_l5": "real_interest_rate_ma4_l5",
    "rr_ma4_l9": "real_interest_rate_ma4_l9",
    "t_bill_d4": "d4_t_bill_rate",
    "t_bill_d4_l1": "d4_t_bill_rate_l1",
    "stock_log_ma4_l1": "log_stock_market_index_ma4_l1",
    "u_ma4_l1": "unemp_rate_ma4_l1",
    "u_ma4_l5": "unemp_rate_ma4_l5",
    "oil_log_ma4_l1": "log_real_oil_price_ma4_l1",
    "oil_log_ma4_l5": "log_real_oil_price_ma4_l5",
    "fx_log_ma4_l1": "log_real_fx_rate_ma4_l1",
    "fx_log_ma4_l5": "log_real_fx_rate_ma4_l5",
}


@dataclass(frozen=True)
class PermanentIncomeForecastInputs:
    """Normalized common-forecast inputs mapped to simulation variable names."""

    coefficient_table: pd.DataFrame
    hac_covariance: pd.DataFrame
    diagnostics: dict[str, float | int]


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def load_permanent_income_forecast_table(path: str | Path) -> pd.DataFrame:
    """Load the permanent-income forecast coefficient table.

    The returned table is indexed by simulation variable names, not the raw
    estimation labels. Columns keep the regression summary fields from the export.
    """
    payload = _load_json(path)

    rows = []
    for raw_name in RAW_VARIABLE_ORDER:
        rows.append(
            {
                "raw_name": raw_name,
                "simulation_name": SIMULATION_VARIABLE_NAME_MAP[raw_name],
                "coefficient": float(payload["coef"][raw_name]),
                "std_err_hac": float(payload["std_err_hac"][raw_name]),
                "t_hac": float(payload["t_hac"][raw_name]),
                "p_hac": float(payload["p_hac"][raw_name]),
                "ci_low": float(payload["ci_low"][raw_name]),
                "ci_high": float(payload["ci_high"][raw_name]),
            }
        )
    table = pd.DataFrame(rows).set_index("simulation_name")
    return table.loc[[SIMULATION_VARIABLE_NAME_MAP[name] for name in RAW_VARIABLE_ORDER]]


def load_permanent_income_forecast_hac_covariance(path: str | Path) -> pd.DataFrame:
    """Load the HAC covariance matrix and map its axes to simulation variable names."""
    raw = pd.DataFrame(_load_json(path), dtype=float)
    raw = raw.reindex(index=RAW_VARIABLE_ORDER, columns=RAW_VARIABLE_ORDER)
    raw.index = [SIMULATION_VARIABLE_NAME_MAP[name] for name in raw.index]
    raw.columns = [SIMULATION_VARIABLE_NAME_MAP[name] for name in raw.columns]
    return raw


def load_permanent_income_forecast_inputs(
    table_path: str | Path,
    cov_path: str | Path,
) -> PermanentIncomeForecastInputs:
    """Load and normalize the common permanent-income forecast inputs."""
    table = load_permanent_income_forecast_table(table_path)
    covariance = load_permanent_income_forecast_hac_covariance(cov_path)
    diagnostics = {
        "table_label": "tab:permanent_income_estimation",
        "nobs": 94,
        "r2": 0.9966,
        "r2_adj": 0.9957,
        "aic": -917.9299,
        "bic": -867.064,
        "dw": 0.8067,
        "hac_lags": 8,
    }
    return PermanentIncomeForecastInputs(
        coefficient_table=table,
        hac_covariance=covariance,
        diagnostics=diagnostics,
    )
