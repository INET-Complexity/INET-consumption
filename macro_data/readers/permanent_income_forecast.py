"""Common permanent-income forecast inputs for Stage 3."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
    residual_variance: float
    diagnostics: dict[str, float | int | str]


@dataclass(frozen=True)
class PermanentIncomeForecast:
    """Point forecast and forecast-error variance for common permanent income."""

    point_forecast: float
    forecast_variance: float


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _validate_permanent_income_forecast_inputs(
    table: pd.DataFrame,
    covariance: pd.DataFrame,
    residual_variance: float,
    *,
    tolerance: float = 1e-10,
) -> None:
    """Validate normalized common-forecast inputs before runtime use."""
    if "coefficient" not in table.columns:
        raise ValueError("Permanent-income coefficient table must include a 'coefficient' column.")

    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Permanent-income HAC covariance matrix must be square.")

    if not table.index.equals(covariance.index) or not table.index.equals(covariance.columns):
        raise ValueError("Permanent-income coefficient-table labels/order must exactly match HAC covariance axes.")

    coefficients = table["coefficient"].to_numpy(dtype=float)
    covariance_values = covariance.to_numpy(dtype=float)
    residual_variance = float(residual_variance)

    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Permanent-income coefficients must be finite.")

    if not np.all(np.isfinite(covariance_values)):
        raise ValueError("Permanent-income HAC covariance values must be finite.")

    if not np.isfinite(residual_variance):
        raise ValueError("Permanent-income residual variance must be finite.")

    if residual_variance <= 0:
        raise ValueError("Permanent-income residual variance must be positive.")

    if not np.allclose(covariance_values, covariance_values.T, atol=tolerance, rtol=0.0):
        raise ValueError("Permanent-income HAC covariance matrix must be symmetric.")

    min_eigenvalue = float(np.linalg.eigvalsh(covariance_values).min())
    if min_eigenvalue < -tolerance:
        raise ValueError("Permanent-income HAC covariance matrix must be positive semidefinite.")


def forecast_common_permanent_income(
    x_t: pd.Series,
    forecast_inputs: PermanentIncomeForecastInputs,
) -> PermanentIncomeForecast:
    """Compute the common permanent-income point forecast and forecast variance."""
    if not isinstance(x_t, pd.Series):
        raise TypeError("x_t must be a labeled pandas Series.")

    expected_index = forecast_inputs.coefficient_table.index
    if not x_t.index.equals(expected_index):
        raise ValueError("x_t index must exactly match the permanent-income coefficient-table order.")

    hac_index = forecast_inputs.hac_covariance.index
    hac_columns = forecast_inputs.hac_covariance.columns
    if not hac_index.equals(expected_index) or not hac_columns.equals(expected_index):
        raise ValueError("Permanent-income HAC covariance axes must exactly match the coefficient-table order.")

    x_values = x_t.to_numpy(dtype=float)
    if not np.all(np.isfinite(x_values)):
        raise ValueError("x_t values must be finite.")

    coefficients = forecast_inputs.coefficient_table["coefficient"].to_numpy(dtype=float)
    covariance = forecast_inputs.hac_covariance.to_numpy(dtype=float)

    point_forecast = float(x_values @ coefficients)
    forecast_variance = float(forecast_inputs.residual_variance + x_values @ covariance @ x_values)
    return PermanentIncomeForecast(
        point_forecast=point_forecast,
        forecast_variance=forecast_variance,
    )


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


def load_permanent_income_forecast_residual_variance(path: str | Path) -> float:
    """Load the scalar residual variance s_u^2 from the regression export."""
    return float(_load_json(path))


def load_permanent_income_forecast_inputs(
    table_path: str | Path,
    cov_path: str | Path,
    residual_variance_path: str | Path,
) -> PermanentIncomeForecastInputs:
    """Load and normalize the common permanent-income forecast inputs."""
    table = load_permanent_income_forecast_table(table_path)
    covariance = load_permanent_income_forecast_hac_covariance(cov_path)
    # JSON round-tripping of the exported HAC covariance introduces ~1e-10 asymmetry,
    # which is symmetric by construction. Symmetrize only when already nearly symmetric
    # (within a loose tolerance) so genuinely asymmetric/non-square inputs still fail
    # validation with their normal error messages.
    if covariance.shape[0] == covariance.shape[1]:
        covariance_values = covariance.to_numpy(dtype=float)
        if np.all(np.isfinite(covariance_values)) and np.allclose(
            covariance_values, covariance_values.T, atol=1e-6, rtol=0.0
        ):
            covariance = (covariance + covariance.T) / 2.0
    residual_variance = load_permanent_income_forecast_residual_variance(residual_variance_path)
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
    _validate_permanent_income_forecast_inputs(table, covariance, residual_variance)
    return PermanentIncomeForecastInputs(
        coefficient_table=table,
        hac_covariance=covariance,
        residual_variance=residual_variance,
        diagnostics=diagnostics,
    )
