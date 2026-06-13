import json

import numpy as np
import pytest

from macro_data.readers.permanent_income_forecast import (
    RAW_VARIABLE_ORDER,
    load_permanent_income_forecast_hac_covariance,
    load_permanent_income_forecast_inputs,
    load_permanent_income_forecast_table,
)


def _write_forecast_fixtures(tmp_path):
    table_path = tmp_path / "FR_table.json"
    cov_path = tmp_path / "FR_cov_hac.json"

    table_payload = {
        "coef": {},
        "std_err_hac": {},
        "t_hac": {},
        "p_hac": {},
        "ci_low": {},
        "ci_high": {},
    }
    for idx, raw_name in enumerate(RAW_VARIABLE_ORDER):
        table_payload["coef"][raw_name] = idx + 0.25
        table_payload["std_err_hac"][raw_name] = idx + 0.5
        table_payload["t_hac"][raw_name] = idx + 0.75
        table_payload["p_hac"][raw_name] = idx + 0.01
        table_payload["ci_low"][raw_name] = idx - 0.5
        table_payload["ci_high"][raw_name] = idx + 0.5

    covariance_payload = {}
    for col_idx, col_name in enumerate(RAW_VARIABLE_ORDER):
        covariance_payload[col_name] = {}
        for row_idx, row_name in enumerate(RAW_VARIABLE_ORDER):
            covariance_payload[col_name][row_name] = (
                1.0 + col_idx if row_idx == col_idx else (row_idx + col_idx) / 100
            )

    table_path.write_text(json.dumps(table_payload))
    cov_path.write_text(json.dumps(covariance_payload))
    return table_path, cov_path


def test__load_permanent_income_forecast_table_maps_to_simulation_names(tmp_path):
    table_path, _ = _write_forecast_fixtures(tmp_path)
    table = load_permanent_income_forecast_table(table_path)
    assert list(table.index) == [
        "constant",
        "time_trend",
        "split_trend_from_2009q4_discounted_present_value",
        "covid19",
        "log_real_pc_income",
        "d4_log_real_pc_income",
        "log_working_age_population_share",
        "survey_expectations_ma4_l1",
        "real_interest_rate_ma4_l1",
        "real_interest_rate_ma4_l5",
        "real_interest_rate_ma4_l9",
        "d4_t_bill_rate",
        "d4_t_bill_rate_l1",
        "log_stock_market_index_ma4_l1",
        "unemp_rate_ma4_l1",
        "unemp_rate_ma4_l5",
        "log_real_oil_price_ma4_l1",
        "log_real_oil_price_ma4_l5",
        "log_real_fx_rate_ma4_l1",
        "log_real_fx_rate_ma4_l5",
    ]
    assert table.loc["constant", "coefficient"] == pytest.approx(0.25)
    assert table.loc["log_real_pc_income", "coefficient"] == pytest.approx(4.25)
    assert table.loc["log_stock_market_index_ma4_l1", "std_err_hac"] == pytest.approx(13.5)


def test__load_permanent_income_forecast_hac_covariance_maps_to_simulation_names(tmp_path):
    _, cov_path = _write_forecast_fixtures(tmp_path)
    covariance = load_permanent_income_forecast_hac_covariance(cov_path)
    assert covariance.shape == (20, 20)
    assert list(covariance.index) == list(covariance.columns)
    assert covariance.loc["constant", "constant"] == pytest.approx(1.0)
    assert covariance.loc["log_real_pc_income", "log_real_pc_income"] == pytest.approx(5.0)
    np.testing.assert_allclose(covariance.to_numpy(), covariance.to_numpy().T)


def test__load_permanent_income_forecast_inputs_combines_payloads(tmp_path):
    table_path, cov_path = _write_forecast_fixtures(tmp_path)
    inputs = load_permanent_income_forecast_inputs(table_path, cov_path)
    assert inputs.diagnostics["hac_lags"] == 8
    assert inputs.coefficient_table.loc["covid19", "p_hac"] == pytest.approx(3.01)
    assert inputs.hac_covariance.loc["log_real_fx_rate_ma4_l5", "log_real_fx_rate_ma4_l5"] == pytest.approx(20.0)
