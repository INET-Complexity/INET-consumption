import json

import numpy as np
import pandas as pd
import pytest

from macro_data.readers import permanent_income_forecast as forecast_reader
from macro_data.readers.permanent_income_forecast import (
    RAW_VARIABLE_ORDER,
    PermanentIncomeForecastInputs,
    forecast_common_permanent_income,
    load_permanent_income_forecast_hac_covariance,
    load_permanent_income_forecast_inputs,
    load_permanent_income_forecast_residual_variance,
    load_permanent_income_forecast_table,
)


def _write_forecast_fixtures(tmp_path):
    table_path = tmp_path / "FR_table.json"
    cov_path = tmp_path / "FR_cov_hac.json"
    residual_variance_path = tmp_path / "FR_sigma2_u.json"

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
            covariance_payload[col_name][row_name] = 1.0 + col_idx if row_idx == col_idx else (row_idx + col_idx) / 100

    table_path.write_text(json.dumps(table_payload))
    cov_path.write_text(json.dumps(covariance_payload))
    residual_variance_path.write_text(json.dumps(7.517963432790559e-06))
    return table_path, cov_path, residual_variance_path


def _normalized_inputs(
    coefficients: list[float] | None = None,
    covariance: list[list[float]] | None = None,
    residual_variance: float = 0.5,
) -> PermanentIncomeForecastInputs:
    if coefficients is None:
        coefficients = [1.0, 2.0, -0.5]
    labels = ["constant", "time_trend", "log_real_pc_income"][: len(coefficients)]
    if covariance is None:
        covariance = [
            [0.1, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.3],
        ]
    return PermanentIncomeForecastInputs(
        coefficient_table=pd.DataFrame({"coefficient": coefficients}, index=labels),
        hac_covariance=pd.DataFrame(covariance, index=labels, columns=labels),
        residual_variance=residual_variance,
        diagnostics={},
    )


def _load_inputs_from_normalized(monkeypatch, inputs: PermanentIncomeForecastInputs):
    monkeypatch.setattr(
        forecast_reader,
        "load_permanent_income_forecast_table",
        lambda _path: inputs.coefficient_table,
    )
    monkeypatch.setattr(
        forecast_reader,
        "load_permanent_income_forecast_hac_covariance",
        lambda _path: inputs.hac_covariance,
    )
    monkeypatch.setattr(
        forecast_reader,
        "load_permanent_income_forecast_residual_variance",
        lambda _path: inputs.residual_variance,
    )
    return forecast_reader.load_permanent_income_forecast_inputs(
        "table.json",
        "cov.json",
        "sigma2.json",
    )


def test__forecast_common_permanent_income_computes_hand_checked_values():
    inputs = _normalized_inputs()
    x_t = pd.Series([1.0, 3.0, 4.0], index=inputs.coefficient_table.index)

    forecast = forecast_common_permanent_income(x_t, inputs)

    assert forecast.point_forecast == pytest.approx(5.0)
    assert forecast.forecast_variance == pytest.approx(7.2)


def test__forecast_common_permanent_income_includes_non_diagonal_covariance_terms():
    inputs = _normalized_inputs(
        covariance=[
            [0.1, 0.05, -0.02],
            [0.05, 0.2, 0.03],
            [-0.02, 0.03, 0.3],
        ],
        residual_variance=0.25,
    )
    x_t = pd.Series([2.0, -1.0, 0.5], index=inputs.coefficient_table.index)

    forecast = forecast_common_permanent_income(x_t, inputs)

    expected_variance = 0.25 + float(x_t.to_numpy() @ inputs.hac_covariance.to_numpy() @ x_t.to_numpy())
    assert forecast.point_forecast == pytest.approx(-0.25)
    assert forecast.forecast_variance == pytest.approx(expected_variance)


def test__forecast_common_permanent_income_rejects_unlabeled_x_t():
    inputs = _normalized_inputs()

    with pytest.raises(TypeError, match="labeled pandas Series"):
        forecast_common_permanent_income([1.0, 2.0, 3.0], inputs)


def test__forecast_common_permanent_income_rejects_reordered_x_t():
    inputs = _normalized_inputs()
    x_t = pd.Series([1.0, 4.0, 3.0], index=list(reversed(inputs.coefficient_table.index)))

    with pytest.raises(ValueError, match="exactly match"):
        forecast_common_permanent_income(x_t, inputs)


def test__forecast_common_permanent_income_rejects_missing_x_t_label():
    inputs = _normalized_inputs()
    x_t = pd.Series([1.0, 3.0], index=inputs.coefficient_table.index[:2])

    with pytest.raises(ValueError, match="exactly match"):
        forecast_common_permanent_income(x_t, inputs)


def test__forecast_common_permanent_income_rejects_mismatched_hac_axes():
    inputs = _normalized_inputs()
    inputs = PermanentIncomeForecastInputs(
        coefficient_table=inputs.coefficient_table,
        hac_covariance=inputs.hac_covariance.loc[
            list(reversed(inputs.hac_covariance.index)),
            list(reversed(inputs.hac_covariance.columns)),
        ],
        residual_variance=inputs.residual_variance,
        diagnostics={},
    )
    x_t = pd.Series([1.0, 3.0, 4.0], index=inputs.coefficient_table.index)

    with pytest.raises(ValueError, match="HAC covariance axes"):
        forecast_common_permanent_income(x_t, inputs)


def test__load_permanent_income_forecast_table_maps_to_simulation_names(tmp_path):
    table_path, _, _ = _write_forecast_fixtures(tmp_path)
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
    _, cov_path, _ = _write_forecast_fixtures(tmp_path)
    covariance = load_permanent_income_forecast_hac_covariance(cov_path)
    assert covariance.shape == (20, 20)
    assert list(covariance.index) == list(covariance.columns)
    assert covariance.loc["constant", "constant"] == pytest.approx(1.0)
    assert covariance.loc["log_real_pc_income", "log_real_pc_income"] == pytest.approx(5.0)
    np.testing.assert_allclose(covariance.to_numpy(), covariance.to_numpy().T)


def test__load_permanent_income_forecast_inputs_combines_payloads(tmp_path):
    table_path, cov_path, residual_variance_path = _write_forecast_fixtures(tmp_path)
    inputs = load_permanent_income_forecast_inputs(table_path, cov_path, residual_variance_path)
    assert inputs.diagnostics["hac_lags"] == 8
    assert inputs.residual_variance == pytest.approx(7.517963432790559e-06)
    assert inputs.coefficient_table.loc["covid19", "p_hac"] == pytest.approx(3.01)
    assert inputs.hac_covariance.loc[
        "log_real_fx_rate_ma4_l5",
        "log_real_fx_rate_ma4_l5",
    ] == pytest.approx(20.0)


def test__load_permanent_income_forecast_inputs_rejects_axis_order_mismatch(monkeypatch):
    inputs = _normalized_inputs()
    inputs = PermanentIncomeForecastInputs(
        coefficient_table=inputs.coefficient_table,
        hac_covariance=inputs.hac_covariance.loc[
            :,
            list(reversed(inputs.hac_covariance.columns)),
        ],
        residual_variance=inputs.residual_variance,
        diagnostics={},
    )

    with pytest.raises(ValueError, match="labels/order"):
        _load_inputs_from_normalized(monkeypatch, inputs)


def test__load_permanent_income_forecast_inputs_rejects_non_square_covariance(monkeypatch):
    inputs = _normalized_inputs()
    inputs = PermanentIncomeForecastInputs(
        coefficient_table=inputs.coefficient_table,
        hac_covariance=inputs.hac_covariance.iloc[:, :2],
        residual_variance=inputs.residual_variance,
        diagnostics={},
    )

    with pytest.raises(ValueError, match="square"):
        _load_inputs_from_normalized(monkeypatch, inputs)


def test__load_permanent_income_forecast_inputs_rejects_non_symmetric_covariance(monkeypatch):
    inputs = _normalized_inputs(
        coefficients=[1.0, 2.0],
        covariance=[
            [1.0, 0.2],
            [0.1, 1.0],
        ],
    )

    with pytest.raises(ValueError, match="symmetric"):
        _load_inputs_from_normalized(monkeypatch, inputs)


def test__load_permanent_income_forecast_inputs_rejects_covariance_beyond_symmetrization_tolerance(monkeypatch):
    # Asymmetry of 2e-5 exceeds the symmetrization tolerance (atol=1e-8, rtol=1e-5;
    # for entries ~1.0 the allclose threshold is ~1.001e-5), so the loader does not
    # symmetrize, and the strict validation symmetry check (tolerance=1e-10) then raises.
    inputs = _normalized_inputs(
        coefficients=[1.0, 2.0],
        covariance=[
            [1.0, 1.0],
            [1.0 + 2e-5, 1.0],
        ],
    )

    with pytest.raises(ValueError, match="symmetric"):
        _load_inputs_from_normalized(monkeypatch, inputs)


def test__load_permanent_income_forecast_inputs_rejects_non_psd_covariance(monkeypatch):
    inputs = _normalized_inputs(
        coefficients=[1.0, 2.0],
        covariance=[
            [1.0, 2.0],
            [2.0, 1.0],
        ],
    )

    with pytest.raises(ValueError, match="positive semidefinite"):
        _load_inputs_from_normalized(monkeypatch, inputs)


def test__load_permanent_income_forecast_inputs_rejects_non_finite_coefficients(monkeypatch):
    inputs = _normalized_inputs(coefficients=[1.0, np.inf, 3.0])

    with pytest.raises(ValueError, match="coefficients must be finite"):
        _load_inputs_from_normalized(monkeypatch, inputs)


def test__load_permanent_income_forecast_inputs_rejects_non_finite_covariance(monkeypatch):
    inputs = _normalized_inputs(
        covariance=[
            [0.1, 0.0, 0.0],
            [0.0, np.nan, 0.0],
            [0.0, 0.0, 0.3],
        ],
    )

    with pytest.raises(ValueError, match="covariance values must be finite"):
        _load_inputs_from_normalized(monkeypatch, inputs)


def test__load_permanent_income_forecast_inputs_rejects_non_finite_residual_variance(monkeypatch):
    inputs = _normalized_inputs(residual_variance=np.inf)

    with pytest.raises(ValueError, match="residual variance must be finite"):
        _load_inputs_from_normalized(monkeypatch, inputs)


def test__load_permanent_income_forecast_inputs_rejects_non_positive_residual_variance(monkeypatch):
    inputs = _normalized_inputs(residual_variance=0.0)

    with pytest.raises(ValueError, match="residual variance must be positive"):
        _load_inputs_from_normalized(monkeypatch, inputs)


def test__load_permanent_income_forecast_residual_variance_reads_scalar(tmp_path):
    _, _, residual_variance_path = _write_forecast_fixtures(tmp_path)
    residual_variance = load_permanent_income_forecast_residual_variance(residual_variance_path)
    assert residual_variance == pytest.approx(7.517963432790559e-06)
