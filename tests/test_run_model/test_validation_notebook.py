import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import acf, ccf

RUN_MODEL_PATH = Path(__file__).resolve().parents[2] / "run_model"
if str(RUN_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL_PATH))


def _validation_cell(transient_periods):
    notebook = json.loads((RUN_MODEL_PATH / "run_model.ipynb").read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "validation_results, period_to_date = compute_validation_results(" in "".join(cell.get("source", []))
    )
    source = re.sub(
        r"^TRANSIENT_PERIODS = .+$", f"TRANSIENT_PERIODS = {transient_periods!r}", source, count=1, flags=re.M
    )
    return source.replace(
        "build_validation_coverage_table(validation_results,",
        "coverage = build_validation_coverage_table(validation_results,",
    )


@pytest.mark.parametrize("transient_periods", [0, 50])
def test_validation_cell_excludes_transient_without_shifting_dates(transient_periods):
    dates = pd.date_range("2017-01-01", periods=8, freq="QS")
    actual = pd.DataFrame(
        {
            ("ESP", "Real GDP (Value)"): [10, 14, 11, 16, 20, 17, 25, 30],
            ("ESP", "CPI (Value)"): [5, 6, 8, 7, 9, 10, 8, 13],
        },
        index=dates,
    )
    data = SimpleNamespace(
        configuration=SimpleNamespace(year=2020, quarter=1),
        calibration_data=actual,
    )
    periods = np.arange(60)
    scenario = pd.DataFrame({"real_gdp_ppi": periods + np.sin(periods), "cpi_transaction": periods + np.cos(periods)})
    namespace = {"data": data, "df_scenario": scenario, "COUNTRY": "ESP", "actual_policy_rate": None}

    exec(_validation_cell(transient_periods), namespace)

    result = namespace["validation_results"]["real_gdp"]
    expected_index = scenario.index[transient_periods:]
    assert result["simulated"].index.equals(expected_index)
    pd.testing.assert_series_equal(result["actual"], actual.iloc[:, 0].astype(float), check_names=False)
    np.testing.assert_allclose(
        result["acf_simulated"],
        acf(scenario["real_gdp_ppi"].iloc[transient_periods:], nlags=min(12, len(expected_index) - 1)),
    )
    np.testing.assert_allclose(result["acf_actual"], acf(actual.iloc[:, 0], nlags=7))
    cpi = namespace["validation_results"]["cpi"]
    np.testing.assert_allclose(
        cpi["ccf_actual"].loc[0:7], ccf(actual.iloc[:, 1], actual.iloc[:, 0], adjusted=True, nlags=8)
    )
    np.testing.assert_allclose(
        cpi["ccf_simulated"].loc[0:8],
        ccf(
            scenario["cpi_transaction"].iloc[transient_periods:],
            scenario["real_gdp_ppi"].iloc[transient_periods:],
            adjusted=True,
            nlags=9,
        ),
    )
    assert namespace["period_to_date"].loc[expected_index[0]] == pd.Timestamp("2020-01-01") + pd.DateOffset(
        months=3 * transient_periods
    )
    assert namespace["coverage"].loc["real_gdp", "n_model_periods"] == len(expected_index)
    assert namespace["coverage"].loc["real_gdp", "n_data_periods"] == len(dates)
    assert namespace["coverage"].loc["real_gdp", "n_total_model_periods"] == len(expected_index)
    assert len(scenario) == 60

    from src.validation import plot_validation_charts

    figure = plot_validation_charts(namespace["validation_results"], namespace["period_to_date"], "ESP")
    assert list(figure.data[0].x) == list(namespace["period_to_date"])
    assert list(figure.data[1].x) == list(dates)
    np.testing.assert_array_equal(figure.data[1].y, actual.iloc[:, 0])


def test_validation_chart_shows_actual_ccf_when_model_sample_is_too_short():
    from src.validation import plot_validation_charts

    actual = pd.DataFrame(
        {("ESP", "Real GDP (Value)"): [10, 14, 11, 16]},
        index=pd.date_range("2017-01-01", periods=4, freq="QS"),
    )
    data = SimpleNamespace(configuration=SimpleNamespace(year=2020, quarter=1), calibration_data=actual)
    namespace = {
        "data": data,
        "df_scenario": pd.DataFrame({"real_gdp_ppi": [3, 5]}),
        "COUNTRY": "ESP",
        "actual_policy_rate": None,
    }

    exec(_validation_cell(1), namespace)

    result = namespace["validation_results"]["real_gdp"]
    assert result["acf_simulated"].empty
    assert result["ccf_simulated"].empty
    assert not result["ccf_actual"].empty
    figure = plot_validation_charts(namespace["validation_results"], namespace["period_to_date"], "ESP")
    assert any(trace.xaxis == "x3" and trace.legendgroup == "actual" for trace in figure.data)


@pytest.mark.parametrize("transient_periods", [-1, 8, 1.5, True])
def test_validation_cell_rejects_invalid_transient(transient_periods):
    namespace = {"df_scenario": pd.DataFrame(index=range(8))}

    with pytest.raises(ValueError, match="TRANSIENT_PERIODS"):
        exec(_validation_cell(transient_periods), namespace)
