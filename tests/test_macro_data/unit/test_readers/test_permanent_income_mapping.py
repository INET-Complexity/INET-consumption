from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macro_data.readers.permanent_income_forecast import RAW_VARIABLE_ORDER
from macro_data.readers.permanent_income_mapping import (
    FORECAST_READER_TO_NOTEBOOK_EXPORT_NAME,
    FROZEN_EXOGENOUS_FORECAST_READER_NAMES,
    PermanentIncomeSimulationSources,
    build_permanent_income_forecast_regressors,
    design_matrix_to_forecast_reader_names,
    load_permanent_income_design_matrix,
    rebase_real_pc_income_index,
)

FR_DESIGN_MATRIX_PATH = (
    Path(__file__).resolve().parents[4]
    / "run_model"
    / "data"
    / "raw_data"
    / "permanent_income"
    / "FR_design_matrix.csv"
)


def _design_matrix(periods: pd.PeriodIndex) -> pd.DataFrame:
    rows = []
    for row_idx, period in enumerate(periods):
        row = {"date": period.to_timestamp(how="end")}
        for col_idx, raw_name in enumerate(RAW_VARIABLE_ORDER):
            row[raw_name] = 1_000.0 + 100.0 * row_idx + col_idx
        row["const"] = 1.0
        row["trend"] = -99.0
        row["splittrend_pdv"] = -88.0
        row["covid19"] = 1.0
        row["log_y"] = np.log(200.0 + row_idx)
        row["d4_log_y"] = 0.01 * row_idx
        row["cci_fs_ma4_l1"] = 700.0 + row_idx
        row["stock_log_ma4_l1"] = 800.0 + row_idx
        row["oil_log_ma4_l1"] = 900.0 + row_idx
        row["oil_log_ma4_l5"] = 1_000.0 + row_idx
        row["fx_log_ma4_l1"] = 1_100.0 + row_idx
        row["fx_log_ma4_l5"] = 1_200.0 + row_idx
        row["log_work_age_pop_share"] = 4.0 + 0.01 * row_idx
        row["t_bill_d4"] = 50.0 + row_idx
        row["t_bill_d4_l1"] = 60.0 + row_idx
        row["rr_ma4_l1"] = 70.0 + row_idx
        row["rr_ma4_l5"] = 80.0 + row_idx
        row["rr_ma4_l9"] = 90.0 + row_idx
        row["u_ma4_l1"] = 10.0 + row_idx
        row["u_ma4_l5"] = 20.0 + row_idx
        rows.append(row)
    return pd.DataFrame(rows)


def _fisher(policy_rate, cpi, periods_per_year=4):
    policy_rate = np.asarray(policy_rate, dtype=float)
    cpi = np.asarray(cpi, dtype=float)
    annual_nominal_rate = policy_rate[4:] * periods_per_year
    yoy_inflation = cpi[4:] / cpi[:-4] - 1.0
    return 100.0 * ((1.0 + annual_nominal_rate) / (1.0 + yoy_inflation) - 1.0)


def test__design_matrix_to_forecast_reader_names_maps_notebook_exports_and_quarter_end_dates():
    periods = pd.period_range("2014Q1", periods=2, freq="Q")
    design = design_matrix_to_forecast_reader_names(_design_matrix(periods))

    assert isinstance(design.index, pd.PeriodIndex)
    assert list(design.index) == list(periods)
    assert "log_real_pc_income" in design.columns
    assert "log_y" not in design.columns
    assert FORECAST_READER_TO_NOTEBOOK_EXPORT_NAME["real_interest_rate_ma4_l1"] == "rr_ma4_l1"


def test__mapping_uses_income_index_scale_directly_and_bootstraps_lags_from_design_matrix_during_warmup():
    periods = pd.period_range("2014Q1", periods=8, freq="Q")
    design = _design_matrix(periods)

    x_t = build_permanent_income_forecast_regressors(
        sources=PermanentIncomeSimulationSources(
            current_period="2015Q4",
            real_pc_income=[92.0, 94.0, 96.0, 98.0],
            policy_rate=[0.01, 0.02, 0.03, 0.04],
            cpi_fixed_basket=[100.0, 101.0, 102.0, 103.0],
            unemployment_rate=[0.05, 0.06, 0.07, 0.08],
        ),
        design_matrix=design,
        start_period=1980,
    )

    assert x_t["log_real_pc_income"] == pytest.approx(np.log(98.0))
    assert x_t["d4_log_real_pc_income"] == pytest.approx(0.07)
    assert x_t["real_interest_rate_ma4_l1"] == pytest.approx(77.0)
    assert x_t["real_interest_rate_ma4_l5"] == pytest.approx(87.0)
    assert x_t["unemp_rate_ma4_l1"] == pytest.approx(17.0)
    assert x_t["constant"] == pytest.approx(1.0)
    assert x_t["time_trend"] == pytest.approx(143.0)
    assert x_t["split_trend_from_2009q4_discounted_present_value"] == pytest.approx(-88.0)
    assert x_t["covid19"] == pytest.approx(0.0)


def test__start_period_aligns_trend_without_rebasing_income_index():
    periods = pd.period_range("1980Q1", periods=5, freq="Q")
    design = _design_matrix(periods)

    sources = PermanentIncomeSimulationSources(
        current_period="1981Q1",
        real_pc_income=[90.0, 91.0, 92.0, 93.0, 94.0],
        policy_rate=[0.01] * 5,
        cpi_fixed_basket=[100.0, 101.0, 102.0, 103.0, 104.0],
        unemployment_rate=[0.05] * 5,
    )

    x_t_1980_anchor = build_permanent_income_forecast_regressors(
        sources=sources,
        design_matrix=design,
        start_period=1980,
    )
    x_t_1979_anchor = build_permanent_income_forecast_regressors(
        sources=sources,
        design_matrix=design,
        start_period="1979Q1",
    )

    assert x_t_1980_anchor["time_trend"] == pytest.approx(4.0)
    assert x_t_1979_anchor["time_trend"] == pytest.approx(8.0)
    assert x_t_1980_anchor["log_real_pc_income"] == pytest.approx(np.log(94.0))
    assert x_t_1979_anchor["log_real_pc_income"] == pytest.approx(np.log(94.0))


def test__mapping_does_not_fall_back_to_start_period_or_initial_design_income_anchor():
    periods = pd.period_range("2014Q1", periods=5, freq="Q")
    design = _design_matrix(periods)

    x_t = build_permanent_income_forecast_regressors(
        sources=PermanentIncomeSimulationSources(
            current_period="2015Q1",
            real_pc_income=[80.0, 82.0, 84.0, 86.0, 88.0],
            policy_rate=[0.01] * 5,
            cpi_fixed_basket=[100.0, 101.0, 102.0, 103.0, 104.0],
            unemployment_rate=[0.05] * 5,
        ),
        design_matrix=design,
        start_period=1980,
    )

    initial_design_index = 200.0
    assert x_t["log_real_pc_income"] == pytest.approx(np.log(88.0))
    assert x_t["log_real_pc_income"] != pytest.approx(np.log(initial_design_index * 88.0 / 80.0))


def test__mapping_uses_endogenous_income_real_rate_and_percent_unemployment_after_warmup():
    periods = pd.period_range("2014Q1", periods=20, freq="Q")
    design = _design_matrix(periods)
    real_pc_income = np.linspace(92.0, 108.0, 17)
    policy_rate = np.linspace(0.005, 0.025, 17)
    cpi = np.linspace(100.0, 120.0, 17)
    unemployment_share = np.linspace(0.002, 0.008, 17)

    x_t = build_permanent_income_forecast_regressors(
        sources=PermanentIncomeSimulationSources(
            current_period="2018Q1",
            real_pc_income=real_pc_income,
            policy_rate=policy_rate,
            cpi_fixed_basket=cpi,
            unemployment_rate=unemployment_share,
        ),
        design_matrix=design,
        start_period="1980Q1",
    )

    real_rate = _fisher(policy_rate, cpi)
    unemployment_percent = unemployment_share * 100.0
    assert x_t["d4_log_real_pc_income"] == pytest.approx(np.log(real_pc_income[-1] / real_pc_income[-5]))
    assert x_t["real_interest_rate_ma4_l1"] == pytest.approx(real_rate[-5:-1].mean())
    assert x_t["real_interest_rate_ma4_l5"] == pytest.approx(real_rate[-9:-5].mean())
    assert x_t["real_interest_rate_ma4_l9"] == pytest.approx(real_rate[-13:-9].mean())
    assert x_t["unemp_rate_ma4_l1"] == pytest.approx(unemployment_percent[-5:-1].mean())
    assert x_t["unemp_rate_ma4_l5"] == pytest.approx(unemployment_percent[-9:-5].mean())


def test__mapping_freezes_selected_exogenous_terms_at_initial_period_including_t_bill():
    periods = pd.period_range("2014Q1", periods=10, freq="Q")
    design = _design_matrix(periods)

    x_t = build_permanent_income_forecast_regressors(
        sources=PermanentIncomeSimulationSources(
            current_period="2016Q2",
            real_pc_income=np.linspace(10.0, 14.0, 10),
            policy_rate=[0.01] * 10,
            cpi_fixed_basket=np.linspace(100.0, 109.0, 10),
            unemployment_rate=np.linspace(0.05, 0.10, 10),
        ),
        design_matrix=design,
        start_period=1980,
    )

    initial = design_matrix_to_forecast_reader_names(design).loc[pd.Period("2014Q1")]
    current = design_matrix_to_forecast_reader_names(design).loc[pd.Period("2016Q2")]
    assert x_t["survey_expectations_ma4_l1"] == pytest.approx(initial["survey_expectations_ma4_l1"])
    assert x_t["log_stock_market_index_ma4_l1"] == pytest.approx(initial["log_stock_market_index_ma4_l1"])
    assert x_t["log_real_oil_price_ma4_l5"] == pytest.approx(initial["log_real_oil_price_ma4_l5"])
    assert x_t["log_real_fx_rate_ma4_l1"] == pytest.approx(initial["log_real_fx_rate_ma4_l1"])
    assert x_t["log_working_age_population_share"] == pytest.approx(initial["log_working_age_population_share"])
    assert x_t["d4_t_bill_rate"] == pytest.approx(initial["d4_t_bill_rate"])
    assert x_t["d4_t_bill_rate_l1"] == pytest.approx(initial["d4_t_bill_rate_l1"])
    assert x_t["d4_t_bill_rate"] != pytest.approx(current["d4_t_bill_rate"])
    assert x_t["d4_t_bill_rate_l1"] != pytest.approx(current["d4_t_bill_rate_l1"])


def test__unemployment_rate_is_converted_from_share_to_percent():
    periods = pd.period_range("2014Q1", periods=8, freq="Q")
    design = _design_matrix(periods)

    x_t = build_permanent_income_forecast_regressors(
        sources=PermanentIncomeSimulationSources(
            current_period="2015Q4",
            real_pc_income=np.linspace(90.0, 98.0, 8),
            policy_rate=[0.01] * 8,
            cpi_fixed_basket=np.linspace(100.0, 103.0, 8),
            unemployment_rate=np.linspace(0.05, 0.08, 8),
        ),
        design_matrix=design,
        start_period=1980,
    )

    assert x_t["unemp_rate_ma4_l1"] == pytest.approx(np.linspace(0.05, 0.08, 8)[-5:-1].mean() * 100.0)
    assert x_t["unemp_rate_ma4_l1"] > 1.0


def test__rebase_real_pc_income_index_hand_computed():
    levels = np.array([200.0, 202.0, 204.0, 196.0])

    rebased = rebase_real_pc_income_index(levels, base_period_index=0)

    assert rebased == pytest.approx([100.0, 101.0, 102.0, 98.0])


def test__rebase_real_pc_income_index_non_zero_base_index():
    levels = np.array([50.0, 100.0, 150.0])

    rebased = rebase_real_pc_income_index(levels, base_period_index=1, index_base_value=50.0)

    assert rebased == pytest.approx([25.0, 50.0, 75.0])


def test__load_permanent_income_design_matrix_loads_fr_csv_and_converts():
    design_matrix = load_permanent_income_design_matrix(FR_DESIGN_MATRIX_PATH)

    assert "date" in design_matrix.columns
    assert "log_y" in design_matrix.columns
    assert not any(str(col).startswith("Unnamed") for col in design_matrix.columns)
    assert len(design_matrix) == 144

    converted = design_matrix_to_forecast_reader_names(design_matrix)
    assert isinstance(converted.index, pd.PeriodIndex)
    assert "log_real_pc_income" in converted.columns
    assert pd.Period("2014Q1", freq="Q") in converted.index
    assert pd.Period("2014Q4", freq="Q") in converted.index


def test__build_permanent_income_forecast_regressors_matches_fr_design_matrix_reference_rows():
    design_matrix = load_permanent_income_design_matrix(FR_DESIGN_MATRIX_PATH)
    converted = design_matrix_to_forecast_reader_names(design_matrix)
    initial_row = converted.loc[pd.Period("2014Q1", freq="Q")]

    for current_period_label in ("2014Q1", "2014Q4"):
        current_period = pd.Period(current_period_label, freq="Q")
        history_length = (current_period.year - 2014) * 4 + (current_period.quarter - 1) + 1

        x_t = build_permanent_income_forecast_regressors(
            sources=PermanentIncomeSimulationSources(
                current_period=current_period,
                real_pc_income=np.full(history_length, 100.0),
                policy_rate=np.full(history_length, 0.01),
                cpi_fixed_basket=np.full(history_length, 100.0),
                unemployment_rate=np.full(history_length, 0.05),
            ),
            design_matrix=design_matrix,
            start_period="2014Q1",
        )

        for name in FROZEN_EXOGENOUS_FORECAST_READER_NAMES:
            assert x_t[name] == pytest.approx(float(initial_row[name])), name

        expected_trend = (current_period.year - 2014) * 4 + (current_period.quarter - 1)
        assert x_t["time_trend"] == pytest.approx(expected_trend)
        assert x_t["covid19"] == pytest.approx(0.0)


def test__build_permanent_income_forecast_regressors_clamps_periods_beyond_design_matrix_range():
    design_matrix = load_permanent_income_design_matrix(FR_DESIGN_MATRIX_PATH)
    converted = design_matrix_to_forecast_reader_names(design_matrix)
    last_row_period = converted.index.max()
    last_row = converted.loc[last_row_period]

    # current_period (and, with history_length=1, initial_period too) is far beyond
    # the design matrix's last available row, so both _design_row_or_nearest lookups
    # should clamp to the last row instead of raising.
    current_period = last_row_period + 20

    x_t = build_permanent_income_forecast_regressors(
        sources=PermanentIncomeSimulationSources(
            current_period=current_period,
            real_pc_income=[100.0],
            policy_rate=[0.01],
            cpi_fixed_basket=[100.0],
            unemployment_rate=[0.05],
        ),
        design_matrix=design_matrix,
        start_period="2014Q1",
    )

    assert np.isfinite(x_t.to_numpy(dtype=float)).all()

    for name in FROZEN_EXOGENOUS_FORECAST_READER_NAMES:
        assert x_t[name] == pytest.approx(float(last_row[name])), name


def test__estimation_epoch_anchors_time_trend_independently_of_start_period():
    """time_trend must equal 146 at 2016Q3 when epoch is 1980Q1 (FR estimation sample)."""
    periods = pd.period_range("2014Q1", periods=11, freq="Q")
    design = _design_matrix(periods)

    x_t_with_epoch = build_permanent_income_forecast_regressors(
        sources=PermanentIncomeSimulationSources(
            current_period="2016Q3",
            real_pc_income=np.linspace(90.0, 101.0, 11),
            policy_rate=[0.01] * 11,
            cpi_fixed_basket=np.linspace(100.0, 110.0, 11),
            unemployment_rate=[0.05] * 11,
        ),
        design_matrix=design,
        start_period="2014Q1",
        estimation_epoch="1980Q1",
    )

    x_t_without_epoch = build_permanent_income_forecast_regressors(
        sources=PermanentIncomeSimulationSources(
            current_period="2016Q3",
            real_pc_income=np.linspace(90.0, 101.0, 11),
            policy_rate=[0.01] * 11,
            cpi_fixed_basket=np.linspace(100.0, 110.0, 11),
            unemployment_rate=[0.05] * 11,
        ),
        design_matrix=design,
        start_period="2014Q1",
    )

    # 2016Q3 is (2016-1980)*4 + (3-1) = 144 + 2 = 146 quarters after 1980Q1
    assert x_t_with_epoch["time_trend"] == pytest.approx(146.0)
    # Without estimation_epoch, start_period=2014Q1 is used: 2016Q3 = 10 quarters after 2014Q1
    assert x_t_without_epoch["time_trend"] == pytest.approx(10.0)
    # All other regressors are unaffected
    assert x_t_with_epoch["log_real_pc_income"] == pytest.approx(x_t_without_epoch["log_real_pc_income"])
    assert x_t_with_epoch["constant"] == pytest.approx(1.0)


def test__estimation_epoch_none_falls_back_to_start_period():
    """Explicit estimation_epoch=None must give the same result as omitting it."""
    periods = pd.period_range("2014Q1", periods=11, freq="Q")
    design = _design_matrix(periods)

    sources = PermanentIncomeSimulationSources(
        current_period="2016Q3",
        real_pc_income=np.linspace(90.0, 101.0, 11),
        policy_rate=[0.01] * 11,
        cpi_fixed_basket=np.linspace(100.0, 110.0, 11),
        unemployment_rate=[0.05] * 11,
    )

    x_t_none = build_permanent_income_forecast_regressors(
        sources=sources,
        design_matrix=design,
        start_period="2014Q1",
        estimation_epoch=None,
    )
    x_t_default = build_permanent_income_forecast_regressors(
        sources=sources,
        design_matrix=design,
        start_period="2014Q1",
    )

    assert x_t_none["time_trend"] == pytest.approx(x_t_default["time_trend"])


def test__unemployment_rate_outside_unit_interval_raises():
    periods = pd.period_range("2014Q1", periods=5, freq="Q")
    design = _design_matrix(periods)

    with pytest.raises(ValueError, match="unemployment_rate"):
        build_permanent_income_forecast_regressors(
            sources=PermanentIncomeSimulationSources(
                current_period="2015Q1",
                real_pc_income=[90.0] * 5,
                policy_rate=[0.01] * 5,
                cpi_fixed_basket=[100.0] * 5,
                unemployment_rate=[5.0] * 5,
            ),
            design_matrix=design,
            start_period=1980,
        )
