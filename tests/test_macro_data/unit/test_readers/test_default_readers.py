from datetime import date
from pathlib import Path

import numpy as np

from macro_data.configuration.countries import Country
from macro_data.readers.default_readers import DataPaths, prune_icio_dict
from macro_data.readers.exogenous_data import create_all_exogenous_data


def test__default_paths_resolves_calibration_paths():
    raw_data_path = Path("/tmp/raw_data")
    paths = DataPaths.default_paths(raw_data_path, icio_years=[2014])
    assert paths.insee_smic_path == raw_data_path / "INSEE" / "SMIC_monthly_values.csv"
    assert paths.income_belief_priors_path == raw_data_path / "CES" / "FR_priors_table5.csv"
    assert paths.permanent_income_forecast_table_path == (
        raw_data_path / "permanent_income" / "FR_table.json"
    )
    assert paths.permanent_income_forecast_covariance_path == (
        raw_data_path / "permanent_income" / "FR_cov_hac.json"
    )
    assert paths.permanent_income_forecast_residual_variance_path == (
        raw_data_path / "permanent_income" / "FR_sigma2_u.json"
    )


def test__prune_icio_dict():
    dummy_dict = {
        2014: "dummy",
        2015: "dummy",
        2016: "dummy",
        2017: "dummy",
        2018: "dummy",
        2019: "dummy",
        2020: "dummy",
    }

    prune_date = date(year=2016, month=1, day=1)

    pruned_dict = prune_icio_dict(dummy_dict, prune_date)
    assert pruned_dict == {2016: "dummy", 2017: "dummy", 2018: "dummy", 2019: "dummy", 2020: "dummy"}


def test__get_benefits_inflation(readers):
    france = Country("FRA")
    exogenous_data = create_all_exogenous_data(readers, [france])
    data = readers.get_benefits_inflation_data(france, 2004, 2014, exogenous_data[france])
    assert data.shape[0] > 0


def test__create_exogenous_data(readers):
    exog_data = create_all_exogenous_data(readers, [Country("FRA")])
    assert exog_data["FRA"]["log_inflation"].shape[0] > 0


def test__readers_disagg_can(readers_disagg_can):
    assert "B05a" in readers_disagg_can.icio[2014].industries


def test__emissions(readers):
    data = readers.emissions.get_emissions_factors(2014)

    assert data["coal"] > 0
    assert data["oil"] > 0
    assert data["gas"] > 0


def test__readers_provincial_can(readers_provincial_can):
    assert "CAN_AB" in readers_provincial_can.icio[2014].considered_countries
    iot = readers_provincial_can.icio[2014].iot

    nontot_rows = iot.index.get_level_values(0) != "TOTAL"

    non_tot_cols = iot.columns.get_level_values(0) != "TOTAL"

    ind_cols = iot.columns.get_level_values(1).isin(readers_provincial_can.icio[2014].industries)

    # compute output from sums over columns
    output_column = iot.loc[nontot_rows, ("TOTAL", "Output")]
    output_from_sums = iot.loc[nontot_rows, non_tot_cols].sum(axis=1)

    assert np.allclose(output_column, output_from_sums, rtol=1e-5)

    intermediate_inputs_sum = iot.loc[nontot_rows, ind_cols].sum(axis=0)
    intermediate_inputs = iot.loc[("TOTAL", "Intermediate Inputs"), ind_cols]

    assert np.allclose(intermediate_inputs_sum, intermediate_inputs, rtol=1e-5)
