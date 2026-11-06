"""Fixed worker labour inputs at initialisation."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macro_data.configuration.dataconfiguration import WorkerProductivityConfiguration
from macro_data.processing.synthetic_matching.worker_productivity import (
    apply_wage_floor,
    balanced_firm_assignment,
    effective_hours,
    match_workers_by_productivity,
)

CONFIG = WorkerProductivityConfiguration(annual_wage_floor=100.0, top_quantile=1.0)


def test_effective_hours_treats_missing_as_full_time_and_clips():
    hours = pd.Series([np.nan, 0.0, 2.0, 20.0, 50.0])
    assert effective_hours(hours, CONFIG).tolist() == [35.0, 35.0, 8.0, 20.0, 35.0]


def test_wage_floor_scales_with_hours_and_applies_to_employed_only():
    data = pd.DataFrame(
        {"Weekly Hours": [35.0, 17.5, 35.0], "Activity Status": [1, 1, 2], "Employee Income": [1.0, 1.0, 0.0]}
    )
    out = apply_wage_floor(data, CONFIG, scale=4, yearly_factor=4.0)
    assert out["Employee Income"].tolist() == [100.0, 50.0, 0.0]


def test_balanced_assignment_fills_each_firm_to_its_size():
    assignment = balanced_firm_assignment(np.arange(6.0), np.array([3, 7]), np.array([4, 2]))
    assert (assignment == 3).sum() == 4 and (assignment == 7).sum() == 2


@pytest.fixture
def economy():
    np.random.seed(0)
    n_employed, n_unemployed = 60, 10
    n = n_employed + n_unemployed
    data = pd.DataFrame(
        {
            "Activity Status": [1] * n_employed + [2] * n_unemployed,
            "Employment Industry": ([0] * 40 + [1] * 20) + [0] * n_unemployed,
            "Employee Income": list(np.random.lognormal(0, 0.5, n_employed)) + [0.0] * n_unemployed,
            "Effective Hours": [35.0] * 50 + [17.5] * 10 + [35.0] * n_unemployed,
            "Age": np.random.randint(20, 60, n).astype(float),
            "Education": np.random.randint(1, 4, n),
            "Gender": np.random.randint(1, 3, n),
            "Corresponding Firm ID": np.nan,
            "Unemployment Benefit Entitlement": 10.0,
            "Income from Unemployment Benefits": [0.0] * n_employed + [10.0] * n_unemployed,
        }
    )
    firm_data = pd.DataFrame(
        {
            "Industry": [0, 0, 1],
            "Number of Employees": [30, 10, 20],
            "Total Wages": [300.0, 100.0, 400.0],
            "Employees ID": [[], [], []],
        }
    ).astype({"Employees ID": object})

    def set_income():
        data["Income"] = data["Employee Income"] + data["Income from Unemployment Benefits"]

    population = SimpleNamespace(individual_data=data, set_income=set_income)
    match_workers_by_productivity(2, 0.8, SimpleNamespace(firm_data=firm_data), population, CONFIG)
    return data, firm_data


def test_firm_labour_equals_headcount_and_wage_bill_is_kept(economy):
    data, firm_data = economy
    for firm_id, firm in firm_data.iterrows():
        workers = data.loc[firm["Employees ID"]]
        assert len(workers) == firm["Number of Employees"]
        assert workers["Labour Inputs"].sum() == pytest.approx(firm["Number of Employees"])
        assert workers["Employee Income"].sum() == pytest.approx(0.8 * firm["Total Wages"])
        assert (workers["Corresponding Firm ID"] == firm_id).all()


def test_pay_within_firm_is_proportional_to_labour(economy):
    data, firm_data = economy
    workers = data.loc[firm_data.at[0, "Employees ID"]]
    ratio = workers["Employee Income"] / workers["Labour Inputs"]
    assert ratio.max() == pytest.approx(ratio.min())


def test_benefits_are_proportional_with_unchanged_total(economy):
    data, _ = economy
    unemployed = data[data["Activity Status"] == 2]
    assert (unemployed["Labour Inputs"] > 0).all()
    assert unemployed["Income from Unemployment Benefits"].sum() == pytest.approx(100.0)
    ratio = unemployed["Income from Unemployment Benefits"] / unemployed["Labour Inputs"]
    assert ratio.max() == pytest.approx(ratio.min())


def test_mismatched_firm_positions_raise(economy):
    data, firm_data = economy
    firm_data.loc[0, "Number of Employees"] = 31
    population = SimpleNamespace(individual_data=data, set_income=lambda: None)
    data["Employee Income"] = 1.0
    with pytest.raises(ValueError, match="firm positions"):
        match_workers_by_productivity(2, 0.8, SimpleNamespace(firm_data=firm_data), population, CONFIG)
