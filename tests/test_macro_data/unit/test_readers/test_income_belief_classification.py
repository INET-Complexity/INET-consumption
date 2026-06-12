import numpy as np
import pytest

from macro_data.readers.income_belief_classification import (
    classify_age_group,
    classify_employment_status,
    compute_household_income_beliefs,
)
from macro_data.readers.income_belief_priors import load_income_belief_priors_table
from macromodel.agents.individuals.individual_properties import ActivityStatus


@pytest.fixture(scope="module")
def priors_table():
    return load_income_belief_priors_table()


def test__classify_age_group_bins_and_clips():
    ages = np.array([10, 18, 34, 35, 49, 50, 70, 71, 85, 95])
    groups = classify_age_group(ages)
    assert list(groups) == [
        "18-34",
        "18-34",
        "18-34",
        "35-49",
        "35-49",
        "50-70",
        "50-70",
        "71-85",
        "71-85",
        "71-85",
    ]


def test__classify_employment_status_maps_all_activity_statuses():
    statuses = np.array(
        [
            ActivityStatus.EMPLOYED.value,
            ActivityStatus.UNEMPLOYED.value,
            ActivityStatus.NOT_ECONOMICALLY_ACTIVE.value,
            ActivityStatus.FIRM_INVESTOR.value,
            ActivityStatus.BANK_INVESTOR.value,
        ]
    )
    assert list(classify_employment_status(statuses)) == [
        "employed",
        "unemployed",
        "retired",
        "retired",
        "retired",
    ]


def test__compute_household_income_beliefs_income_weighted_average(priors_table):
    # Household 0: 40yo employed earning 3000, 68yo retired earning 1000
    age = np.array([40, 68])
    activity_status = np.array([ActivityStatus.EMPLOYED.value, ActivityStatus.NOT_ECONOMICALLY_ACTIVE.value])
    household_id = np.array([0, 0])
    income = np.array([3000.0, 1000.0])

    result = compute_household_income_beliefs(
        individuals_age=age,
        individuals_activity_status=activity_status,
        individuals_household_id=household_id,
        individuals_income=income,
        n_households=1,
        priors_table=priors_table,
    )

    employed_3549 = priors_table.loc[("employed", "35-49")]
    retired_5070 = priors_table.loc[("retired", "50-70")]
    w1, w2 = 0.75, 0.25

    assert result["income_belief_mu"][0] == pytest.approx(
        w1 * employed_3549["mean_income_exp_dev"] + w2 * retired_5070["mean_income_exp_dev"]
    )
    assert result["income_belief_p"][0] == pytest.approx(
        w1 * employed_3549["mean_income_uncertainty_var_prior"] + w2 * retired_5070["mean_income_uncertainty_var_prior"]
    )
    assert result["sigma2_xi"][0] == pytest.approx(
        w1 * employed_3549["sigma2_xi"] + w2 * retired_5070["sigma2_xi"]
    )
    assert result["sigma2_v"][0] == pytest.approx(
        w1 * employed_3549["sigma2_v"] + w2 * retired_5070["sigma2_v"]
    )


def test__compute_household_income_beliefs_equal_weight_fallback_when_zero_income(priors_table):
    age = np.array([40, 68])
    activity_status = np.array([ActivityStatus.EMPLOYED.value, ActivityStatus.NOT_ECONOMICALLY_ACTIVE.value])
    household_id = np.array([0, 0])
    income = np.array([0.0, 0.0])

    result = compute_household_income_beliefs(
        individuals_age=age,
        individuals_activity_status=activity_status,
        individuals_household_id=household_id,
        individuals_income=income,
        n_households=1,
        priors_table=priors_table,
    )

    employed_3549 = priors_table.loc[("employed", "35-49")]
    retired_5070 = priors_table.loc[("retired", "50-70")]

    assert result["income_belief_mu"][0] == pytest.approx(
        0.5 * employed_3549["mean_income_exp_dev"] + 0.5 * retired_5070["mean_income_exp_dev"]
    )


def test__compute_household_income_beliefs_updates_when_member_group_changes(priors_table):
    age = np.array([40, 68])
    household_id = np.array([0, 0])
    income = np.array([3000.0, 1000.0])

    employed = compute_household_income_beliefs(
        individuals_age=age,
        individuals_activity_status=np.array(
            [ActivityStatus.EMPLOYED.value, ActivityStatus.NOT_ECONOMICALLY_ACTIVE.value]
        ),
        individuals_household_id=household_id,
        individuals_income=income,
        n_households=1,
        priors_table=priors_table,
    )

    # Member 0 loses their job
    unemployed = compute_household_income_beliefs(
        individuals_age=age,
        individuals_activity_status=np.array(
            [ActivityStatus.UNEMPLOYED.value, ActivityStatus.NOT_ECONOMICALLY_ACTIVE.value]
        ),
        individuals_household_id=household_id,
        individuals_income=np.array([0.0, 1000.0]),
        n_households=1,
        priors_table=priors_table,
    )

    assert unemployed["sigma2_xi"][0] != employed["sigma2_xi"][0]
    assert unemployed["sigma2_v"][0] != employed["sigma2_v"][0]


def test__compute_household_income_beliefs_uses_fallback_for_unemployed_71_85(priors_table):
    age = np.array([80.0])
    activity_status = np.array([ActivityStatus.UNEMPLOYED.value])
    household_id = np.array([0])
    income = np.array([0.0])

    result = compute_household_income_beliefs(
        individuals_age=age,
        individuals_activity_status=activity_status,
        individuals_household_id=household_id,
        individuals_income=income,
        n_households=1,
        priors_table=priors_table,
    )

    unemployed_5070 = priors_table.loc[("unemployed", "50-70")]
    assert result["sigma2_xi"][0] == pytest.approx(unemployed_5070["sigma2_xi"])
    assert result["sigma2_v"][0] == pytest.approx(unemployed_5070["sigma2_v"])
    assert result["income_belief_p"][0] == pytest.approx(unemployed_5070["mean_income_uncertainty_var_prior"])
