"""Independent unit checks of paired path arithmetic and missing-data handling."""

import sys
from pathlib import Path

import numpy as np
import pytest

RUN_MODEL = Path(__file__).resolve().parents[3] / "run_model"
if str(RUN_MODEL) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL))

from src.experiments.paired_response import measure_paired_response, summarize_seed_responses  # noqa: E402


def inputs(**changes):
    data = dict(
        outcome="realised_total_consumption",
        intervention_kind="income_transfer",
        value_units="nominal",
        control_nominal=np.tile([100.0, 200.0], (8, 1)),
        shock_nominal=np.tile([220.0, 420.0], (8, 1)),
        control_ids=["a", "b"],
        shock_ids=["a", "b"],
        control_periods=list(range(21, 29)),
        shock_periods=list(range(21, 29)),
        control_deflators=np.full(8, 2.0),
        shock_deflators=np.full(8, 4.0),
        intervention_nominal=[20.0, 40.0],
        pre_shock_control_deflator=2.0,
    )
    data.update(changes)
    return data


def test_own_path_deflation_fixed_denominator_and_impact_four_eight_quarters():
    result = measure_paired_response(**inputs())
    assert result.response_kind == "income_mpc"
    np.testing.assert_array_equal(result.real_difference, np.full((8, 2), 5.0))
    np.testing.assert_array_equal(result.household_denominator_real, [10.0, 20.0])
    for horizon in (1, 4, 8):
        row = result.summaries[horizon]["all"]
        assert row["real_denominator"] == 30.0
        assert row["nominal_denominator"] == 60.0
        assert row["real_response"] == pytest.approx(horizon / 3)
        assert row["nominal_response"] == pytest.approx(horizon * 17 / 3)
        assert row["household_median"] == pytest.approx(horizon * 0.375)
        assert row["household_p10"] == pytest.approx(horizon * 0.275)
        assert row["household_p90"] == pytest.approx(horizon * 0.475)
    # Cumulative responses above one are retained without clipping.
    assert result.summaries[8]["all"]["real_response"] > 1


def test_time_varying_prices_and_explicit_short_smoke_horizons():
    result = measure_paired_response(
        **inputs(
            control_nominal=[[100, 200], [100, 200]],
            shock_nominal=[[220, 420], [220, 420]],
            control_deflators=[2, 4],
            shock_deflators=[4, 2],
            control_periods=[21, 22],
            shock_periods=[21, 22],
            horizons=(1, 2),
        )
    )
    assert result.summaries[2]["all"]["real_numerator"] == 255
    assert result.summaries[2]["all"]["real_response"] == 8.5


def test_scale_invariance_and_distinct_stock_response_label():
    source = inputs(intervention_kind="lfa_endowment")
    reference = measure_paired_response(**source)
    for key in ("control_nominal", "shock_nominal", "intervention_nominal"):
        source[key] = np.asarray(source[key]) * 5000
    scaled = measure_paired_response(**source)
    assert scaled.response_kind == "lfa_spending_response"
    assert scaled.summaries[8]["all"]["real_response"] == reference.summaries[8]["all"]["real_response"]
    assert scaled.summaries[1]["all"]["real_numerator"] == 5000 * reference.summaries[1]["all"]["real_numerator"]


def test_negative_responses_groups_empty_and_nonrecipient_effects_are_retained():
    result = measure_paired_response(
        **inputs(
            shock_nominal=np.tile([180.0, 420.0], (8, 1)),
            intervention_nominal=[20, 0],
            groups={"recipient": [True, False], "empty": [False, False]},
        )
    )
    row = result.summaries[1]["all"]
    # (-5 + 5)/10, including the nonrecipient's endogenous response.
    assert row["real_response"] == 0
    assert row["household_negative_share"] == 1
    assert row["defined_household_ratios"] == 1 and row["zero_injection_households"] == 1
    assert result.summaries[1]["recipient"]["real_response"] == -0.5
    empty = result.summaries[1]["empty"]
    assert empty["status"] == "empty_group" and empty["real_response"] is None


def test_zero_intervention_exposes_numerators_and_na_ratios():
    result = measure_paired_response(**inputs(intervention_nominal=[0, 0]))
    row = result.summaries[1]["all"]
    assert row["status"] == "zero_intervention"
    assert row["real_response"] is None and row["household_median"] is None
    assert row["real_numerator"] == 10
    assert row["household_count"] == 2


def test_results_are_detached_immutable_and_missing_input_never_filtered():
    source = inputs()
    result = measure_paired_response(**source)
    source["control_nominal"][:] = 0
    np.testing.assert_array_equal(result.real_difference, np.full((8, 2), 5.0))
    with pytest.raises(ValueError):
        result.real_difference.setflags(write=True)
    with pytest.raises(TypeError):
        result.summaries[1]["all"]["real_response"] = 0
    with pytest.raises(ValueError):
        measure_paired_response(**inputs(shock_nominal=np.full((8, 2), np.nan)))


@pytest.mark.parametrize(
    "changes",
    [
        {"shock_ids": ["b", "a"]},
        {"control_ids": ["a", "a"]},
        {"control_ids": [True, False]},
        {"shock_periods": list(range(22, 30))},
        {"control_periods": [21, 22, 23, 25, 26, 27, 28, 29]},
        {"shock_nominal": np.ones((7, 2))},
        {"shock_deflators": np.zeros(8)},
        {"control_deflators": [2]},
        {"pre_shock_control_deflator": 0},
        {"pre_shock_control_deflator": [2]},
        {"intervention_nominal": [-1, 2]},
        {"intervention_nominal": [np.inf, 2]},
        {"horizons": (1, 9)},
        {"horizons": (1, 1)},
        {"horizons": (True,)},
        {"groups": {"all": [True, True]}},
        {"groups": {"bad": [1, 0]}},
        {"value_units": "real"},
        {"intervention_kind": "calibration_14"},
        {"shock_nominal": np.full((8, 2), 1e308)},
    ],
)
def test_invalid_or_incomplete_design_rejected(changes):
    with pytest.raises(ValueError):
        measure_paired_response(**inputs(**changes))


def test_seed_summary_uses_registered_cohort_sample_sd_and_reports_missing():
    result = summarize_seed_responses({12: -1.0, 13: 2.0, 14: 5.0}, registered_seeds=[12, 13, 14])
    assert result["mean"] == 2 and result["sd"] == 3
    assert result["min"] == -1 and result["max"] == 5
    missing = summarize_seed_responses({12: 1.0, 13: 2.0}, registered_seeds=[12, 13, 14])
    assert missing["status"] == "incomplete" and missing["mean"] is None
    assert missing["per_seed"][14] is None and missing["available_seed_count"] == 2
    single = summarize_seed_responses({12: 1.0}, registered_seeds=[12])
    assert single["mean"] == 1 and single["sd"] is None
    with pytest.raises(ValueError):
        summarize_seed_responses({12: np.nan}, registered_seeds=[12])
    with pytest.raises(ValueError):
        summarize_seed_responses({15: 1}, registered_seeds=[12])
