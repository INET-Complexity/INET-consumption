"""Scenario overrides that change the data configuration (e.g. worker productivity rho)."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

RUN_MODEL_PATH = Path(__file__).resolve().parents[2] / "run_model"
if str(RUN_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL_PATH))

from config import SCENARIO_PRESETS  # noqa: E402
from src import notebook_workflow as nw  # noqa: E402

FRA_CFG = SimpleNamespace(config_dir=RUN_MODEL_PATH / "config", country_iso3="FRA")


def test_split_separates_data_keys():
    data, country = nw.split_scenario_overrides({"data.worker_productivity": {"annual_wage_floor": 1.0}, "a.b": 2})
    assert data == {"worker_productivity": {"annual_wage_floor": 1.0}}
    assert country == {"a.b": 2}


def test_base_fra_config_has_rho_off():
    assert nw._load_data_config(FRA_CFG).country_configs["FRA"].worker_productivity is None


def test_workers_productivity_preset_carries_all_four_settings():
    data, country = nw.split_scenario_overrides(SCENARIO_PRESETS["workers_productivity"])
    assert data == {"worker_productivity": {"annual_wage_floor": 13544.0}}
    assert country == {
        "firms.functions.wage_setter.parameters['initial_rate_source']": "individual",
        "firms.functions.wage_setter.parameters['offer_base']": "per_labour_input",
        "labour_market.functions.clearing.parameters['sorted_firing']": False,
    }


def test_apply_data_overrides_validates_and_leaves_base_unchanged():
    base = nw._load_data_config(FRA_CFG)
    data, _ = nw.split_scenario_overrides(SCENARIO_PRESETS["workers_productivity"])
    updated = nw.apply_data_config_overrides(base, data)
    rho = updated.country_configs["FRA"].worker_productivity
    assert type(rho).__name__ == "WorkerProductivityConfiguration"
    assert rho.annual_wage_floor == 13544.0
    assert base.country_configs["FRA"].worker_productivity is None


def test_apply_data_overrides_rejects_unknown_field():
    with pytest.raises(ValueError, match="Unknown data-config field"):
        nw.apply_data_config_overrides(nw._load_data_config(FRA_CFG), {"no_such_field": 1})


def test_data_overrides_get_their_own_cache_file():
    assert nw.data_cache_name_for("data.pkl", {}) == "data.pkl"
    rho = nw.data_cache_name_for("data.pkl", {"worker_productivity": {"annual_wage_floor": 13544.0}})
    other = nw.data_cache_name_for("data.pkl", {"worker_productivity": {"annual_wage_floor": 1.0}})
    assert rho != "data.pkl" and rho.endswith(".pkl") and rho != other


def test_country_overrides_refuse_data_keys():
    with pytest.raises(ValueError, match="must be applied when the data are built"):
        nw.apply_country_config_overrides(SimpleNamespace(), {"data.worker_productivity": {}})


def test_wage_plan_presets_are_cumulative_and_build_on_worker_productivity():
    arms = ["A5", "B", "C", "D", "E"]
    previous = SCENARIO_PRESETS["workers_productivity"]
    for arm in arms:
        preset = SCENARIO_PRESETS[f"wage_plan_{arm}"]
        assert previous.items() < preset.items()  # strict superset: one step added, nothing changed
        previous = preset
    full = SCENARIO_PRESETS["wage_plan_E"]
    assert full["firms.parameters.pricing_labour_units"] == "tfp_effective"
    assert full["firms.functions.wage_setter.parameters['incumbent_indexation_pass_through']"] == 0.1
    assert full["firms.functions.wage_setter.parameters['indexation_base']"] == "tfp_multiplier"
    assert full["labour_market.functions.clearing.parameters['on_the_job_search']"] is True
    search_intensity = full["labour_market.functions.clearing.parameters['individuals_quitting_temperature']"]
    assert abs((1.0 - math.exp(-search_intensity)) - 0.015) < 1e-12
    assert full["firms.functions.wage_setter.parameters['offer_ceiling']"] == "value_added"
    assert full["firms.functions.wage_setter.parameters['slack_response_down']"] == 0.05
    data, _ = nw.split_scenario_overrides(full)
    assert data == {"worker_productivity": {"annual_wage_floor": 13544.0}}
