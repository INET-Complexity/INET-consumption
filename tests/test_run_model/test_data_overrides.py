"""Scenario overrides that change the data configuration (e.g. worker productivity rho)."""

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
