"""Scenario overrides that change the data configuration (e.g. worker productivity rho)."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

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


def test_base_fra_config_has_rho_on():
    rho = nw._load_data_config(FRA_CFG).country_configs["FRA"].worker_productivity
    assert type(rho).__name__ == "WorkerProductivityConfiguration"
    assert rho.annual_wage_floor == 13544.0


def test_wage_competition_off_preset_splits_into_data_and_country_keys():
    data, country = nw.split_scenario_overrides(SCENARIO_PRESETS["wage_competition_off"])
    assert data == {"worker_productivity": None}
    assert country == {
        "labour_market.functions.clearing.parameters['on_the_job_search']": False,
        "labour_market.functions.clearing.parameters['individuals_quitting_temperature']": 1.0,
        "firms.functions.wage_setter.parameters['offer_ceiling']": "none",
        "firms.functions.wage_setter.parameters['slack_response_down']": 0.0,
    }


def test_apply_data_overrides_validates_and_leaves_base_unchanged():
    base = nw._load_data_config(FRA_CFG)
    data, _ = nw.split_scenario_overrides(SCENARIO_PRESETS["wage_competition_off"])
    updated = nw.apply_data_config_overrides(base, data)
    assert updated.country_configs["FRA"].worker_productivity is None
    assert base.country_configs["FRA"].worker_productivity.annual_wage_floor == 13544.0


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


def test_base_fra_config_is_the_wage_competition_specification():
    assert SCENARIO_PRESETS["wage_competition"] == {}
    fra = yaml.safe_load((RUN_MODEL_PATH / "config" / "country_config_FRA.yaml").read_text())["FRA"]
    firms = fra["firms"]
    wage = firms["functions"]["wage_setter"]["parameters"]
    clearing = fra["labour_market"]["functions"]["clearing"]["parameters"]
    assert firms["parameters"]["pricing_labour_units"] == "tfp_effective"
    assert wage["incumbent_indexation_pass_through"] == 0.1
    assert wage["indexation_base"] == "tfp_multiplier"
    assert wage["indexation_common_component"] == "lagged_full"
    assert wage["offer_ceiling"] == "value_added"
    assert wage["slack_response_down"] == 0.05
    assert wage["initial_rate_source"] == "individual" and wage["offer_base"] == "per_labour_input"
    assert clearing["on_the_job_search"] is True
    assert abs((1.0 - math.exp(-clearing["individuals_quitting_temperature"])) - 0.033) < 1e-12
    off = SCENARIO_PRESETS["wage_competition_off"]
    for path in off:
        if not path.startswith("data."):
            assert path.startswith(("firms.functions.wage_setter.", "labour_market.functions.clearing."))


def test_cache_with_other_rho_setting_requires_rebuild():
    data_config = nw._load_data_config(FRA_CFG)
    rho = data_config.country_configs["FRA"].worker_productivity

    def cached(worker_productivity):
        return SimpleNamespace(
            synthetic_countries={
                "FRA": SimpleNamespace(population=SimpleNamespace(worker_productivity=worker_productivity))
            }
        )

    assert not nw._requires_worker_productivity_cache_rebuild(cached(rho.model_copy()), data_config)
    assert nw._requires_worker_productivity_cache_rebuild(cached(None), data_config)
    off = nw.apply_data_config_overrides(data_config, {"worker_productivity": None})
    assert nw._requires_worker_productivity_cache_rebuild(cached(rho), off)
    assert not nw._requires_worker_productivity_cache_rebuild(cached(None), off)
