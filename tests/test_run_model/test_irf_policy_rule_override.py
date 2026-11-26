import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "run_model"))
from src.experiments.irf import _apply_policy_rule_overrides  # noqa: E402


def _configs():
    central_bank = SimpleNamespace(
        functions=SimpleNamespace(policy_rate=SimpleNamespace(parameters={"lower_bound": 0.0, "other": 1})),
        taylor_rule_overrides=SimpleNamespace(r_star=None, rho=None),
    )
    return {"FRA": SimpleNamespace(central_bank=central_bank)}


def test_removes_floor_and_sets_taylor_field_only_where_given():
    configs = _configs()
    _apply_policy_rule_overrides(configs, "FRA", {"lower_bound": None, "r_star": 0.01})
    cb = configs["FRA"].central_bank
    assert cb.functions.policy_rate.parameters == {"lower_bound": None, "other": 1}
    assert cb.taylor_rule_overrides.r_star == 0.01
    assert cb.taylor_rule_overrides.rho is None


def test_none_is_a_no_op_and_unknown_keys_fail():
    configs = _configs()
    _apply_policy_rule_overrides(configs, "FRA", None)
    assert configs["FRA"].central_bank.functions.policy_rate.parameters["lower_bound"] == 0.0
    with pytest.raises(ValueError, match="Unknown policy-rule"):
        _apply_policy_rule_overrides(configs, "FRA", {"floor": 0.0})
