"""Load-time consistency between worker productivity rho and the wage setter."""

import pytest

from macromodel.agents.firms.func.wage_setter import (
    ContractWageSetter,
    validate_worker_productivity_configuration,
)


def _setter(initial_rate_source: str, offer_base: str) -> ContractWageSetter:
    return ContractWageSetter(
        labour_market_tightness_markup_scale=0.05,
        markup_time_span=4,
        initial_rate_source=initial_rate_source,
        offer_base=offer_base,
    )


def test_rho_with_consistent_setter_passes():
    validate_worker_productivity_configuration(True, _setter("individual", "per_labour_input"))


@pytest.mark.parametrize(
    ("initial_rate_source", "offer_base"),
    [("individual", "per_head"), ("firm_anchor", "per_labour_input"), ("firm_anchor", "per_head")],
)
def test_rho_with_inconsistent_setter_raises(initial_rate_source, offer_base):
    with pytest.raises(ValueError, match="worker_productivity"):
        validate_worker_productivity_configuration(True, _setter(initial_rate_source, offer_base))


def test_rho_with_setter_lacking_the_options_raises():
    with pytest.raises(ValueError, match="worker_productivity"):
        validate_worker_productivity_configuration(True, object())


@pytest.mark.parametrize(
    ("initial_rate_source", "offer_base"),
    [("individual", "per_head"), ("firm_anchor", "per_head"), ("firm_anchor", "per_labour_input")],
)
def test_without_rho_any_setting_passes(initial_rate_source, offer_base):
    validate_worker_productivity_configuration(False, _setter(initial_rate_source, offer_base))
