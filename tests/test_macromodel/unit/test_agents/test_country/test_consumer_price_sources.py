import numpy as np
import pytest


class _StopAfterCapture(Exception):
    pass


def test__prepare_housing_market_clearing_uses_configured_period_inflation_for_rent(test_country, monkeypatch):
    test_country.economy.consumer_price_index_source = "fixed_basket_cpi"
    test_country.economy.ts.dicts["cpi_fixed_basket_pop_change"] = [[0.11], [0.12]]
    captured = {}
    if "properties" not in test_country.housing_market.states:
        test_country.housing_market.states = {"properties": test_country.housing_market.states}

    monkeypatch.setattr(test_country.housing_market, "update_property_value", lambda: None)
    monkeypatch.setattr(test_country.households, "prepare_housing_market_clearing", lambda **kwargs: None)

    def capture_update_rent(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(test_country.households, "update_rent", capture_update_rent)

    test_country.prepare_housing_market_clearing()

    np.testing.assert_allclose(captured["historic_inflation"], np.array([[0.11], [0.12]]))


def test__update_planning_metrics_uses_configured_period_inflation_for_benefits(test_country, monkeypatch):
    test_country.economy.consumer_price_index_source = "chained_basket_cpi"
    test_country.economy.ts.dicts["cpi_chained_basket_pop_change"] = [[0.21], [0.22]]
    test_country.economy.ts.dicts["estimated_cpi_inflation"] = [[0.03]]
    captured = {}

    def capture_update_benefits(**kwargs):
        captured.update(kwargs)
        raise _StopAfterCapture

    monkeypatch.setattr(test_country.central_government, "update_benefits", capture_update_benefits)

    with pytest.raises(_StopAfterCapture):
        test_country.update_planning_metrics()

    np.testing.assert_allclose(captured["historic_benefit_indexation_inflation"], np.array([[0.21], [0.22]]))
    assert captured["current_estimated_benefit_indexation_inflation"] == pytest.approx(0.03)


def test__update_planning_metrics_uses_configured_annual_inflation_for_central_bank(test_country, monkeypatch):
    test_country.economy.consumer_price_index_source = "fixed_basket_cpi"
    test_country.economy.ts.dicts["cpi_fixed_basket_yoy_change"] = [[0.07]]
    captured = {}

    monkeypatch.setattr(test_country.central_government, "update_benefits", lambda **kwargs: None)
    monkeypatch.setattr(
        test_country.central_government,
        "distribute_unemployment_benefits_to_individuals",
        lambda **kwargs: np.zeros(len(test_country.individuals.states["Activity Status"])),
    )

    def capture_compute_rate(**kwargs):
        captured.update(kwargs)
        raise _StopAfterCapture

    monkeypatch.setattr(test_country.central_bank, "compute_rate", capture_compute_rate)

    with pytest.raises(_StopAfterCapture):
        test_country.update_planning_metrics()

    assert captured["cpi_yoy_inflation"] == pytest.approx(0.07)


def test__update_planning_metrics_uses_configured_level_and_expected_inflation_for_consumption(
    test_country, monkeypatch
):
    test_country.economy.consumer_price_index_source = "fixed_basket_cpi"
    test_country.economy.ts.dicts["cpi_fixed_basket"] = [[1.0], [1.23]]
    test_country.economy.ts.dicts["cpi_fixed_basket_pop_change"] = [[0.04]]
    test_country.economy.ts.dicts["estimated_cpi_inflation"] = [[0.05]]
    captured = {}

    monkeypatch.setattr(test_country.central_government, "update_benefits", lambda **kwargs: None)
    monkeypatch.setattr(
        test_country.central_government,
        "distribute_unemployment_benefits_to_individuals",
        lambda **kwargs: np.zeros(len(test_country.individuals.states["Activity Status"])),
    )
    monkeypatch.setattr(test_country.central_bank, "compute_rate", lambda **kwargs: 0.01)

    def capture_compute_target_consumption(**kwargs):
        captured.update(kwargs)
        raise _StopAfterCapture

    monkeypatch.setattr(test_country.households, "compute_target_consumption", capture_compute_target_consumption)

    with pytest.raises(_StopAfterCapture):
        test_country.update_planning_metrics()

    assert captured["current_cpi"] == pytest.approx(1.23)
    assert captured["initial_cpi"] == pytest.approx(1.0)
    assert captured["expected_inflation"] == pytest.approx(0.05)
