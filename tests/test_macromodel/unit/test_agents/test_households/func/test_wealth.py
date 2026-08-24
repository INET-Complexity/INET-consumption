import numpy as np
import pytest

from macromodel.agents.households.func.wealth import PaperAssetReturnWealthSetter


def _paper_setter(**overrides):
    params = {
        "other_real_assets_depreciation_rate": 0.05,
        "mu_eq": 0.0029,
        "mu_bond": 0.0081,
        "sigma_eq": 0.0,
        "sigma_bond": 0.0,
        "rho": 0.0,
        "equity_weight": 0.5,
        "draw_scope": "country_period",
    }
    params.update(overrides)
    return PaperAssetReturnWealthSetter(**params)


def test__paper_asset_return_wealth_setter_uses_common_illiquid_return():
    setter = _paper_setter(mu_eq=0.10, mu_bond=0.00, equity_weight=0.5)
    opening_ifa = np.array([100.0, 200.0, -10.0])

    income = setter.compute_income_from_financial_assets(opening_ifa, period_index=1)
    updated_ifa = setter.compute_wealth_in_other_financial_assets(
        current_wealth_in_other_financial_assets=opening_ifa,
        new_wealth_in_other_financial_assets=np.array([1.0, 2.0, 3.0]),
        used_up_wealth_in_other_financial_assets=np.array([4.0, 5.0, 6.0]),
        period_index=1,
    )

    expected_rate = np.exp(0.05) - 1.0
    np.testing.assert_allclose(income, np.array([100.0, 200.0, 0.0]) * expected_rate)
    assert setter.current_illiquid_return_rate() == pytest.approx(expected_rate)
    np.testing.assert_allclose(
        updated_ifa, opening_ifa + income + np.array([1.0, 2.0, 3.0]) - np.array([4.0, 5.0, 6.0])
    )


def test__paper_asset_return_wealth_setter_seeded_draw_is_reproducible():
    opening_ifa = np.array([100.0, 200.0])
    setter = _paper_setter(sigma_eq=0.0935, sigma_bond=0.0316, rho=-0.2585)

    np.random.seed(123)
    first = setter.compute_income_from_financial_assets(opening_ifa)
    first_rate = setter.current_illiquid_return_rate()
    np.random.seed(123)
    second = setter.compute_income_from_financial_assets(opening_ifa)
    second_rate = setter.current_illiquid_return_rate()

    assert first_rate == pytest.approx(second_rate)
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first / opening_ifa, np.full(opening_ifa.shape, first_rate))


def test__paper_asset_return_wealth_setter_expected_return_includes_log_variance():
    setter = _paper_setter(
        mu_eq=0.10,
        mu_bond=0.02,
        sigma_eq=0.20,
        sigma_bond=0.10,
        rho=0.25,
        equity_weight=0.75,
    )

    expected_rate = (
        np.exp(
            0.75 * 0.10
            + 0.25 * 0.02
            + 0.5 * (0.75**2 * 0.20**2 + 0.25**2 * 0.10**2 + 2.0 * 0.75 * 0.25 * 0.25 * 0.20 * 0.10)
        )
        - 1.0
    )

    assert setter.expected_illiquid_return_rate() == pytest.approx(expected_rate)


def test__paper_asset_return_wealth_setter_caps_drawdown_after_negative_return():
    setter = _paper_setter(mu_eq=np.log(0.2), mu_bond=np.log(0.2), equity_weight=0.5)
    opening_ifa = np.array([100.0])

    setter.compute_income_from_financial_assets(opening_ifa, period_index=1)
    used_deposits, used_ifa = setter.use_up_wealth(
        used_up_wealth=np.array([50.0]),
        current_wealth_in_deposits=np.array([100.0]),
        current_wealth_in_other_financial_assets=opening_ifa,
        period_index=1,
    )
    updated_ifa = setter.compute_wealth_in_other_financial_assets(
        current_wealth_in_other_financial_assets=opening_ifa,
        new_wealth_in_other_financial_assets=np.array([0.0]),
        used_up_wealth_in_other_financial_assets=used_ifa,
        period_index=1,
    )

    np.testing.assert_allclose(used_ifa, np.array([20.0]))
    np.testing.assert_allclose(used_deposits, np.array([30.0]))
    np.testing.assert_allclose(updated_ifa, np.array([0.0]), atol=1e-12)


def test__paper_asset_return_wealth_setter_rejects_negative_volatility():
    with pytest.raises(ValueError, match="sigma_eq and sigma_bond must be non-negative"):
        _paper_setter(sigma_eq=-0.01)

    with pytest.raises(ValueError, match="sigma_eq and sigma_bond must be non-negative"):
        _paper_setter(sigma_bond=-0.01)


def test__paper_asset_return_wealth_setter_reuses_one_draw_until_wealth_update(monkeypatch):
    setter = _paper_setter()
    opening_ifa = np.array([100.0, 200.0])
    return_rates = iter([0.10, 0.20])
    monkeypatch.setattr(setter, "draw_illiquid_return_rate", lambda: next(return_rates))

    first_income = setter.compute_income_from_financial_assets(opening_ifa, period_index=1)
    second_income = setter.compute_income_from_financial_assets(opening_ifa, period_index=1)
    updated_ifa = setter.compute_wealth_in_other_financial_assets(
        current_wealth_in_other_financial_assets=opening_ifa,
        new_wealth_in_other_financial_assets=np.array([0.0, 0.0]),
        used_up_wealth_in_other_financial_assets=np.array([0.0, 0.0]),
        period_index=1,
    )

    with pytest.raises(ValueError, match="already been applied for the current period"):
        setter.compute_income_from_financial_assets(updated_ifa, period_index=1)

    next_period_income = setter.compute_income_from_financial_assets(updated_ifa, period_index=2)

    np.testing.assert_allclose(first_income, np.array([10.0, 20.0]))
    np.testing.assert_allclose(second_income, first_income)
    np.testing.assert_allclose(updated_ifa, np.array([110.0, 220.0]))
    np.testing.assert_allclose(next_period_income, np.array([22.0, 44.0]))


def test__paper_asset_return_wealth_setter_allows_planning_draw_before_period_binding(monkeypatch):
    setter = _paper_setter()
    opening_ifa = np.array([100.0, 200.0])
    monkeypatch.setattr(setter, "draw_illiquid_return_rate", lambda: 0.1)

    planning_income = setter.compute_income_from_financial_assets(opening_ifa)
    realized_income = setter.compute_income_from_financial_assets(opening_ifa, period_index=1)
    updated_ifa = setter.compute_wealth_in_other_financial_assets(
        current_wealth_in_other_financial_assets=opening_ifa,
        new_wealth_in_other_financial_assets=np.array([0.0, 0.0]),
        used_up_wealth_in_other_financial_assets=np.array([0.0, 0.0]),
        period_index=1,
    )

    np.testing.assert_allclose(planning_income, np.array([10.0, 20.0]))
    np.testing.assert_allclose(realized_income, planning_income)
    np.testing.assert_allclose(updated_ifa, np.array([110.0, 220.0]))


def test__paper_asset_return_wealth_setter_requires_period_draw_before_wealth_update():
    setter = _paper_setter()

    with pytest.raises(ValueError, match="has not been drawn for the current period"):
        setter.compute_wealth_in_other_financial_assets(
            current_wealth_in_other_financial_assets=np.array([100.0]),
            new_wealth_in_other_financial_assets=np.array([0.0]),
            used_up_wealth_in_other_financial_assets=np.array([0.0]),
            period_index=1,
        )


def test__paper_asset_return_wealth_setter_accepts_scalar_target_share_source_by_default():
    setter = _paper_setter(uses_portfolio_choice=True, fixed_cost_share=0.0)
    assert setter.target_share_source == "scalar"


def test__paper_asset_return_wealth_setter_accepts_frm_magnitude_target_share_source():
    setter = _paper_setter(uses_portfolio_choice=True, target_share_source="frm_magnitude", fixed_cost_share=0.0)
    assert setter.target_share_source == "frm_magnitude"


def test__paper_asset_return_wealth_setter_rejects_unknown_target_share_source_when_portfolio_choice_active():
    with pytest.raises(ValueError, match="target_share_source"):
        _paper_setter(uses_portfolio_choice=True, target_share_source="precomputed", fixed_cost_share=0.0)


def test__paper_asset_return_wealth_setter_does_not_validate_target_share_source_when_portfolio_choice_off():
    # uses_portfolio_choice=False (the default) leaves target_share_source inert,
    # so an invalid value must not block construction of an otherwise-ordinary
    # asset-return setter that never reads it.
    setter = _paper_setter(target_share_source="not-a-real-source")
    assert setter.uses_portfolio_choice is False


def test__paper_asset_return_wealth_setter_accepts_settled_portfolio_choice_with_choice_enabled():
    setter = _paper_setter(uses_portfolio_choice=True, settles_portfolio_choice=True)
    assert setter.settles_portfolio_choice is True


def test__paper_asset_return_wealth_setter_rejects_settlement_without_choice():
    with pytest.raises(ValueError, match="uses_portfolio_choice"):
        _paper_setter(settles_portfolio_choice=True)


def test__paper_asset_return_wealth_setter_rejects_dynamic_shifters_enabled():
    with pytest.raises(ValueError, match="dynamic_shifters_enabled"):
        _paper_setter(dynamic_shifters_enabled=True)


def test__paper_asset_return_wealth_setter_rejects_liquid_asset_policy_rate_markup():
    with pytest.raises(ValueError, match="liquid_asset_policy_rate_markup"):
        _paper_setter(liquid_asset_policy_rate_markup=1.0)


def test__paper_asset_return_wealth_setter_rejects_unsupported_participation_source():
    with pytest.raises(ValueError, match="participation_source"):
        _paper_setter(participation_source="dynamic")


def test__paper_asset_return_wealth_setter_rejects_unsupported_liquid_return_source():
    with pytest.raises(ValueError, match="liquid_return_source"):
        _paper_setter(liquid_return_source="deposit_rate")
