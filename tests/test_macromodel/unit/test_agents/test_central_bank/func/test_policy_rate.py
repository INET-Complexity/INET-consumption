import pytest

from macromodel.agents.central_bank.func.policy_rate import ConstantPolicyRate, SmoothTaylorRule


class TestPolicyRate:
    def test__compute_rate(self):
        assert (
            ConstantPolicyRate().compute_rate(
                prev_rate=0.01,
                central_bank_states={},
                growth=0.01,
                inflation=0.01,
            )
            == 0.01
        )

    def test__smooth_taylor_rule_converts_annual_rate_to_period_rate(self):
        rate = SmoothTaylorRule().compute_rate(
            prev_rate=0.01,
            inflation=0.03,
            growth=0.0,
            central_bank_states={
                "rho": 0.5,
                "r_star": 0.02,
                "targeted_inflation_rate": 0.02,
                "phi_pi": 1.5,
                "phi_q": 0.5,
            },
            cpi_yoy_inflation=0.03,
            output_gap=0.02,
            time_unit=3,
        )

        # Annual rate = 0.5 * 0.04 + 0.5 * (0.02 + 0.02 + 1.5 * 0.01 + 0.5 * 0.02) = 0.0525
        assert rate == pytest.approx(0.0525 / 4.0)

    @pytest.mark.parametrize(
        ("lower_bound", "expected_annual"),
        [(None, -0.03), (0.0, 0.0), (-0.005, -0.005), (-0.1, -0.03)],
    )
    def test__smooth_taylor_rule_lower_bound_on_the_annual_rate(self, lower_bound, expected_annual):
        rate = SmoothTaylorRule(lower_bound=lower_bound).compute_rate(
            prev_rate=0.0,
            inflation=0.0,
            growth=0.0,
            central_bank_states={
                "rho": 0.0,
                "r_star": -0.01,
                "targeted_inflation_rate": 0.02,
                "phi_pi": 2.0,
                "phi_q": 0.0,
            },
            cpi_yoy_inflation=0.0,
            output_gap=0.0,
            time_unit=3,
        )

        # Unbounded annual rate = -0.01 + 0.02 + 2.0 * (0.0 - 0.02) = -0.03
        assert rate == pytest.approx(expected_annual / 4.0)

    def test__smooth_taylor_rule_rejects_a_non_finite_lower_bound(self):
        with pytest.raises(ValueError, match="lower_bound"):
            SmoothTaylorRule(lower_bound=float("nan"))
