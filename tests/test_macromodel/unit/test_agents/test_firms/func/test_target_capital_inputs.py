import numpy as np
import pytest

from macromodel.agents.firms.func.target_capital_inputs import (
    FinancialTargetCapitalInputsSetter,
    TFPAdjustedTargetCapitalInputsSetter,
)


def _targets(setter, prev_stock, financed_share=None, **extra):
    """Two firms, one capital good: a = 0.1, y* = y0 = 10, K0 = 30, so benchmark K* = 30."""
    return setter.compute_unconstrained_target_capital_inputs(
        current_target_production=np.array([10.0, 10.0]),
        capital_input_use_matrix=np.array([[0.1], [0.1]]),
        prev_capital_inputs_stock=np.asarray(prev_stock, dtype=float)[:, None],
        initial_capital_inputs_stock=np.array([[30.0], [30.0]]),
        prev_production=np.array([10.0, 10.0]),
        initial_production=np.array([10.0, 10.0]),
        previous_good_prices=np.ones(1),
        substitution_bundle_matrix=np.eye(1),
        financed_share=financed_share,
        **extra,
    )[:, 0]


def test_without_memory_rule_is_replacement_plus_phi_gap():
    setter = FinancialTargetCapitalInputsSetter(0.25, 0.0)
    # firm 0: gap +10 -> 1 + 2.5; firm 1: gap -2 -> 1 - 0.5. s is ignored when memory is off.
    assert _targets(setter, [20.0, 32.0], financed_share=np.array([0.2, 0.2])) == pytest.approx([3.5, 0.5])


def test_memory_scales_only_the_expansion_gap():
    setter = FinancialTargetCapitalInputsSetter(0.25, 0.0, financed_share_memory=True)
    targets = _targets(setter, [20.0, 32.0], financed_share=np.array([0.2, 0.2]))
    # firm 0: 1 + 0.25 * 0.2 * 10; firm 1 (excess capital): unchanged 1 - 0.5
    assert targets == pytest.approx([1.5, 0.5])


def test_memory_with_full_financing_equals_the_original_rule():
    with_memory = FinancialTargetCapitalInputsSetter(0.25, 0.0, financed_share_memory=True)
    without_memory = FinancialTargetCapitalInputsSetter(0.25, 0.0)
    np.testing.assert_allclose(
        _targets(with_memory, [20.0, 32.0], financed_share=np.ones(2)),
        _targets(without_memory, [20.0, 32.0]),
    )


def test_memory_never_scales_replacement_and_keeps_the_floor():
    setter = FinancialTargetCapitalInputsSetter(0.25, 0.0, financed_share_memory=True)
    # s = 0: refused firm still plans replacement; large excess capital is floored at zero
    assert _targets(setter, [20.0, 80.0], financed_share=np.zeros(2)) == pytest.approx([1.0, 0.0])


def test_tfp_adjusted_setter_forwards_the_financed_share():
    setter = TFPAdjustedTargetCapitalInputsSetter(0.25, 0.0, financed_share_memory=True)
    targets = _targets(
        setter,
        [20.0, 20.0],
        financed_share=np.array([0.2, 1.0]),
        current_tfp_multiplier=np.ones(2),
        initial_tfp_multiplier=np.ones(2),
    )
    assert targets == pytest.approx([1.5, 3.5])
