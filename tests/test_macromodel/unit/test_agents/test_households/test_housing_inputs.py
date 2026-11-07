import numpy as np
import pytest

from macromodel.agents.households.func.housing_inputs import (
    add_housing_input_goods,
    housing_input_base,
    initial_housing_inputs,
)


def test_base_covers_owners_and_private_tenants_but_not_social_housing():
    rent = np.array([0.0, 10.0, 7.0])
    imputed = np.array([20.0, 0.0, 0.0])
    house_id = np.array([1, 2, -1])

    assert housing_input_base(rent, imputed, house_id).tolist() == [20.0, 10.0, 0.0]


def test_initial_payment_sums_to_the_moved_inputs():
    rate, weights, goods = initial_housing_inputs(np.array([3.0, 1.0]), np.array([30.0, 10.0, 0.0]), vat=0.25)

    assert rate == pytest.approx(0.125)
    assert weights.tolist() == pytest.approx([0.75, 0.25])
    assert goods.sum() == pytest.approx(4.0)
    assert goods[2] == 0.0


def test_no_moved_inputs_means_no_payment():
    rate, weights, goods = initial_housing_inputs(np.zeros(2), np.array([30.0]), vat=0.25)

    assert rate == 0.0
    assert weights.tolist() == [0.0, 0.0]
    assert goods.tolist() == [0.0]


def test_inputs_are_added_on_top_when_goods_ignore_rent():
    target = np.array([[6.0, 4.0]])

    result = add_housing_input_goods(target, np.array([2.0]), np.array([1.0, 0.0]), within_goods_budget=False)

    assert result[0].tolist() == pytest.approx([8.0, 4.0])


def test_inputs_stay_within_a_carved_out_goods_budget():
    target = np.array([[6.0, 4.0], [1.0, 1.0], [0.0, 0.0]])

    result = add_housing_input_goods(target, np.array([2.0, 5.0, 3.0]), np.array([1.0, 0.0]), within_goods_budget=True)

    assert result.sum(axis=1).tolist() == pytest.approx([10.0, 2.0, 0.0])
    assert result[0].tolist() == pytest.approx([6.8, 3.2])
    assert result[1].tolist() == pytest.approx([2.0, 0.0])


def test_calibrated_rule_keeps_inputs_inside_its_goods_budget():
    assert CreditAugmentedConsumption.uses_housing_carve_out is True
