"""Household payments for the intermediate inputs of housing services.

Rents are gross. Part of what owners impute and tenants pay covers inputs bought from other
sectors. These payments reach suppliers through goods demand, and the rents are held net of them.
"""

import numpy as np


def housing_input_base(rent: np.ndarray, rent_imputed: np.ndarray, inhabited_house_id: np.ndarray) -> np.ndarray:
    """Return the net rent on which the payment is due. Social housing tenants have no house ID."""
    in_private_housing = np.asarray(inhabited_house_id) != -1
    return np.asarray(rent_imputed, dtype=float) + np.where(in_private_housing, np.asarray(rent, dtype=float), 0.0)


def initial_housing_inputs(
    housing_input_demand: np.ndarray, housing_flows: np.ndarray, vat: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """Calibrate the payment so that its total equals the moved inputs at t=0.

    Args:
        housing_input_demand (np.ndarray): Moved inputs by industry per period, before VAT.
        housing_flows (np.ndarray): Net rent base of each household.
        vat (float): Value added tax rate.

    Returns:
        tuple[float, np.ndarray, np.ndarray]: Cash payment per unit of net rent, input mix by
            industry, and each household's initial goods purchase before VAT.
    """
    demand = np.asarray(housing_input_demand, dtype=float)
    if demand.sum() <= 0.0:
        return 0.0, np.zeros_like(demand), np.zeros_like(housing_flows, dtype=float)
    rate = (1.0 + vat) * demand.sum() / np.sum(housing_flows)
    return rate, demand / demand.sum(), rate * housing_flows / (1.0 + vat)


def add_housing_input_goods(
    target_consumption: np.ndarray, housing_input_goods: np.ndarray, weights: np.ndarray, within_goods_budget: bool
) -> np.ndarray:
    """Add housing inputs to a household goods matrix.

    Rules that carve housing out of a total budget already give the goods budget, which covers the
    inputs. There the inputs replace other goods, up to that budget. Other rules set goods
    independently of rent, so the inputs are added on top.

    Args:
        target_consumption (np.ndarray): Goods by household and industry, before VAT.
        housing_input_goods (np.ndarray): Each household's inputs, before VAT.
        weights (np.ndarray): Input mix by industry.
        within_goods_budget (bool): Whether the goods total already covers the inputs.

    Returns:
        np.ndarray: Goods by household and industry, including the inputs.
    """
    if not within_goods_budget:
        return target_consumption + np.outer(housing_input_goods, weights)
    goods = target_consumption.sum(axis=1)
    inputs = np.minimum(housing_input_goods, goods)
    other_share = 1.0 - np.divide(inputs, goods, out=np.zeros_like(goods), where=goods > 0.0)
    return target_consumption * other_share[:, None] + np.outer(inputs, weights)
