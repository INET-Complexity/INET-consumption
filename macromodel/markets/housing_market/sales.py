"""Shared rules for matched property sales."""

import numpy as np
import pandas as pd

VACANT_HOUSEHOLD_ID = -1
#: Tenure status of a household with no property to live in.
SOCIAL_HOUSING_TENURE_STATUS = -1


def self_sale_mask(sales: pd.DataFrame) -> np.ndarray:
    """Return True for ``Sell`` rows whose buyer is the seller.

    An owner hoping to move lists its owner-occupied or vacant property, and
    matching can assign that property back to its owner. Such a row is a move
    (into an own vacant property) or a no-op (own current home), never a
    purchase: no price is paid, no mortgage is requested, no proceeds accrue
    and the property value is not reset.
    """
    if len(sales) == 0 or not {"sales_types", "buyer_id", "seller_id"}.issubset(sales.columns):
        return np.zeros(len(sales), dtype=bool)
    is_sell = sales["sales_types"].to_numpy() == "Sell"
    same_household = sales["buyer_id"].to_numpy(dtype=int) == sales["seller_id"].to_numpy(dtype=int)
    return is_sell & same_household


def sold_collateral_share(
    *,
    pre_sale_owner: pd.Series,
    pre_sale_value: pd.Series,
    completed_sales: pd.DataFrame,
    n_households: int,
) -> np.ndarray:
    """Share of each household's property value sold this period, at pre-sale values.

    Mortgages are recorded per household, not per property, so a mortgage is
    treated as secured on all the household's properties in proportion to
    value: selling properties worth ``V_sold`` out of ``V_own`` releases the
    share ``V_sold / V_own`` of the debt. ``pre_sale_owner`` and
    ``pre_sale_value`` are indexed by property id and taken before ownership
    and values are rewritten by sale processing. Self-sales sell nothing.
    """
    share = np.zeros(n_households)
    if len(completed_sales) == 0:
        return share
    purchases = completed_sales.loc[
        (completed_sales["sales_types"] == "Sell").to_numpy() & ~self_sale_mask(completed_sales)
    ]
    if len(purchases) == 0:
        return share
    values = pre_sale_value.fillna(0.0).astype(float)
    owners = pre_sale_owner.fillna(-1).astype(int)
    property_ids = purchases["property_id"].to_numpy(dtype=int)
    seller_ids = purchases["seller_id"].to_numpy(dtype=int)
    if np.any(owners.loc[property_ids].to_numpy() != seller_ids):
        raise ValueError("A completed sale's seller must own the property before the sale.")
    sold_value = np.bincount(seller_ids, weights=values.loc[property_ids].to_numpy(), minlength=n_households)
    owned = owners.to_numpy() >= 0
    owned_value = np.bincount(owners.to_numpy()[owned], weights=values.to_numpy()[owned], minlength=n_households)
    np.divide(sold_value, owned_value, out=share, where=owned_value > 0.0)
    return np.clip(share, 0.0, 1.0)


def move_chain_feasible(sales: pd.DataFrame, kept: np.ndarray, property_inhabitant: pd.Series) -> np.ndarray:
    """Drop kept sales whose property stays occupied by a household that does not move.

    A buyer can move in only if the property is vacant, already its home, or
    its inhabitant is itself the buyer of a kept sale. Dropping one sale can
    strand the next link of a chain, so the pass repeats until nothing changes.
    ``kept`` is a boolean mask over the rows of ``sales``; the result is a
    subset of it.
    """
    kept = np.asarray(kept, dtype=bool).copy()
    if not kept.any():
        return kept
    buyers = sales["buyer_id"].to_numpy(dtype=int)
    inhabitants = (
        property_inhabitant.fillna(VACANT_HOUSEHOLD_ID)
        .astype(int)
        .reindex(sales["property_id"].to_numpy(dtype=int))
        .fillna(VACANT_HOUSEHOLD_ID)
        .to_numpy(dtype=int)
    )
    while True:
        can_transfer = (
            (inhabitants == VACANT_HOUSEHOLD_ID) | (inhabitants == buyers) | np.isin(inhabitants, buyers[kept])
        )
        still_kept = kept & can_transfer
        if still_kept.sum() == kept.sum():
            return still_kept
        kept = still_kept


def median_price_to_value_ratio(prices: np.ndarray, previous_values: np.ndarray) -> float:
    """Median of price over the property's own previous value; 1 if there is no valid pair.

    Comparing each price with the value of the same property removes the
    effect of which properties are in the sample.
    """
    prices = np.asarray(prices, dtype=float)
    previous_values = np.asarray(previous_values, dtype=float)
    valid = np.isfinite(prices) & np.isfinite(previous_values) & (prices > 0.0) & (previous_values > 0.0)
    if not valid.any():
        return 1.0
    return float(np.median(prices[valid] / previous_values[valid]))


def revalue_unsold_properties(*, values: pd.Series, sold_property_ids: np.ndarray, ratio: float) -> pd.Series:
    """Scale the value of every property not sold this period by ``ratio``.

    ``values`` is indexed by property id and already holds the price for the
    properties sold; those are left as they are.
    """
    revalued = values.astype(float).copy()
    unsold = ~revalued.index.isin(np.asarray(sold_property_ids, dtype=int))
    revalued.loc[unsold] = revalued.loc[unsold] * ratio
    return revalued
