"""Shared rules for matched property sales."""

import numpy as np
import pandas as pd


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
