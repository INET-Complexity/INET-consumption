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
