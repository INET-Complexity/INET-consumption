"""Completion rule for matched property purchases.

A matched purchase completes only if the buyer can pay for it after reserving
this period's cash needs. The rule is pure: it reads the matched sales, the
buyers' balance sheets and the mortgages granted, and returns which sales
complete and how each buyer funds its purchase. Nothing is booked here.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from macromodel.markets.housing_market.sales import move_chain_feasible, self_sale_mask, sold_collateral_share


@dataclass(frozen=True)
class PurchaseCompletionInputs:
    """What the completion rule needs, taken before credit clearing.

    ``current_sales`` holds the matched rows (sales and rentals). The property
    series are indexed by property id. The household vectors are: ``cash_need``
    (N, this period's cash uses not covered by expected income), opening liquid
    and illiquid assets, the Stage 5 liquidation already planned, and the
    illiquid-trade cost.
    """

    current_sales: pd.DataFrame
    property_owner: pd.Series
    property_value: pd.Series
    property_inhabitant: pd.Series
    cash_need: np.ndarray
    liquid_assets: np.ndarray
    illiquid_assets: np.ndarray
    planned_liquidation: np.ndarray
    sale_cost: np.ndarray


@dataclass(frozen=True)
class PurchaseCompletion:
    """Outcome of the completion rule.

    ``kept_sales`` is a mask over the rows of the matched ``current_sales``
    (rentals are always kept). The rest are household vectors:
    ``mortgage_zeroed`` marks granted mortgages that must not be booked,
    ``net_housing_cash_need`` is X = price - mortgage - (proceeds - repayment),
    ``purchase_liquidation`` the illiquid assets sold to pay X (net of the trade
    cost) and ``housing_lfa_draw`` (D) the liquid assets X spends.
    """

    kept_sales: np.ndarray
    mortgage_zeroed: np.ndarray
    sale_proceeds: np.ndarray
    mortgage_sale_repayment: np.ndarray
    net_housing_cash_need: np.ndarray
    purchase_liquidation: np.ndarray
    housing_lfa_draw: np.ndarray


def resolve_purchase_completion(
    inputs: PurchaseCompletionInputs,
    *,
    granted_mortgage: np.ndarray,
    principal_left: np.ndarray,
    mortgage_rejected: Callable[[np.ndarray], np.ndarray] | None = None,
) -> PurchaseCompletion:
    """Return the largest set of matched sales that can all complete together.

    Buyer ``h`` completes if ``X_h <= max(LFA_h - N_h, 0) + max(IFA_h -
    planned_h - cost_h, 0)``. ``X_h`` counts the proceeds of the buyer's own
    kept sales net of the mortgage repaid on them, so a mover's purchase
    depends on its own sale, which depends on its buyer's purchase. The kept
    set is therefore a greatest fixed point: start from every matched sale and
    repeat {proceeds and repayment from kept sales, mortgage test, completion
    test, move-chain pass} until no sale is dropped. Each step can only drop
    sales, so the loop ends after at most one pass per sale.

    ``granted_mortgage`` is the mortgage cleared per household,
    ``principal_left`` the opening mortgage principal left after this period's
    scheduled service, and ``mortgage_rejected(repayment)`` the underwriting
    test given the mortgage repaid on sale (no test if ``None``). Self-sales
    move no money and always complete.
    """
    sales = inputs.current_sales
    granted = np.asarray(granted_mortgage, dtype=float)
    n_households = granted.shape[0]
    principal_left = np.asarray(principal_left, dtype=float)
    free_lfa = np.maximum(
        np.asarray(inputs.liquid_assets, dtype=float) - np.asarray(inputs.cash_need, dtype=float), 0.0
    )
    free_ifa = np.maximum(
        np.asarray(inputs.illiquid_assets, dtype=float)
        - np.asarray(inputs.planned_liquidation, dtype=float)
        - np.asarray(inputs.sale_cost, dtype=float),
        0.0,
    )
    for name, values in (("principal_left", principal_left), ("free_lfa", free_lfa), ("free_ifa", free_ifa)):
        if values.shape != (n_households,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Purchase completion requires a finite household vector for {name}.")

    zeros = np.zeros(n_households)
    if len(sales) == 0 or "sales_types" not in sales.columns:
        return PurchaseCompletion(
            kept_sales=np.ones(len(sales), dtype=bool),
            mortgage_zeroed=granted > 0.0,
            sale_proceeds=zeros,
            mortgage_sale_repayment=zeros.copy(),
            net_housing_cash_need=zeros.copy(),
            purchase_liquidation=zeros.copy(),
            housing_lfa_draw=zeros.copy(),
        )

    is_sell = sales["sales_types"].to_numpy() == "Sell"
    is_self_sale = self_sale_mask(sales)
    buyers = sales["buyer_id"].to_numpy(dtype=int)
    sellers = sales["seller_id"].to_numpy(dtype=int)
    prices = sales["price_or_rent"].to_numpy(dtype=float)

    kept = is_sell.copy()
    while True:
        purchases = kept & ~is_self_sale
        proceeds = np.bincount(sellers[purchases], weights=prices[purchases], minlength=n_households)
        price_paid = np.bincount(buyers[purchases], weights=prices[purchases], minlength=n_households)
        has_purchase = np.bincount(buyers[purchases], minlength=n_households) > 0
        share = sold_collateral_share(
            pre_sale_owner=inputs.property_owner,
            pre_sale_value=inputs.property_value,
            completed_sales=sales.loc[kept],
            n_households=n_households,
        )
        repayment = np.minimum(share * principal_left, proceeds)
        rejected = (
            np.zeros(n_households, dtype=bool)
            if mortgage_rejected is None
            else np.asarray(mortgage_rejected(repayment), dtype=bool)
        )
        mortgage_zeroed = (granted > 0.0) & (rejected | ~has_purchase)
        mortgage = np.where(mortgage_zeroed, 0.0, granted)
        net_need = price_paid - mortgage - (proceeds - repayment)
        completes = net_need <= free_lfa + free_ifa
        still_kept = move_chain_feasible(sales, kept & (is_self_sale | completes[buyers]), inputs.property_inhabitant)
        if still_kept.sum() == kept.sum():
            break
        kept = still_kept

    purchase_liquidation = np.maximum(net_need - free_lfa, 0.0)
    return PurchaseCompletion(
        kept_sales=kept | ~is_sell,
        mortgage_zeroed=mortgage_zeroed,
        sale_proceeds=proceeds,
        mortgage_sale_repayment=repayment,
        net_housing_cash_need=net_need,
        purchase_liquidation=purchase_liquidation,
        housing_lfa_draw=np.maximum(net_need - purchase_liquidation, 0.0),
    )
