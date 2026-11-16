"""Apply the registered HFCS opening budget after runtime income/tax setup."""

import numpy as np

from macro_data.util.consumption_initialisation import allocate_hfcs_ratio_consumption


def initialise_hfcs_consumption(country, source_ratios):
    """Replace opening expenditure without changing income, assets or transfers.

    The raw aligned income includes the model's net-interest/imputed-rent
    bridge, before its planning-income floor and smoothing. Aggregate goods
    already include housing inputs; no separate housing-goods amount is added.
    Infeasible household inputs fail rather than invoking a new support rule.
    """
    hh = country.households
    if "expected_behavioural_income_raw" not in hh.ts.dicts:
        raise ValueError("HFCS initialisation requires the aligned model-income definition.")
    if any(len(rows) != 1 for rows in hh.ts.dicts.values()):
        raise ValueError("HFCS initialisation can only run on an unadvanced opening state.")
    income = hh.ts.current("expected_behavioural_income_raw")
    vat = country.central_government.states["Value-added Tax"]
    sectors = np.asarray(hh.ts.current("industry_consumption"), dtype=float)
    price = country.economy.initial_consumer_price_level()
    if sectors.ndim != 1 or not np.all(np.isfinite(sectors)) or np.any(sectors < 0):
        raise ValueError("Opening sector goods must be a finite nonnegative vector.")
    if not np.isfinite(price) or price <= 0:
        raise ValueError("Opening consumer price level must be finite and positive.")
    result = allocate_hfcs_ratio_consumption(
        hfcs_c_y=source_ratios,
        model_income=income,
        cash_rent=hh.ts.current("rent"),
        imputed_rent=hh.ts.current("rent_imputed"),
        vat=vat,
        aggregate_net_goods=float(sectors.sum()),
    )
    # A common opening sector basket preserves both the aggregate sector totals
    # and the new household totals. Dynamic household preferences are unchanged.
    weights = sectors / sectors.sum() if sectors.sum() > 0 else np.zeros_like(sectors)
    matrix = np.outer(result.goods, weights)
    cash = (1 + vat) * result.goods + hh.ts.current("rent")
    replacements = {
        "consumption": result.goods,
        "consumption_cash_expenditure": cash,
        "consumption_including_housing": result.total,
        "cacf_real_consumption_budget": result.total / price,
        "target_consumption": matrix,
        "realised_household_expenditure": result.goods + hh.ts.current("investment").sum(axis=1),
    }
    for name, values in replacements.items():
        hh.ts.override_current(name, values)
    for name, values in {
        "initial_hfcs_source_c_y": np.asarray(source_ratios, dtype=float).copy(),
        "initial_hfcs_comparable_income": np.asarray(income, dtype=float).copy(),
        "initial_hfcs_candidate_total": result.candidate_total,
        "initial_hfcs_final_c_y": result.total / income,
        "initial_hfcs_goods_scale": np.full(len(result.goods), result.goods_scale),
    }.items():
        hh.ts[name] = values
    # National-accounts goods and VAT aggregates are unchanged by redistribution.
    return result
