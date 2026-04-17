from __future__ import annotations

from copy import deepcopy
import numpy as np
import pandas as pd

from macromodel.configurations import CountryConfiguration


def _default_substitution_bundles(n_industries: int) -> list[int]:
    return list(range(n_industries))


def align_country_configuration_to_data(
    country_cfg: CountryConfiguration,
    n_industries: int,
) -> CountryConfiguration:
    """
    Align industry-dimensioned parts of a CountryConfiguration to the data shape.

    The goal is to preserve YAML-loaded behavioral settings and only repair
    fields whose lengths must match the model dimension implied by the data.
    """
    cfg = deepcopy(country_cfg)

    if len(cfg.firms.parameters.capital_inputs_delay) != n_industries:
        cfg.firms.parameters.capital_inputs_delay = [0] * n_industries

    if len(cfg.firms.parameters.depreciation_rates) != n_industries:
        cfg.firms.parameters.depreciation_rates = [0.0] * n_industries

    if len(cfg.firms.substitution_bundles) != n_industries:
        cfg.firms.substitution_bundles = _default_substitution_bundles(n_industries)

    if len(cfg.households.substitution_bundles) != n_industries:
        cfg.households.substitution_bundles = _default_substitution_bundles(n_industries)

    planner_params = cfg.firms.functions.productivity_investment_planner.parameters
    if planner_params.get("n_firms") != n_industries:
        planner_params["n_firms"] = n_industries

    return cfg


def unpack_cell(x):
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return x.item()
        return x.tolist()
    if isinstance(x, list) and len(x) == 1:
        return x[0]
    return x
