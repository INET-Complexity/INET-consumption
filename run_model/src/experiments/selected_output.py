"""Experiment output selection without serializing the complete simulation."""

import h5py
import numpy as np


def select_names(requested, available):
    """Validate an explicit, nonempty selection before running simulations."""
    names = tuple(available) if requested is None else tuple(requested)
    if isinstance(requested, str) or not names or len(set(names)) != len(names):
        raise ValueError("output_variables must be a nonempty sequence of unique names.")
    unknown = set(names) - set(available)
    if unknown:
        raise ValueError(f"Unknown output_variables: {sorted(unknown)}; choose from {list(available)}")
    return names


def write_selected(model, path, country_code, paths):
    """Serialize only named analysis inputs, preserving existing HDF5 layouts."""
    country = model.countries[country_code]
    with h5py.File(path, "w") as handle:
        for dataset_path in dict.fromkeys(paths):
            code, agent, name = dataset_path.split("/")
            attribute = {"central_Bank": "central_bank", "CM": "credit_market"}.get(agent, agent)
            ts = getattr(country, attribute).ts
            if name not in ts.dicts:
                continue
            group = handle.require_group(f"{code}/{agent}")
            ts.write_field_to_h5(np.asarray(ts.historic(name)), name, group)
