"""Exposure the bank capital requirement applies to."""

import numpy as np


def capital_requirement_exposure(banks) -> np.ndarray:
    """Loans plus household and firm overdrafts, by bank, each with weight one.

    The model holds equity against unweighted loans with a single capital
    adequacy ratio. Overdrafts are retail credit like consumer loans (Basel
    CRE20.65-68 treats them as regulatory retail), so they enter with the same
    weight. ``total_outstanding_loans`` stays loans only: deposits and market
    shares already add overdrafts to it separately.
    """
    ts = banks.ts
    return (
        np.asarray(ts.current("total_outstanding_loans"), dtype=float)
        + np.asarray(ts.current("overdrafts_to_households"), dtype=float)
        + np.asarray(ts.current("overdrafts_to_firms"), dtype=float)
    )
