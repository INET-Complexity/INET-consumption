import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

RUN_MODEL_PATH = Path(__file__).resolve().parents[2] / "run_model"
if str(RUN_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL_PATH))

from src.diagnostics.household_investment_counterfactual import (  # noqa: E402
    _period_total,
    reconcile_investment_counterfactual,
)


def _write_series(path: Path, values: np.ndarray) -> h5py.File:
    h5_file = h5py.File(path, "w")
    h5_file.create_dataset("FRA/households/series", data=values)
    return h5_file


# Every series reconcile_investment_counterfactual reads at period 1, held at
# zero across both periods unless overridden below.
_ZERO_SERIES = (
    "total_investment_before_vat",
    "total_investment",
    "total_received_consumption_loans",
    "total_received_mortgages",
    "realised_household_expenditure",
    "interest_paid",
    "debt_installments",
    "price_paid_for_property",
    "portfolio_settlement_committed_adjustment_cost",
    "income_for_residual_saving",
    "total_liquid_financial_assets",
    "total_illiquid_financial_assets",
    "total_illiquid_financial_asset_capital_gains",
    "stage5_cash_ledger_residual",
)


def _write_reconciliation_fixture(
    path: Path,
    *,
    liquidation_planned_period1: float,
    forced_liquidation_amount_period1: float,
) -> None:
    """Write a minimal one-household, two-period fixture (periods 0 and 1)."""
    with h5py.File(path, "w") as h5_file:
        for name in _ZERO_SERIES:
            h5_file.create_dataset(f"FRA/households/{name}", data=np.array([[0.0], [0.0]]))
        h5_file.create_dataset(
            "FRA/households/liquidation_planned",
            data=np.array([[0.0], [liquidation_planned_period1]]),
        )
        # Decoy: the stale carrier the fix stopped reading. A different value
        # here must not move the result.
        h5_file.create_dataset(
            "FRA/households/forced_liquidation_amount",
            data=np.array([[0.0], [forced_liquidation_amount_period1]]),
        )


def test__period_total_returns_finite_household_panel_total(tmp_path: Path) -> None:
    h5_file = _write_series(tmp_path / "series.h5", np.array([[1.0, 2.0], [3.0, 4.0]]))
    try:
        assert _period_total(h5_file, "FRA", "households/series", 1) == 7.0
    finally:
        h5_file.close()


def test__period_total_rejects_missing_series_and_invalid_period(tmp_path: Path) -> None:
    h5_file = _write_series(tmp_path / "series.h5", np.array([[1.0, 2.0]]))
    try:
        with pytest.raises(KeyError, match="Missing HDF5 series"):
            _period_total(h5_file, "FRA", "households/missing", 0)
        with pytest.raises(IndexError, match="Period 1 is unavailable"):
            _period_total(h5_file, "FRA", "households/series", 1)
    finally:
        h5_file.close()


def test__period_total_rejects_non_finite_values(tmp_path: Path) -> None:
    h5_file = _write_series(tmp_path / "series.h5", np.array([[1.0, np.nan]]))
    try:
        with pytest.raises(ValueError, match="Non-finite values"):
            _period_total(h5_file, "FRA", "households/series", 0)
    finally:
        h5_file.close()


def test_reconcile_investment_counterfactual_reads_liquidation_planned_not_forced_liquidation_amount(
    tmp_path: Path,
) -> None:
    """`reduced_asset_liquidation` must track `liquidation_planned`, the
    authoritative Stage 5 post-grant-settled carrier -- not the stale
    Increment 3 shadow estimate `forced_liquidation_amount`, which never
    mutates live household balances.
    """
    baseline_path = tmp_path / "baseline.h5"
    off_path = tmp_path / "off.h5"
    _write_reconciliation_fixture(
        baseline_path,
        liquidation_planned_period1=100.0,
        forced_liquidation_amount_period1=999.0,
    )
    _write_reconciliation_fixture(
        off_path,
        liquidation_planned_period1=40.0,
        forced_liquidation_amount_period1=1.0,
    )

    result = reconcile_investment_counterfactual(baseline_path, off_path, country="FRA", period=1)

    assert result["reduced_asset_liquidation"] == pytest.approx(60.0)
    assert result["liquidation_internal_reclassification"] == pytest.approx(60.0)
    # Pins the regression: the old forced_liquidation_amount-based delta
    # (999.0 - 1.0 = 998.0) must not leak through.
    assert result["reduced_asset_liquidation"] != pytest.approx(998.0)
