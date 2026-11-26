"""MPC owner-financing labels must follow live debt, not historical tenure flags."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "run_model"))
from src.mpc_analysis import make_household_metadata, reconcile_owner_mortgage_tenure
from src.mpc_paper_groups import prepare_paper_groups


def test_owner_financing_refresh_keeps_renters_and_other_tenures():
    source = pd.Series(["owner_outright", "owner_mortgage", "renter", "owner_other", "unknown"], index=[9, 3, 7, 1, 5])
    original = source.copy()
    result = reconcile_owner_mortgage_tenure(source, np.array([100.0, 0.0, 200.0, 300.0, 0.0]))
    assert result.tolist() == ["owner_mortgage", "owner_outright", "renter", "owner_other", "unknown"]
    assert result.index.equals(source.index)
    pd.testing.assert_series_equal(source, original)


@pytest.mark.parametrize("debt", [np.nan, np.inf, -1.0])
def test_owner_financing_requires_valid_mortgage_balance(debt):
    with pytest.raises(ValueError, match="mortgage balances"):
        reconcile_owner_mortgage_tenure(pd.Series(["owner_outright"]), [debt])


def test_saved_panel_groups_refresh_owner_financing_without_reclassifying_landlord():
    panel = pd.DataFrame(
        {"housing_tenure": ["owner_outright", "owner_mortgage", "renter"], "mortgage_debt": [100.0, 0.0, 200.0]}
    )
    grouped, _ = prepare_paper_groups(panel)
    assert grouped.tenure.tolist() == ["Mortgagors", "Outright\nowners", "Renters"]
    assert panel.housing_tenure.tolist() == ["owner_outright", "owner_mortgage", "renter"]


def test_saved_owner_panel_cannot_guess_when_mortgage_balances_are_missing():
    with pytest.raises(ValueError, match="requires mortgage_debt"):
        prepare_paper_groups(pd.DataFrame({"housing_tenure": ["owner_outright"]}))


def test_new_metadata_refreshes_labels_but_preserves_model_tenure_codes(monkeypatch):
    import src.mpc_analysis as analysis

    values = {
        "n_households": 2,
        "wealth_main_residence": np.array([100.0, 200.0]),
        "wealth_other_properties": np.zeros(2),
        "liquid_financial_assets": np.array([10.0, 20.0]),
        "illiquid_financial_assets": np.zeros(2),
        "consumption_loan_debt": np.zeros(2),
        "mortgage_debt": np.array([50.0, 0.0]),
        "net_wealth": np.array([60.0, 220.0]),
        "income": np.array([10.0, 20.0]),
    }
    codes = np.array([1, 2])
    households = SimpleNamespace(
        ts=SimpleNamespace(current=lambda name: values[name]),
        states={"Tenure Status of the Main Residence": codes},
        functions={},
    )
    monkeypatch.setattr(analysis, "_household_head_age_proxy", lambda *_args, **_kwargs: np.array([40.0, 50.0]))
    metadata = make_household_metadata(SimpleNamespace(households=households), seed=1, pre_shock_row=60)
    assert metadata.housing_tenure.tolist() == ["owner_mortgage", "owner_outright"]
    assert metadata.housing_tenure_recorded.tolist() == ["owner_outright", "owner_mortgage"]
    np.testing.assert_array_equal(metadata.tenure_code, codes)
    np.testing.assert_array_equal(codes, [1, 2])
