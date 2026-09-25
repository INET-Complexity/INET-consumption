"""Purchase-price MPC outcomes and conservative legacy-file reconstruction."""

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "run_model"))
from src.mpc_analysis import build_household_mpc_panel, read_household_consumption_outcomes  # noqa: E402


def write_file(path, *, shock=False, persisted=False, cpi=1.0, missing=()):
    goods = np.full((4, 2), 10.0)
    if shock:
        goods[1:] += [2.0, 0.0]  # Second household has a goods floor/rationing.
    rent = np.tile([4.0, 0.0], (4, 1))
    housing = np.tile([0.0, 8.0], (4, 1))
    target = np.full((4, 2), 30.0)
    if shock:
        target[1:] += 5.0
    with h5py.File(path, "w") as handle:
        group = handle.create_group("FRA/households")
        fields = dict(
            consumption=goods,
            rent=rent,
            rent_imputed=housing,
            total_consumption=1.2 * goods.sum(axis=1, keepdims=True),
            total_consumption_before_vat=goods.sum(axis=1, keepdims=True),
            target_consumption=goods,
            target_consumption_columns=[[0, 0], [1, 0]],
            target_consumption_calibrated_total=target,
            target_consumption_total_mpc=np.full((4, 2), 0.5),
        )
        if persisted:
            fields["consumption_cash_expenditure"] = 1.2 * goods + rent
            fields["consumption_including_housing"] = 1.2 * goods + rent + housing
        for name, value in fields.items():
            if name not in missing:
                group.create_dataset(name, data=value)
        handle.create_dataset("FRA/economy/cpi_fixed_basket", data=np.full((4, 1), cpi))


def panel_for(tmp_path, *, persisted=False, shock_cpi=1.0, missing=()):
    baseline, shock = tmp_path / "baseline.h5", tmp_path / "shock.h5"
    write_file(baseline, persisted=persisted, missing=missing)
    write_file(shock, shock=True, persisted=persisted, cpi=shock_cpi, missing=missing)
    return build_household_mpc_panel(
        baseline_h5=baseline,
        shock_h5=shock,
        metadata=pd.DataFrame({"household_id": [1, 0], "income": [100.0, 100.0]}),
        country_code="FRA",
        shock_row=1,
        horizon_periods=3,
        shock_fraction=0.1,
        cumulative_mpc_column="cmpc_3p",
        real_cumulative_mpc_column="real_cmpc_3p",
    )


@pytest.mark.parametrize("persisted", [False, True])
def test_nominal_cash_total_identity_and_gross_target(tmp_path, persisted):
    panel = panel_for(tmp_path, persisted=persisted)
    np.testing.assert_allclose(panel["mpc_impact"], [0.2, 0.0])  # Legacy net goods.
    np.testing.assert_allclose(panel["cash_mpc_impact"], [0.24, 0.0])
    np.testing.assert_allclose(panel["cash_mpc_impact"], panel["total_mpc_impact"])
    np.testing.assert_allclose(panel["cash_cmpc_3p"], [0.72, 0.0])
    np.testing.assert_allclose(panel["total_cmpc_2p"], [0.48, 0.0])
    np.testing.assert_allclose(panel["behavioural_total_target_mpc_impact"], [0.5, 0.5])
    np.testing.assert_allclose(panel["baseline_target_consumption_total_mpc"], [0.5, 0.5])
    np.testing.assert_allclose(panel["behavioural_total_target_cmpc_3p"], [1.5, 1.5])


def test_own_cpi_explains_real_cash_total_difference(tmp_path):
    panel = panel_for(tmp_path, shock_cpi=2.0)
    # Nominal H is unchanged; deflated owner services change by 8/2 - 8.
    np.testing.assert_allclose(panel["total_mpc_impact"], panel["cash_mpc_impact"])
    np.testing.assert_allclose(panel["imputed_rent_real_mpc_impact"], [0.0, -0.8])
    for suffix in ["mpc_impact", "cmpc_2p", "cmpc_3p"]:
        np.testing.assert_allclose(
            panel[f"total_real_{suffix}"] - panel[f"cash_real_{suffix}"],
            panel[f"imputed_rent_real_{suffix}"],
            atol=1e-12,
        )


@pytest.mark.parametrize("missing", [("rent",), ("total_consumption",), ("total_consumption_before_vat",)])
def test_insufficient_legacy_data_unavailable_without_changing_old_columns(tmp_path, missing):
    panel = panel_for(tmp_path, missing=missing)
    np.testing.assert_allclose(panel["mpc_impact"], [0.2, 0.0])
    assert panel["cash_mpc_impact"].isna().all()
    assert panel["total_mpc_impact"].isna().all()


def test_missing_imputed_rent_is_not_zero(tmp_path):
    panel = panel_for(tmp_path, missing=("rent_imputed",))
    np.testing.assert_allclose(panel["cash_mpc_impact"], [0.24, 0.0])
    assert panel["total_mpc_impact"].isna().all()


def test_persisted_values_take_precedence_over_missing_legacy_inputs(tmp_path):
    panel = panel_for(tmp_path, persisted=True, missing=("rent", "rent_imputed", "total_consumption"))
    np.testing.assert_allclose(panel["cash_mpc_impact"], [0.24, 0.0])
    np.testing.assert_allclose(panel["total_mpc_impact"], [0.24, 0.0])


@pytest.mark.parametrize("marker", [False, True])
def test_old_non_cacf_zero_target_is_unavailable(tmp_path, marker):
    path = tmp_path / "old.h5"
    write_file(path, missing=() if marker else ("target_consumption_total_mpc",))
    with h5py.File(path, "a") as handle:
        group = handle["FRA/households"]
        group["target_consumption_calibrated_total"][:] = 0.0
        if marker:
            group["target_consumption_total_mpc"][:] = np.nan
    result = read_household_consumption_outcomes(path, "FRA")
    assert np.isnan(result["behavioural_total_target"]).all()


@pytest.mark.parametrize("field", ["rent", "rent_imputed", "consumption_cash_expenditure"])
def test_reader_rejects_wrong_household_shape(tmp_path, field):
    path = tmp_path / "bad.h5"
    write_file(path, persisted=True)
    with h5py.File(path, "a") as handle:
        group = handle["FRA/households"]
        del group[field]
        group.create_dataset(field, data=np.ones((4, 1)))
    with pytest.raises(ValueError, match=field):
        read_household_consumption_outcomes(path, "FRA")


def test_total_target_survives_unavailable_derivative(tmp_path):
    path = tmp_path / "target.h5"
    write_file(path)
    with h5py.File(path, "a") as handle:
        handle["FRA/households/target_consumption_total_mpc"][:] = np.nan
    result = read_household_consumption_outcomes(path, "FRA")
    np.testing.assert_allclose(result["behavioural_total_target"], 30.0)


def test_changed_imputed_rent_is_exact_nominal_bridge(tmp_path):
    baseline, shock = tmp_path / "baseline.h5", tmp_path / "shock.h5"
    write_file(baseline)
    write_file(shock, shock=True)
    with h5py.File(shock, "a") as handle:
        handle["FRA/households/rent_imputed"][1:, 1] = 11.0
    panel = build_household_mpc_panel(
        baseline_h5=baseline,
        shock_h5=shock,
        metadata=pd.DataFrame({"household_id": [0, 1], "income": [100.0, 100.0]}),
        country_code="FRA",
        shock_row=1,
        horizon_periods=3,
        shock_fraction=0.1,
    )
    np.testing.assert_allclose(panel["total_mpc_impact"] - panel["cash_mpc_impact"], [0.0, 0.3])
    np.testing.assert_allclose(panel["total_cmpc_4q"] - panel["cash_cmpc_4q"], [0.0, 0.9])


def test_legacy_all_missing_diagnostics_preserves_results(tmp_path):
    panel = panel_for(
        tmp_path,
        missing=(
            "rent",
            "rent_imputed",
            "total_consumption",
            "total_consumption_before_vat",
            "target_consumption_calibrated_total",
            "target_consumption_total_mpc",
        ),
    )
    np.testing.assert_allclose(panel["mpc_impact"], [0.2, 0.0])
    np.testing.assert_allclose(panel["target_mpc_impact"], [0.2, 0.0])
    assert panel["behavioural_total_target_mpc_impact"].isna().all()
    assert panel["cash_mpc_impact"].isna().all()
    assert panel["total_mpc_impact"].isna().all()


def test_zero_vat_legacy_reconstruction(tmp_path):
    path = tmp_path / "no_vat.h5"
    write_file(path)
    with h5py.File(path, "a") as handle:
        group = handle["FRA/households"]
        group["total_consumption"][:] = group["total_consumption_before_vat"][:]
    result = read_household_consumption_outcomes(path, "FRA")
    np.testing.assert_allclose(result["cash"], np.tile([14.0, 10.0], (4, 1)))
    np.testing.assert_allclose(result["total"], np.tile([14.0, 18.0], (4, 1)))
