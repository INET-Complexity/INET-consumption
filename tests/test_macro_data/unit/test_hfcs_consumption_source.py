import hashlib

import numpy as np
import pandas as pd
import pytest

from macro_data.util.hfcs_consumption_source import (
    HFCSConsumptionRegistration,
    read_hfcs_consumption_source,
    source_ids_fingerprint,
)


def source_frame():
    frame = pd.DataFrame(
        {
            "source_id": ["001", "002", "003", "004"],
            "wave": [2014] * 4,
            "c_y": [0.01, 0.6, 0.9, np.nan],
            "survey_weight": [1.0, 9.0, 1.0, 1.0],
            "hard_valid": [True, True, True, False],
            "threshold_pool": [True] * 4,
        }
    )
    frame.attrs.update(
        consumption_concept="total",
        income_concept="net_of_interest_plus_imputed_rent",
        income_scale_var="disp_geom_avg_income",
        preparation_manifest_sha256="a" * 64,
        preparation_stage="before_percentile_trimming",
    )
    return frame


def registration():
    # Registered from the original full fixture, independently of any altered table.
    return HFCSConsumptionRegistration(
        source_count=4,
        source_ids_sha256=source_ids_fingerprint(source_frame().source_id.to_numpy()),
        preparation_manifest_sha256="a" * 64,
    )


def save_read(tmp_path, frame, expected=None):
    path = tmp_path / "private_ratios.parquet"
    frame.to_parquet(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return read_hfcs_consumption_source(
        path, wave=2014, expected_sha256=digest, registration=expected or registration()
    )


def test_roundtrip_links_replacement_draws_and_preserves_source_ids(tmp_path):
    source = save_read(tmp_path, source_frame())
    ratios = source.ratios_for(np.array(["003", "001", "003", "002"]))
    np.testing.assert_allclose(ratios, source.winsorised.ratios[[2, 0, 2, 1]])
    assert len(ratios) == 4
    ratios[0] = -999
    assert source.winsorised.ratios[2] >= 0


def test_unknown_and_invalid_sources_never_fall_back_to_legacy(tmp_path):
    source = save_read(tmp_path, source_frame())
    with pytest.raises(ValueError, match="no HFCS source match"):
        source.ratios_for(np.array(["missing"]))
    with pytest.raises(ValueError, match="hard-invalid"):
        source.ratios_for(np.array(["004"]))
    with pytest.raises(ValueError, match="stable strings"):
        source.ratios_for(np.array([1]))


def test_source_hash_mismatch_is_rejected_before_parquet_loading(tmp_path):
    path = tmp_path / "bad.parquet"
    path.write_bytes(b"wrong data")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_hfcs_consumption_source(path, wave=2014, expected_sha256="0" * 64, registration=registration())


@pytest.mark.parametrize("change", ["duplicate", "missing_id", "missing_column", "mask", "concept", "stage", "wave"])
def test_rejects_inconsistent_source_contract(tmp_path, change):
    frame = source_frame()
    if change == "duplicate":
        frame.loc[1, "source_id"] = "001"
    elif change == "missing_id":
        frame.loc[1, "source_id"] = None
    elif change == "missing_column":
        frame = frame.drop(columns="c_y")
    elif change == "mask":
        frame["hard_valid"] = frame.hard_valid.astype(int)
    elif change == "concept":
        frame.attrs["income_concept"] = "gross"
    elif change == "stage":
        frame.attrs["preparation_stage"] = "after_trimming"
    else:
        frame["wave"] = 2017
    with pytest.raises(ValueError):
        save_read(tmp_path, frame)


def test_other_waves_do_not_change_source_thresholds(tmp_path):
    frame = source_frame()
    other = frame.copy()
    other["wave"] = 2017
    other["c_y"] = 100
    combined = pd.concat([other, frame], ignore_index=True)
    combined.attrs = frame.attrs.copy()
    reference = save_read(tmp_path, frame)
    result = save_read(tmp_path, combined)
    np.testing.assert_allclose(result.winsorised.ratios, reference.winsorised.ratios)
    assert result.winsorised.lower == reference.winsorised.lower
    assert result.winsorised.upper == reference.winsorised.upper


@pytest.mark.parametrize("change", ["truncate", "substitute_id", "income_scale", "missing_income_scale", "manifest"])
def test_rejects_review_findings_even_with_valid_candidate_file_hash(tmp_path, change):
    frame = source_frame()
    if change == "truncate":
        frame = frame.iloc[:2].copy()
        message = "source count"
    elif change == "substitute_id":
        frame.loc[0, "source_id"] = "replacement"
        message = "ID set"
    elif change == "income_scale":
        frame.attrs["income_scale_var"] = "perm_income"
        message = "income_scale_var"
    elif change == "missing_income_scale":
        del frame.attrs["income_scale_var"]
        message = "income_scale_var"
    else:
        frame.attrs["preparation_manifest_sha256"] = "b" * 64
        message = "preparation_manifest_sha256"
    with pytest.raises(ValueError, match=message):
        save_read(tmp_path, frame)


def test_registration_is_order_independent_and_supplies_quantile_limits(tmp_path):
    from dataclasses import replace

    from macro_data.util.consumption_initialisation import winsorise_hfcs_consumption_ratios

    frame = source_frame().iloc[::-1].copy()
    expected = replace(registration(), lower_quantile=0.25, upper_quantile=0.75)
    source = save_read(tmp_path, frame, expected=expected)
    manual = winsorise_hfcs_consumption_ratios(
        c_y=frame.c_y.to_numpy(),
        survey_weights=frame.survey_weight.to_numpy(),
        hard_valid=frame.hard_valid.to_numpy(),
        threshold_pool=frame.threshold_pool.to_numpy(),
        lower_quantile=0.25,
        upper_quantile=0.75,
    )
    assert source.winsorised.lower == manual.lower
    assert source.winsorised.upper == manual.upper
