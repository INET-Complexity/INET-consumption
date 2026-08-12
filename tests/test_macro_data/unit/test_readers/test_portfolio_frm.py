import json
from pathlib import Path

import pytest

from macro_data.readers.portfolio_frm import FRMCoefficients, load_frm_coefficients

REPO_ROOT = Path(__file__).resolve().parents[4]
REAL_FRM_PATH = next(
    path
    for path in (
        REPO_ROOT / "data" / "raw_data" / "portfolio" / "FR_portfolio_frm_coefficients.json",
        REPO_ROOT
        / "run_model"
        / "data"
        / "raw_data"
        / "portfolio"
        / "FR_portfolio_frm_coefficients.json",
    )
    if path.is_file()
)


def _base_payload() -> dict:
    return {
        "participation": {
            "coefficients": {
                "constant": {"value": 0.0851, "se": 0.1518, "sig": "", "notebook_var": "const", "units": "intercept"},
                "windfall_income": {
                    "value": 0.2079,
                    "se": 0.0905,
                    "sig": "**",
                    "notebook_var": "windfall_income",
                    "units": "binary",
                },
            }
        },
        "magnitude": {
            "coefficients": {
                "constant": {"value": 0.0418, "se": 0.1098, "sig": "", "notebook_var": "const", "units": "intercept"},
                "net_wealth": {
                    "value": 0.0345,
                    "se": 0.0037,
                    "sig": "***",
                    "notebook_var": "net_wealth",
                    "units": "EUR / 100000",
                },
            }
        },
        "adjustment_parameters": {
            "lambda_kappa": 0.1,
            "phi_1": 5.0,
            "phi_2_derived": "phi_1 * (1 - lambda_kappa) / lambda_kappa = 45.0",
        },
        "notes": {
            "scale_convention": {
                "gross_household_income": "x = model_gross_income / (S * 10_000)",
                "net_wealth": "x = model_net_wealth / (S * 100_000)",
            }
        },
    }


def _write(tmp_path: Path, payload: dict, name: str = "frm.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def test__load_frm_coefficients_real_file_happy_path():
    result = load_frm_coefficients(REAL_FRM_PATH)

    assert isinstance(result, FRMCoefficients)
    assert result.phi_1 == pytest.approx(5.0)
    assert result.lambda_kappa == pytest.approx(0.1)
    assert result.phi_2 == pytest.approx(45.0)
    assert result.income_scale_divisor == pytest.approx(10_000.0)
    assert result.net_wealth_scale_divisor == pytest.approx(100_000.0)
    assert result.participation_coefficients["windfall_income"] == pytest.approx(0.2079)
    assert result.magnitude_coefficients["net_wealth"] == pytest.approx(0.0345)
    # provenance metadata must be dropped, only the plain float values remain
    assert all(isinstance(v, float) for v in result.participation_coefficients.values())
    assert all(isinstance(v, float) for v in result.magnitude_coefficients.values())


def test__load_frm_coefficients_missing_file_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"
    with pytest.raises(ValueError, match="not found"):
        load_frm_coefficients(missing_path)


def test__load_frm_coefficients_malformed_json_raises(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not valid json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_frm_coefficients(path)


def test__load_frm_coefficients_missing_participation_key_raises(tmp_path):
    payload = _base_payload()
    del payload["participation"]
    path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="participation"):
        load_frm_coefficients(path)


def test__load_frm_coefficients_missing_magnitude_key_raises(tmp_path):
    payload = _base_payload()
    del payload["magnitude"]
    path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="magnitude"):
        load_frm_coefficients(path)


def test__load_frm_coefficients_empty_coefficients_dict_raises(tmp_path):
    payload = _base_payload()
    payload["participation"]["coefficients"] = {}
    path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="non-empty"):
        load_frm_coefficients(path)


def test__load_frm_coefficients_non_finite_value_raises(tmp_path):
    payload = _base_payload()
    payload["participation"]["coefficients"]["constant"]["value"] = float("nan")
    path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="finite"):
        load_frm_coefficients(path)


def test__load_frm_coefficients_lambda_kappa_zero_raises(tmp_path):
    payload = _base_payload()
    payload["adjustment_parameters"]["lambda_kappa"] = 0.0
    path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        load_frm_coefficients(path)


def test__load_frm_coefficients_lambda_kappa_negative_raises(tmp_path):
    payload = _base_payload()
    payload["adjustment_parameters"]["lambda_kappa"] = -0.5
    path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        load_frm_coefficients(path)


def test__load_frm_coefficients_phi_2_consistency_check_raises_on_mismatch(tmp_path):
    payload = _base_payload()
    # phi_1=5.0, lambda_kappa=0.1 computes to 45.0, but the documented string says 99.0
    payload["adjustment_parameters"]["phi_2_derived"] = "phi_1 * (1 - lambda_kappa) / lambda_kappa = 99.0"
    path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="does not match"):
        load_frm_coefficients(path)


def test__load_frm_coefficients_malformed_phi_2_derived_string_raises(tmp_path):
    payload = _base_payload()
    # Present but unparseable: must raise, not silently skip the consistency check.
    payload["adjustment_parameters"]["phi_2_derived"] = "see notebook cell 12"
    path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="Could not parse"):
        load_frm_coefficients(path)
