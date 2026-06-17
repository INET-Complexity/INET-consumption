"""Loader for the FR portfolio-choice GTP-FRM coefficient table (Stage 4, Increment 1)."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

# Documented scale-convention divisors from notes.scale_convention in the source JSON:
# "x = model_gross_income / (S * 10_000)" and "x = model_net_wealth / (S * 100_000)".
# These strings are documentation, not a machine-readable spec, so the divisors are
# hardcoded here per that documented convention.
INCOME_SCALE_DIVISOR = 10_000.0
NET_WEALTH_SCALE_DIVISOR = 100_000.0


@dataclass(frozen=True)
class FRMCoefficients:
    """Typed view of the estimated GTP-FRM portfolio-choice coefficients."""

    participation_coefficients: dict[str, float]
    magnitude_coefficients: dict[str, float]
    phi_1: float
    lambda_kappa: float
    phi_2: float
    income_scale_divisor: float
    net_wealth_scale_divisor: float


def _load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise ValueError(f"FRM coefficients file not found: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"FRM coefficients file is not valid JSON: {path}") from exc


def _extract_coefficient_values(payload: dict, equation_key: str) -> dict[str, float]:
    if equation_key not in payload:
        raise ValueError(f"FRM coefficients payload is missing the '{equation_key}' equation.")

    equation = payload[equation_key]
    coefficients = equation.get("coefficients")
    if not coefficients:
        raise ValueError(f"FRM '{equation_key}' equation must include a non-empty 'coefficients' dict.")

    values: dict[str, float] = {}
    for name, entry in coefficients.items():
        try:
            value = float(entry["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"FRM '{equation_key}' coefficient '{name}' is missing a numeric 'value'.") from exc
        if not math.isfinite(value):
            raise ValueError(f"FRM '{equation_key}' coefficient '{name}' must be finite, got {value}.")
        values[name] = value

    return values


def _parse_phi_2_derived(text: str) -> float:
    """Extract the trailing numeric literal from the documented phi_2_derived formula string."""
    tail = text.rsplit("=", 1)[-1].strip()
    try:
        return float(tail)
    except ValueError as exc:
        raise ValueError(f"Could not parse a numeric value from phi_2_derived: {text!r}") from exc


def _validate_adjustment_parameters(payload: dict) -> tuple[float, float]:
    adjustment_parameters = payload.get("adjustment_parameters")
    if not adjustment_parameters:
        raise ValueError("FRM coefficients payload is missing 'adjustment_parameters'.")

    if "phi_1" not in adjustment_parameters or "lambda_kappa" not in adjustment_parameters:
        raise ValueError("FRM 'adjustment_parameters' must include both 'phi_1' and 'lambda_kappa'.")

    try:
        phi_1 = float(adjustment_parameters["phi_1"])
        lambda_kappa = float(adjustment_parameters["lambda_kappa"])
    except (TypeError, ValueError) as exc:
        raise ValueError("FRM 'phi_1' and 'lambda_kappa' must be numeric.") from exc

    if not math.isfinite(phi_1):
        raise ValueError(f"FRM 'phi_1' must be finite, got {phi_1}.")
    if not math.isfinite(lambda_kappa):
        raise ValueError(f"FRM 'lambda_kappa' must be finite, got {lambda_kappa}.")
    if not (0.0 < lambda_kappa <= 1.0):
        raise ValueError(f"FRM 'lambda_kappa' must be in (0, 1], got {lambda_kappa}.")

    computed_phi_2 = phi_1 * (1.0 - lambda_kappa) / lambda_kappa

    phi_2_derived_text = adjustment_parameters.get("phi_2_derived")
    if isinstance(phi_2_derived_text, str):
        # A present-but-unparseable phi_2_derived string is a data error, not an
        # absent field: raise here rather than silently falling back to the weaker
        # finite/positive-only check below, which would mask a real inconsistency.
        parsed_phi_2 = _parse_phi_2_derived(phi_2_derived_text)
        if abs(computed_phi_2 - parsed_phi_2) >= 1e-6:
            raise ValueError(
                "Computed phi_2 (phi_1 * (1 - lambda_kappa) / lambda_kappa) = "
                f"{computed_phi_2} does not match the documented phi_2_derived value "
                f"{parsed_phi_2} within tolerance."
            )
    else:
        if not math.isfinite(computed_phi_2) or computed_phi_2 <= 0:
            raise ValueError(f"Computed phi_2 must be a positive finite number, got {computed_phi_2}.")

    return phi_1, lambda_kappa


def load_frm_coefficients(path: str | Path) -> FRMCoefficients:
    """Load and validate the FR portfolio-choice GTP-FRM coefficient table.

    Returns a typed FRMCoefficients structure with provenance metadata
    (standard errors, significance stars, notebook variable names, units)
    stripped out, keeping only the numeric values needed at runtime.
    """
    payload = _load_json(path)

    participation_coefficients = _extract_coefficient_values(payload, "participation")
    magnitude_coefficients = _extract_coefficient_values(payload, "magnitude")
    phi_1, lambda_kappa = _validate_adjustment_parameters(payload)
    phi_2 = phi_1 * (1.0 - lambda_kappa) / lambda_kappa

    return FRMCoefficients(
        participation_coefficients=participation_coefficients,
        magnitude_coefficients=magnitude_coefficients,
        phi_1=phi_1,
        lambda_kappa=lambda_kappa,
        phi_2=phi_2,
        income_scale_divisor=INCOME_SCALE_DIVISOR,
        net_wealth_scale_divisor=NET_WEALTH_SCALE_DIVISOR,
    )
