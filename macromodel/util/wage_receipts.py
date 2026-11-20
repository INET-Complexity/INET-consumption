"""Shared nominal employee receipts and labour-tax bridge.

Stored employee earnings are net of the model's employee labour deductions.
Firm labour cost additionally includes employer contributions.
"""

import numpy as np


def wage_price_factor(cpi: float, elasticity: float = 1.0, anchor_factor: float = 1.0) -> float:
    """Payroll price factor ``cpi**elasticity * anchor_factor``.

    ``anchor_factor`` is 1.0 unless the wage inflation anchor is on (see
    :func:`wage_anchor_factor`); multiplying by exactly 1.0 leaves the unanchored
    factor bit-identical.
    """
    if not np.isfinite(cpi) or cpi <= 0 or not np.isfinite(elasticity):
        raise ValueError("Wage CPI must be positive and finite; elasticity must be finite.")
    if not np.isfinite(anchor_factor) or anchor_factor <= 0:
        raise ValueError("Wage anchor factor must be positive and finite.")
    return float(cpi**elasticity * anchor_factor)


def wage_anchor_factor(
    target_annual_inflation: float,
    elasticity: float,
    elapsed_periods: int,
    periods_per_year: int,
) -> float:
    """Level factor that anchors wage indexation to the inflation target.

    With ``elasticity`` = psi the payroll factor becomes ``CPI**psi * (1 + pi*)**((1 - psi) * n / ppy)``,
    so its growth per period is ``psi * pi_{t-1} + (1 - psi) * pi*`` (pi* converted from the annual
    target). At psi = 1 the factor is 1.0 and the anchor has no effect.
    """
    if not np.isfinite(target_annual_inflation) or target_annual_inflation <= -1.0:
        raise ValueError("Annual inflation target must be finite and above -100%.")
    if not np.isfinite(elasticity) or elapsed_periods < 0 or periods_per_year <= 0:
        raise ValueError("Elasticity must be finite; elapsed periods >= 0; periods per year > 0.")
    return float((1.0 + target_annual_inflation) ** ((1.0 - elasticity) * elapsed_periods / periods_per_year))


def labour_income_components(net_receipts, income_tax, employee_si, employer_si):
    """Return gross wages, employee deductions and total employer cost."""
    factor = (1.0 - employee_si) * (1.0 - income_tax)
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Employee deductions must leave a positive net-income factor.")
    gross = np.asarray(net_receipts, dtype=float) / factor
    return {
        "gross_wages": gross,
        "income_tax": gross * (1.0 - employee_si) * income_tax,
        "employee_si": gross * employee_si,
        "employer_si": gross * employer_si,
        "employer_cost": gross * (1.0 + employer_si),
    }
