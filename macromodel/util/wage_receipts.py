"""Shared nominal employee receipts and labour-tax bridge.

Stored employee earnings are net of the model's employee labour deductions.
Firm labour cost additionally includes employer contributions.
"""

import numpy as np


def wage_price_factor(cpi: float, elasticity: float = 1.0) -> float:
    if not np.isfinite(cpi) or cpi <= 0 or not np.isfinite(elasticity):
        raise ValueError("Wage CPI must be positive and finite; elasticity must be finite.")
    return float(cpi**elasticity)


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
