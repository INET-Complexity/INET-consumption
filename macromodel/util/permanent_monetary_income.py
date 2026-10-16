"""Conditional monetary-income moments; property receipts stay fixed in real terms.

The learned log multiplier applies only to employment/transfers. Housing
services, borrower interest, capital gains and financing proceeds are not inputs.
The existing common macro forecast is added separately by the caller.
"""

from functools import lru_cache

import numpy as np
from scipy.special import ndtr


@lru_cache(maxsize=5)
def _quadrature(order):
    return np.polynomial.legendre.leggauss(order)


def _log_income(log_non_property, property_income, log_floor):
    """Stable log of a floored sum, including signed net property receipts."""
    positive = property_income >= 0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        added = np.logaddexp(log_non_property, np.log(np.maximum(property_income, 0)))
        log_negative = np.log(np.maximum(-property_income, 0))
        difference = log_non_property + np.log1p(-np.exp(log_negative - log_non_property))
    result = np.where(positive, added, np.where(log_non_property > log_negative, difference, -np.inf))
    return np.maximum(result, log_floor)


def monetary_income_moments(*, non_property, property_income, income_floor, log_mean, log_variance):
    """Return E[log(Mp/M)] and its variance under a Gaussian earnings belief.

    Mp=max(max(N,0)*exp(X)+P,S), M=max(max(N,0)+P,S).
    The positive floor is an evaluation convention, never a cash receipt.
    Gaussian tails beyond twelve standard deviations are negligible at model
    precision. Split integration at the exact floor threshold; refine until
    both moments agree to 1e-8 absolute plus 1e-6 relative tolerance. Failure
    raises rather than silently installing unconverged consumption inputs.
    """
    inputs = [np.asarray(v, dtype=float) for v in (non_property, property_income, income_floor, log_mean, log_variance)]
    n, p, floor, mu, variance = inputs
    if n.ndim != 1 or any(v.shape != n.shape or not np.all(np.isfinite(v)) for v in inputs):
        raise ValueError("Monetary bridge requires aligned finite household vectors.")
    if np.any(floor <= 0) or np.any(variance < 0):
        raise ValueError("Monetary bridge requires a positive income floor and nonnegative variance.")
    n = np.maximum(n, 0.0)
    with np.errstate(divide="ignore"):
        log_n = np.log(n)
    log_floor = np.log(floor)
    log_current = _log_income(log_n, p, log_floor)
    mean = _log_income(log_n + mu, p, log_floor) - log_current
    result_variance = np.zeros_like(mean)
    probability = ((_log_income(log_n + mu, p, log_floor)) <= log_floor).astype(float)
    errors = np.zeros_like(mean)
    active = (n > 0) & (variance > 0)
    indices = np.flatnonzero(active)
    if indices.size:
        ln, prop, lf, baseline = log_n[indices], p[indices], log_floor[indices], log_current[indices]
        m, sd = mu[indices], np.sqrt(variance[indices])
        threshold = np.full(indices.size, -np.inf)
        can_bind = floor[indices] > prop
        binding_property = prop[can_bind]
        binding_floor = floor[indices][can_bind]
        # Form log(S-P) without overflowing when finite S and negative P
        # are both near the largest representable currency magnitude.
        with np.errstate(divide="ignore", over="ignore"):
            log_gap = np.where(
                binding_property < 0,
                np.logaddexp(lf[can_bind], np.log(np.maximum(-binding_property, 0))),
                np.log(binding_floor - np.maximum(binding_property, 0)),
            )
        threshold[can_bind] = (log_gap - ln[can_bind] - m[can_bind]) / sd[can_bind]
        probability[indices] = ndtr(threshold)
        lower = np.minimum(np.maximum(threshold, -12.0), 12.0)
        left_mass = ndtr(threshold)
        offset = lf - baseline
        previous = None
        converged = np.zeros(indices.size, dtype=bool)
        for order in (32, 64, 128, 256, 512):
            nodes, weights = _quadrature(order)
            half = (12.0 - lower) / 2
            z = lower[:, None] + half[:, None] * (nodes + 1)
            w = half[:, None] * weights * np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)
            g = _log_income(ln[:, None] + m[:, None] + sd[:, None] * z, prop[:, None], lf[:, None]) - baseline[:, None]
            mass = left_mass + w.sum(axis=1)
            current_mean = (left_mass * offset + (w * g).sum(axis=1)) / mass
            current_variance = (
                left_mass * (offset - current_mean) ** 2 + (w * (g - current_mean[:, None]) ** 2).sum(axis=1)
            ) / mass
            if previous is not None:
                difference = np.maximum(np.abs(current_mean - previous[0]), np.abs(current_variance - previous[1]))
                stable = (np.abs(current_mean - previous[0]) <= 1e-8 + 1e-6 * np.abs(current_mean)) & (
                    np.abs(current_variance - previous[1]) <= 1e-8 + 1e-6 * current_variance
                )
                newly = stable & ~converged
                mean[indices[newly]] = current_mean[newly]
                result_variance[indices[newly]] = current_variance[newly]
                errors[indices[newly]] = difference[newly]
                converged |= stable
                if converged.all():
                    break
            previous = current_mean, current_variance
        if not converged.all():
            raise ValueError("Monetary-income moment integration did not converge.")
    with np.errstate(over="ignore"):
        current = np.exp(log_current)
    if any(not np.all(np.isfinite(value)) for value in (mean, result_variance, current)):
        raise ValueError("Monetary-income moments and current income must be finite.")
    return {
        "mean": mean,
        "variance": result_variance,
        "current": current,
        "property": p.copy(),
        "floor_used": ((n + p) < floor).astype(float),
        "component_floor_used": (inputs[0] < 0).astype(float),
        "projected_floor_probability": probability,
        "integration_error": errors,
    }
