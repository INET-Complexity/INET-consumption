"""Nominal cash-flow rates: annual units, debt coverage and first-borrowing guard."""

import numpy as np
import pytest

from macromodel.agents.households.cashflow_rates import (
    DebtServiceRateChanges,
    applied_debt_service_rate_changes,
)


def test_annual_simple_nominal_changes_cover_both_debt_types():
    _, reference = applied_debt_service_rate_changes(
        [0.01, 0],
        [0.02, 0.03],
        [100, 0],
        [200, 300],
    )
    changes, current = applied_debt_service_rate_changes(
        [0.0125, 0.04],
        [0.025, 0.02],
        [100, 50],
        [200, 300],
        reference=reference,
    )
    np.testing.assert_allclose(changes.consumer, [0.01, 0])  # first borrowing has no rate reset
    np.testing.assert_allclose(changes.mortgage, [0.02, -0.04])
    np.testing.assert_allclose(current["mortgage_rate_annual"], [0.10, 0.08])
    np.testing.assert_allclose(reference["mortgage_rate_annual"], [0.08, 0.12])


def test_fixed_rates_and_debt_quantity_changes_do_not_create_rate_change():
    _, reference = applied_debt_service_rate_changes([0.01], [0.02], [100], [200])
    changes, _ = applied_debt_service_rate_changes([0.01], [0.02], [50], [800], reference=reference)
    np.testing.assert_array_equal(changes.consumer, [0])
    np.testing.assert_array_equal(changes.mortgage, [0])


def test_initial_observation_has_no_invented_rate_shock():
    changes, _ = applied_debt_service_rate_changes([0.1], [0.2], [100], [200])
    np.testing.assert_array_equal(changes.consumer, [0])
    np.testing.assert_array_equal(changes.mortgage, [0])


@pytest.mark.parametrize("consumer,mortgage", [([np.nan], [0]), ([0], [np.inf]), ([0], [0, 1])])
def test_rate_change_record_rejects_invalid_vectors(consumer, mortgage):
    with pytest.raises(ValueError):
        DebtServiceRateChanges(consumer, mortgage)


def test_rate_change_record_copies_and_freezes_inputs():
    values = np.array([0.01])
    record = DebtServiceRateChanges(values, [0])
    values[0] = 99
    assert record.consumer[0] == 0.01
    with pytest.raises(ValueError):
        record.consumer[0] = 0


@pytest.mark.parametrize("annualization", [0, -1, np.nan])
def test_invalid_annualization_rejected(annualization):
    with pytest.raises(ValueError, match="annualization"):
        applied_debt_service_rate_changes([0], [0], [0], [0], annualization=annualization)
