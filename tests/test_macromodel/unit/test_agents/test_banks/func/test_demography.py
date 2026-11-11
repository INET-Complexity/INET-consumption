import numpy as np
import pytest

from macromodel.agents.banks.func.demography import DefaultBankDemography, NoBankDemography


class TestBankDemography:
    def test__handle_bank_insolvency(self):
        NoBankDemography(solvency_ratio=0.05).handle_bank_insolvency(
            current_bank_equity=np.array([100.0, 100.0]),
            current_bank_deposits=np.array([50.0, 50.0]),
            current_bank_loans=np.array([10.0, 20.0]),
            is_insolvent=np.array([False, False]),
        )


def _insolvent(demography, *, equity, loans, reserves):
    flags = np.zeros(len(equity), dtype=bool)
    demography.handle_bank_insolvency(
        current_bank_equity=np.array(equity, dtype=float),
        current_bank_loans=np.array(loans, dtype=float),
        current_bank_deposits=np.array(reserves, dtype=float),
        is_insolvent=flags,
    )
    return flags.tolist()


def test_default_demography_counts_reserves_in_full_by_default():
    # Seed-1 trace at quarter 78: equity 215, loans 690, reserves 3,588 -> ratio 0.050.
    demography = DefaultBankDemography(solvency_ratio=0.05)
    assert _insolvent(demography, equity=[213.0, 215.0], loans=[690.0, 690.0], reserves=[3588.0, 3588.0]) == [
        True,
        False,
    ]


def test_zero_reserve_weight_tests_equity_against_loans_only():
    demography = DefaultBankDemography(solvency_ratio=0.05, reserve_weight=0.0)
    # 0.05 x 690 = 34.5: reserves of any size do not matter.
    assert _insolvent(demography, equity=[213.0, 34.0], loans=[690.0, 690.0], reserves=[3588.0, 0.0]) == [False, True]


def test_bank_without_weighted_assets_is_insolvent_only_with_negative_equity():
    for demography in (
        DefaultBankDemography(solvency_ratio=0.05),
        DefaultBankDemography(solvency_ratio=0.05, reserve_weight=0.0),
    ):
        assert _insolvent(demography, equity=[10.0, -1.0], loans=[0.0, 0.0], reserves=[0.0, 0.0]) == [False, True]


def test_reserve_weight_outside_unit_interval_raises():
    with pytest.raises(ValueError, match="reserve_weight"):
        DefaultBankDemography(solvency_ratio=0.05, reserve_weight=1.5)
