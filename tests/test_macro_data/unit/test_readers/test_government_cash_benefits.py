import numpy as np
import pandas as pd
import pytest

import macro_data.processing.synthetic_population.hfcs_synthetic_population as hfcs_population_module
from macro_data.processing.synthetic_population.hfcs_synthetic_population import SyntheticHFCSPopulation

# Eurostat gov_10a_exp, S13 D62, France 2014, mio EUR
GF10, GF1002, GF1003, GF1005 = 421183.0, 267984.1, 33667.3, 35209.2


class TestGovernmentCashBenefits:
    def test__cofog_values(self, readers):
        assert readers.eurostat.government_cash_social_benefits("FRA", 2014, "GF10") == pytest.approx(GF10 * 1e6)
        assert readers.eurostat.government_cash_social_benefits("FRA", 2014, "GF1005") == pytest.approx(GF1005 * 1e6)

    def test__no_cross_country_fallback(self, readers):
        assert readers.eurostat.government_cash_social_benefits("USA", 2014, "GF10") is None

    def test__envelope_is_s13_social_protection_minus_unemployment(self, readers):
        # The unemployment argument is ignored when COFOG data exist.
        envelope = readers.get_non_unemployment_benefits_lcu("FRA", 2014, 4.0, unemployment_benefits_lcu=1e12)
        assert envelope == pytest.approx((GF10 - GF1005) * 1e6 / 4.0)

    def test__all_benefits_prefer_government_over_household_d62(self, readers):
        assert readers.get_total_benefits_lcu("FRA", 2014, 4.0) == pytest.approx(GF10 * 1e6 / 4.0)

    def test__public_pension_share(self, readers):
        expected = (GF1002 + GF1003) / (GF10 - GF1005)
        assert readers.get_public_pension_share("FRA", 2014) == pytest.approx(expected)

    def test__public_pension_share_is_none_without_cofog(self, readers):
        assert readers.get_public_pension_share("USA", 2014) is None


def _population(monkeypatch):
    population = SyntheticHFCSPopulation.__new__(SyntheticHFCSPopulation)
    population.scale = 10
    population.yearly_factor = 4.0
    population.social_transfers_model = object()
    population.household_data = pd.DataFrame(
        {
            "Type": [1, 1],
            "Net Wealth": [0.0, 0.0],
            "Income": [1.0, 1.0],
            "Debt": [0.0, 0.0],
            "Income from Pensions": [13.0, 24.0],
            "Pension Income": [13.0, 24.0],
            "Public Pension Income": [10.0, 20.0],
            "Regular Social Transfers": [1.0, 3.0],
            "Social Transfer Income": [1.0, 3.0],
        }
    )
    monkeypatch.setattr(hfcs_population_module, "apply_iterative_imputer", lambda data, *_a, **_k: data)
    monkeypatch.setattr(hfcs_population_module, "fit_linear", lambda **_kwargs: None)
    return population


def test__national_accounts_share_sets_totals_and_hfcs_allocates(monkeypatch):
    population = _population(monkeypatch)
    population.set_household_social_transfers(total_social_transfers=100.0, public_pension_share=0.75)

    pensions = population.household_data["Allocated Public Pension Benefits"]
    other = population.household_data["Allocated Other Social Transfers"]
    assert pensions.sum() == pytest.approx(75.0)
    assert other.sum() == pytest.approx(25.0)
    np.testing.assert_allclose(pensions, [25.0, 50.0])  # HFCS weights 10:20
    np.testing.assert_allclose(other, [6.25, 18.75])  # HFCS weights 1:3


@pytest.mark.parametrize("share", [None, np.nan])
def test__missing_share_falls_back_to_hfcs_composition(monkeypatch, share):
    population = _population(monkeypatch)
    population.set_household_social_transfers(total_social_transfers=100.0, public_pension_share=share)
    assert population.household_data["Allocated Public Pension Benefits"].sum() == pytest.approx(100.0 * 30.0 / 34.0)
