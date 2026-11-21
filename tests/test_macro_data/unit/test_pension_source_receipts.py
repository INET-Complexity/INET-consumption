import numpy as np
import pandas as pd

from macro_data.processing.synthetic_population.hfcs_synthetic_population import SyntheticHFCSPopulation


def test_observed_pension_without_retiree_keeps_household_share():
    p = SyntheticHFCSPopulation.__new__(SyntheticHFCSPopulation)
    p.household_data = pd.DataFrame({"Corresponding Individuals ID": [[0, 1], [2], [3]]})
    p.individual_data = pd.DataFrame(
        {
            "Age": [40, 10, 70, 45],
            "Is Retired": [False, False, True, False],
            "Individual Public Pension Income": [30.0, 0.0, 60.0, 0.0],
        }
    )
    p._set_individual_public_pension_weights(pd.Series([30.0, 60.0, 0.0]), 90.0)
    np.testing.assert_allclose(p.individual_data["Public Pension Benefits"], [30.0, 0.0, 60.0, 0.0])
    np.testing.assert_array_equal(p.individual_data["Is Retired"], [False, False, True, False])
    assert p.individual_data["Public Pension Benefits"].sum() == 90.0


def test_minors_only_source_recipient_is_not_dropped():
    p = SyntheticHFCSPopulation.__new__(SyntheticHFCSPopulation)
    p.household_data = pd.DataFrame({"Corresponding Individuals ID": [[0, 1]]})
    p.individual_data = pd.DataFrame(
        {"Age": [10, 12], "Is Retired": [False, False], "Individual Public Pension Income": [5.0, 15.0]}
    )
    p._set_individual_public_pension_weights(pd.Series([20.0]), 20.0)
    np.testing.assert_allclose(p.individual_data["Public Pension Benefits"], [5.0, 15.0])


def test_runtime_pays_source_recipient_without_retired_status():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from macromodel.country.country import Country

    country = Country.__new__(Country)
    country.individuals = SimpleNamespace(
        states={
            "Public Pension Weight": np.array([1.0]),
            "Is Retired": np.array([False]),
            "Corresponding Household ID": np.array([0]),
        }
    )
    distribute = Mock(return_value=np.array([30.0]))
    country.central_government = SimpleNamespace(
        distribute_public_pension_benefits_to_individuals=distribute,
        ts=SimpleNamespace(current=lambda name: [0.0]),
    )
    country.economy = SimpleNamespace(current_consumer_price_level=lambda: 1.0)
    country.households = SimpleNamespace(
        aggregate_individual_amount=lambda individual_amount, corr_households: individual_amount,
        compute_social_transfer_income=lambda **kwargs: np.zeros(1),
    )
    country._household_unemployment_benefits_from_individuals = lambda: np.zeros(1)
    country.compute_realised_stage5_subsistence_support = lambda: np.zeros(1)
    country.compute_realised_minimum_income = lambda: np.zeros(1)
    pensions, *_ = country.compute_realised_household_social_income_components()
    np.testing.assert_array_equal(distribute.call_args.kwargs["retirement_eligibility"], [True])
    np.testing.assert_allclose(pensions, [30.0])


def test_actual_recipient_wins_over_retired_household_member():
    p = SyntheticHFCSPopulation.__new__(SyntheticHFCSPopulation)
    p.household_data = pd.DataFrame({"Corresponding Individuals ID": [[0, 1, 2]]})
    p.individual_data = pd.DataFrame(
        {
            "Is Retired": [True, False, False],
            "Individual Public Pension Income": [0.0, 20.0, 80.0],
        }
    )
    p._set_individual_public_pension_weights(pd.Series([100.0]), 50.0)
    np.testing.assert_allclose(p.individual_data["Public Pension Benefits"], [0.0, 10.0, 40.0])


def test_positive_household_receipts_require_individual_recipient():
    import pytest

    p = SyntheticHFCSPopulation.__new__(SyntheticHFCSPopulation)
    p.household_data = pd.DataFrame({"Corresponding Individuals ID": [[0]]})
    p.individual_data = pd.DataFrame({"Individual Public Pension Income": [0.0]})
    with pytest.raises(ValueError, match="matching observed individual recipients"):
        p._set_individual_public_pension_weights(pd.Series([30.0]), 30.0)


def test_missing_individual_receipts_rejected():
    import pytest

    p = SyntheticHFCSPopulation.__new__(SyntheticHFCSPopulation)
    p.individual_data = pd.DataFrame({"Is Retired": [True]})
    with pytest.raises(ValueError, match="PG0310"):
        p._set_individual_public_pension_weights(pd.Series([30.0]), 30.0)


def test_retirement_cutoff_and_source_preservation(monkeypatch):
    import macro_data.processing.synthetic_population.hfcs_individual_tools as t

    people = pd.DataFrame(
        {
            "Age": [64, 65, 70, 70, 50, 10],
            "Gender": [1] * 6,
            "Education": [3] * 6,
            "Labour Status": [1, 1, 3, 1, 5, 4],
            "Individual Public Pension Income": [10.0, 20.0, 30.0, 0.0, 40.0, np.nan],
            "Employee Income": [1.0] * 6,
            "Corresponding Household ID": range(6),
            "Relation to Reference Person": [1] * 6,
        }
    )
    for name in (
        "fill_missing_gender",
        "fill_individual_age",
        "fill_individual_education",
        "fill_individual_labour_status",
    ):
        monkeypatch.setattr(t, name, lambda data: data)
    monkeypatch.setattr(t, "fill_individual_nace", lambda data, *a, **kw: data.assign(**{"Employment Industry": 0}))
    monkeypatch.setattr(t, "fill_individual_employee_income", lambda data, **kw: data)
    monkeypatch.setattr(
        t,
        "set_individual_unemployed_income",
        lambda data, **kw: data.assign(**{"Income from Unemployment Benefits": 0.0}),
    )
    # Targets match the proposed classification, so no macro adjustment is needed.
    out = t.process_individual_data(people, ["A"], 1, 0.0, 0.0, 0.4, [1])
    np.testing.assert_array_equal(out["HFCS Labour Status"], [1, 1, 3, 1, 5, 4])
    np.testing.assert_array_equal(out["Is Retired"], [False, True, True, False, True, False])
    np.testing.assert_array_equal(out["Activity Status"], [1, 3, 3, 1, 3, 3])
    np.testing.assert_allclose(out["Individual Public Pension Income"], [10, 20, 30, 0, 40, np.nan])


def test_participation_calibration_cannot_reactivate_retired_recipients():
    from macro_data.processing.synthetic_population.hfcs_individual_tools import increase_participation_rate

    people = pd.DataFrame(
        {
            "Age": [65, 40],
            "Is Retired": [True, False],
            "Activity Status": [3, 3],
            "Employment Industry": [np.nan] * 2,
            "Employee Income": [0.0] * 2,
            "Income from Unemployment Benefits": [0.0] * 2,
            "Income": [0.0] * 2,
        }
    )
    increase_participation_rate(people, 1.0)
    np.testing.assert_array_equal(people["Activity Status"], [3, 2])


def test_reader_keeps_individual_pension_amounts_and_converts_currency(tmp_path):
    from types import SimpleNamespace

    from macro_data.readers.population_data.hfcs_reader import HFCSReader

    path = tmp_path / "P1.csv"
    pd.DataFrame({"ID": [1, 2], "SA0100": ["FR", "FR"], "PG0310": [100.0, "M"], "PG0410": [20.0, "A"]}).to_csv(
        path, index=False
    )
    reader = HFCSReader.read_csv(path, "FRA", "FR", 2014, SimpleNamespace(from_eur_to_lcu=lambda **kw: 2.0))
    np.testing.assert_allclose(reader["Individual Public Pension Income"], [200.0, np.nan])
    np.testing.assert_allclose(reader["Individual Occupational and Private Pension Income"], [40.0, np.nan])


def test_household_shares_preserved_with_individual_source_shares():
    p = SyntheticHFCSPopulation.__new__(SyntheticHFCSPopulation)
    p.household_data = pd.DataFrame({"Corresponding Individuals ID": [[0, 1], [2]]})
    p.individual_data = pd.DataFrame({"Individual Public Pension Income": [10.0, 30.0, 60.0]})
    p._set_individual_public_pension_weights(pd.Series([80.0, 120.0]), 100.0)
    np.testing.assert_allclose(p.individual_data["Public Pension Benefits"], [10.0, 30.0, 60.0])


def test_younger_pension_recipients_keep_all_three_activity_categories():
    from macro_data.processing.synthetic_population.hfcs_individual_tools import set_individual_activity_status

    people = pd.DataFrame(
        {
            "Age": [64, 64, 64, 65, 10],
            "Labour Status": [1, 3, 9, 1, 4],
            "Is Retired": [False, False, False, True, False],
            "Individual Public Pension Income": [10.0, 20.0, 30.0, 40.0, 0.0],
        }
    )
    out = set_individual_activity_status(people, unemployment_rate=0.5, participation_rate=0.5)
    np.testing.assert_array_equal(out["Activity Status"], [1, 2, 3, 3, 3])


def test_negative_household_nonreceipt_matches_existing_household_allocator():
    p = SyntheticHFCSPopulation.__new__(SyntheticHFCSPopulation)
    p.household_data = pd.DataFrame({"Corresponding Individuals ID": [[0], [1]]})
    p.individual_data = pd.DataFrame({"Individual Public Pension Income": [np.nan, 20.0]})
    p._set_individual_public_pension_weights(pd.Series([-1.0, 20.0]), 40.0)
    np.testing.assert_allclose(p.individual_data["Public Pension Benefits"], [0.0, 40.0])
