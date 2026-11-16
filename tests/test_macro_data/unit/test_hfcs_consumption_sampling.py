import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macro_data.processing.synthetic_population.hfcs_synthetic_population import (
    RESTRICT_COLS,
    SyntheticHFCSPopulation,
    eligible_hfcs_consumption_households,
    sample_households,
)
from macro_data.readers.population_data.hfcs_reader import HFCSReader
from macro_data.util.hfcs_consumption_source import (
    HFCSConsumptionRegistration,
    read_hfcs_consumption_source,
    source_ids_fingerprint,
)


@pytest.fixture
def population(tmp_path):
    ids = np.array(["001", "002", "003"])
    frame = pd.DataFrame(
        dict(
            source_id=ids,
            wave=2014,
            c_y=[0.1, 0.9, np.nan],
            survey_weight=[1.0, 3.0, 100.0],
            hard_valid=[True, True, False],
            threshold_pool=True,
        )
    )
    frame.attrs.update(
        consumption_concept="total",
        income_concept="net_of_interest_plus_imputed_rent",
        income_scale_var="disp_geom_avg_income",
        preparation_stage="before_percentile_trimming",
        preparation_manifest_sha256="a" * 64,
    )
    path = tmp_path / "private.parquet"
    frame.to_parquet(path, index=False)
    source = read_hfcs_consumption_source(
        path,
        wave=2014,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        registration=HFCSConsumptionRegistration(3, source_ids_fingerprint(ids), "a" * 64),
    )
    households = pd.DataFrame({"HFCS Source ID": ids, "Weight": [1.0, 3.0, 100.0]}, index=[10, 20, 30])
    individuals = pd.DataFrame({"Corresponding Household ID": [10, 20, 20, 30], "Age": [40, 50, 10, 60]})
    return households, individuals, source


def test_new_sampling_filters_before_drawing_and_uses_weights_once(population, monkeypatch):
    households, individuals, source = population
    original = households.copy(deep=True)

    def choice(indices, n, p, replace):
        np.testing.assert_array_equal(indices, [10, 20])
        np.testing.assert_allclose(p, [0.25, 0.75])
        assert n == 4 and replace
        return np.array([20, 10, 20, 20])

    monkeypatch.setattr(np.random, "choice", choice)
    hh, people = sample_households(households, individuals, 4, consumption_source=source)
    assert len(hh) == 4 and people["New Household ID"].nunique() == 4
    assert hh["Corresponding Individuals ID"].map(len).tolist() == [2, 1, 2, 2]
    assert hh["HFCS Source ID"].tolist() == ["002", "001", "002", "002"]
    np.testing.assert_allclose(hh["HFCS Consumption Ratio"], source.ratios_for(hh["HFCS Source ID"].to_numpy()))
    pd.testing.assert_frame_equal(households, original)
    for household_id, row in hh.iterrows():
        assert (people.loc[row["Corresponding Individuals ID"], "Corresponding Household ID"] == household_id).all()


def test_legacy_sampling_keeps_original_eligibility_and_rng_path(population, monkeypatch):
    households, individuals, _ = population

    def choice(indices, n, p, replace):
        np.testing.assert_array_equal(indices, [10, 20, 30])
        np.testing.assert_allclose(p, [1 / 104, 3 / 104, 100 / 104])
        return np.full(n, 30)

    monkeypatch.setattr(np.random, "choice", choice)
    hh, _ = sample_households(households, individuals, 5)
    assert len(hh) == 5 and (hh["HFCS Source ID"] == "003").all()
    assert "HFCS Consumption Ratio" not in hh


@pytest.mark.parametrize(
    "defect", ["missing_id", "different_id", "duplicate_id", "duplicate_index", "no_members", "weights"]
)
def test_source_mismatches_fail_before_sampling(population, defect):
    hh, people, source = population
    if defect == "missing_id":
        hh = hh.drop(columns="HFCS Source ID")
    elif defect == "different_id":
        hh.loc[10, "HFCS Source ID"] = "unknown"
    elif defect == "duplicate_id":
        hh.loc[10, "HFCS Source ID"] = "002"
    elif defect == "duplicate_index":
        hh.index = [10, 10, 30]
    elif defect == "no_members":
        people = people[people["Corresponding Household ID"] != 20]
    else:
        hh.loc[10, "Weight"] = 99
    with pytest.raises(ValueError):
        eligible_hfcs_consumption_households(hh, people, source)


def test_restrict_preserves_ratio_diagnostics_without_requiring_them_for_legacy(population):
    _, _, source = population
    for new in [False, True]:
        obj = SyntheticHFCSPopulation.__new__(SyntheticHFCSPopulation)
        obj.household_data = pd.DataFrame({c: [0.0] for c in RESTRICT_COLS})
        if new:
            obj.household_data["HFCS Source ID"] = "001"
            obj.household_data["HFCS Consumption Ratio"] = source.ratios_for(np.array(["001"]))
        obj.restrict()
        assert ("HFCS Consumption Ratio" in obj.household_data) == new


def test_reader_preserves_stable_ids_and_leading_zeros(tmp_path):
    path = tmp_path / "H1.csv"
    path.write_text("ID,SA0010,SA0100,HW0010\n10,001,FR,1\n20,002,FR,3\n")
    rates = SimpleNamespace(from_eur_to_lcu=lambda **_: 1.0)
    frame = HFCSReader.read_csv(path, "FRA", "FR", 2014, rates)
    assert frame["HFCS Source ID"].tolist() == ["001", "002"]


def test_reader_legacy_extract_without_source_id_still_loads(tmp_path):
    path = tmp_path / "H1.csv"
    path.write_text("ID,SA0100,HW0010\n10,FR,1\n")
    rates = SimpleNamespace(from_eur_to_lcu=lambda **_: 1.0)
    frame = HFCSReader.read_csv(path, "FRA", "FR", 2014, rates)
    assert "HFCS Source ID" not in frame


def test_from_readers_forwards_registered_source_and_rejects_wrong_wave(population, monkeypatch):
    from dataclasses import replace

    import macro_data.processing.synthetic_population.hfcs_synthetic_population as module

    hh, people, source = population
    readers = SimpleNamespace(
        eurostat=SimpleNamespace(number_of_households=lambda *_: 100),
        hfcs={"FRA": SimpleNamespace(households_df=hh, individuals_df=people)},
    )
    args = dict(
        readers=readers,
        country_name="FRA",
        country_name_short="FR",
        scale=10,
        year=2014,
        quarter=1,
        industry_data={},
        industries=["custom"],
        total_unemployment_benefits=0,
        exogenous_data=None,
    )
    with pytest.raises(ValueError, match="wave must match"):
        SyntheticHFCSPopulation.from_readers(**args, consumption_source=replace(source, wave=2017))

    def sampling(households, individuals, n, **kwargs):
        assert households is hh and individuals is people and n == 10
        assert kwargs["consumption_source"] is source
        raise RuntimeError("reached sampling boundary")

    monkeypatch.setattr(module, "sample_households", sampling)
    with pytest.raises(RuntimeError, match="reached sampling boundary"):
        SyntheticHFCSPopulation.from_readers(**args, consumption_source=source)


def test_reader_join_keeps_household_source_identity(monkeypatch):
    def read_csv(path, **kwargs):
        if path.stem.startswith("H"):
            return pd.DataFrame({"Weight": [1.0], "HFCS Source ID": ["001"]}, index=[10])
        if path.stem.startswith("D"):
            return pd.DataFrame({"Weight": [1.0], "HFCS Source ID": ["derived-id"]}, index=[10])
        return pd.DataFrame({"Corresponding Household ID": [10]}, index=[100])

    monkeypatch.setattr(HFCSReader, "read_csv", read_csv)
    from pathlib import Path

    reader = HFCSReader.from_csv("FRA", "FR", 2014, Path("/unused"), None, num_surveys=1)
    assert reader.households_df["HFCS Source ID"].tolist() == ["001"]
