from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from macro_data.configuration import DataConfiguration, split_country_configs
from macro_data.configuration.dataconfiguration import HouseholdConsumptionInitialisationConfiguration
from macro_data.data_wrapper import DataWrapper
from macro_data.readers.default_readers import DataReaders
from macro_data.util.consumption_source_config import (
    CONSUMPTION_DATA_STAGE,
    consumption_cache_requires_rebuild,
    consumption_data_signature,
    load_configured_consumption_sources,
    preserve_consumption_cache,
    require_completed_consumption_allocation,
)
from macro_data.util.consumption_source_export import export_consumption_source
from macro_data.util.hfcs_consumption_source import HFCSConsumptionRegistration, source_ids_fingerprint


@pytest.fixture
def configured(tmp_path):
    ids = np.array(["001", "002", "003"])
    frame = pd.DataFrame(
        dict(
            source_id=ids,
            wave=2014,
            c_y=[0.5, 0.9, np.nan],
            survey_weight=[1.0, 2.0, 3.0],
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
    registration = HFCSConsumptionRegistration(3, source_ids_fingerprint(ids), "a" * 64)
    fields = export_consumption_source(frame, registration, tmp_path / "hfcs/source.parquet", wave=2014)
    for name in ["source_file", "registration_file"]:
        fields[name] = str(Path(fields[name]).relative_to(tmp_path))
    cfgdict = yaml.safe_load(
        (Path(__file__).resolve().parents[3] / "run_model/config/data_config_FRA.yaml").read_text()
    )
    cfgdict["country_configs"]["FRA"]["household_consumption_initialisation"] = fields
    cfgdict["country_configs"] = split_country_configs(cfgdict["country_configs"])
    return DataConfiguration(**cfgdict), tmp_path


def test_configured_source_resolves_relative_paths_and_first_implicate(configured):
    config, root = configured
    source = load_configured_consumption_sources(config, root)["FRA"]
    assert source.wave == 2014 and source.winsorised.valid.sum() == 2
    signature = consumption_data_signature(config, root)
    assert signature["sources"]["FRA"]["implicate"] == 1
    assert signature["stage"] == CONSUMPTION_DATA_STAGE
    assert signature["sources"]["FRA"]["source"]["path"] == str(root / "hfcs/source.parquet")


@pytest.mark.parametrize("field", ["source_file", "registration_file"])
def test_source_or_registration_modification_cannot_reuse_cache(configured, field):
    config, root = configured
    cache = SimpleNamespace(household_consumption_signature=consumption_data_signature(config, root))
    c = config.country_configs["FRA"].household_consumption_initialisation
    with (root / getattr(c, field)).open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        consumption_cache_requires_rebuild(cache, config, root)


def test_both_direction_cache_switching_old_metadata_and_stage_changes(configured):
    config, root = configured
    signature = consumption_data_signature(config, root)
    assert consumption_cache_requires_rebuild(SimpleNamespace(), config, root)
    cache = SimpleNamespace(household_consumption_signature=signature)
    assert not consumption_cache_requires_rebuild(cache, config, root)
    assert consumption_cache_requires_rebuild(cache, config, root, single_hfcs_survey=True)
    old = SimpleNamespace(household_consumption_signature={**signature, "stage": "older-stage"})
    assert consumption_cache_requires_rebuild(old, config, root)
    config.country_configs["FRA"].household_consumption_initialisation.method = "legacy"
    legacy_signature = consumption_data_signature(config, root)
    assert legacy_signature["stage"] == "legacy_first_implicate_individual_pensions_retirement_v3"
    assert consumption_cache_requires_rebuild(cache, config, root)
    assert consumption_cache_requires_rebuild(SimpleNamespace(), config, root)
    assert not consumption_cache_requires_rebuild(
        SimpleNamespace(household_consumption_signature=legacy_signature), config, root
    )
    config.country_configs["FRA"].hfcs_implicate = None
    assert not consumption_cache_requires_rebuild(SimpleNamespace(), config, root)


def test_config_rejects_missing_inputs_and_other_implicates():
    with pytest.raises(ValidationError):
        HouseholdConsumptionInitialisationConfiguration(method="hfcs_cy")
    with pytest.raises(ValidationError):
        HouseholdConsumptionInitialisationConfiguration(implicate=5)


def test_preserve_cache_is_exact_and_reusable(tmp_path):
    path = tmp_path / "data.pkl"
    path.write_bytes(b"legacy data")
    backup = preserve_consumption_cache(path)
    path.write_bytes(b"new data")
    assert backup.read_bytes() == b"legacy data"
    assert preserve_consumption_cache(backup).read_bytes() == b"legacy data"


def test_sampling_only_cannot_run_as_if_allocation_were_complete():
    country = SimpleNamespace(population=SimpleNamespace(household_consumption_stage="hfcs_sampling_only_v1"))
    with pytest.raises(ValueError, match="opening consumption allocation"):
        require_completed_consumption_allocation(country)
    country.population = SimpleNamespace()
    require_completed_consumption_allocation(country)


def test_builder_passes_first_implicate_to_reader_before_downstream_work(configured, monkeypatch):
    config, root = configured

    def reader(**kwargs):
        assert kwargs["hfcs_num_surveys"] == {"FRA": 1}
        assert kwargs["force_single_hfcs_survey"] is False
        raise RuntimeError("reader boundary reached")

    monkeypatch.setattr(DataReaders, "from_raw_data", reader)
    with pytest.raises(RuntimeError, match="reader boundary reached"):
        DataWrapper.from_config(config, root, single_hfcs_survey=False)


def test_export_rejects_truncated_input_before_publishing(configured):
    config, root = configured
    fields = config.country_configs["FRA"].household_consumption_initialisation
    frame = pd.read_parquet(root / fields.source_file)
    import json

    registration = HFCSConsumptionRegistration(**json.loads((root / fields.registration_file).read_text()))
    output = root / "invalid.parquet"
    with pytest.raises(ValueError, match="source count"):
        export_consumption_source(frame.iloc[:1], registration, output, wave=2014)
    assert not output.exists()


def test_country_constructor_rejects_sampling_only_before_account_initialisation():
    from macromodel.country.country import Country

    country = SimpleNamespace(population=SimpleNamespace(household_consumption_stage="hfcs_sampling_only_v1"))
    with pytest.raises(ValueError, match="opening consumption allocation"):
        Country.from_pickled_country(
            synthetic_country=country,
            country_configuration=None,
            exchange_rates=None,
            country_name="FRA",
            all_country_names=["FRA"],
            industries=[],
            initial_year=2014,
            t_max=1,
            time_unit=3,
            running_multiple_countries=False,
            emission_factors_usd=np.array([]),
        )
