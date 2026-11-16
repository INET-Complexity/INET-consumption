"""Configured HFCS source loading and cache compatibility."""

import hashlib
import json
import shutil
from pathlib import Path

from macro_data.util.hfcs_consumption_source import HFCSConsumptionRegistration, read_hfcs_consumption_source

# Opening allocation occurs at Country construction, after effective taxes and
# contractual opening income are available. Reject older sampling-only caches.
CONSUMPTION_DATA_STAGE = "hfcs_opening_allocation_individual_pensions_retirement_v3"


def configured_consumption_inputs(configuration, raw_data_path):
    """Resolve and verify private sources before country building/cache reuse."""
    result = {}
    country_configs = (
        configuration.get("country_configs", {})
        if isinstance(configuration, dict)
        else getattr(configuration, "country_configs", {})
    )
    for country, country_config in country_configs.items():
        config = (
            country_config.get("household_consumption_initialisation")
            if isinstance(country_config, dict)
            else getattr(country_config, "household_consumption_initialisation", None)
        )
        if config is not None and not isinstance(config, dict):
            config = config.model_dump(mode="json")
        if config is None or config.get("method", "legacy") == "legacy":
            continue
        if config.get("method") != "hfcs_cy" or config.get("implicate", 1) != 1:
            raise ValueError("Expected method hfcs_cy with implicate 1.")
        files = {}
        for field in ("source", "registration"):
            path = Path(config[field + "_file"])
            if not path.is_absolute():
                path = Path(raw_data_path) / path
            path = path.resolve()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != config[field + "_sha256"]:
                raise ValueError(f"{country}: {field} SHA-256 mismatch; regenerate the registered handoff/config.")
            files[field] = {"path": str(path), "sha256": digest}
        result[getattr(country, "value", str(country))] = {
            "method": config["method"],
            "implicate": config.get("implicate", 1),
            **files,
        }
    return result


def configured_hfcs_extractions(configuration):
    """Explicit first-implicate selection, including legacy comparisons."""
    configs = (
        configuration.get("country_configs", {})
        if isinstance(configuration, dict)
        else getattr(configuration, "country_configs", {})
    )
    result = {}
    for country, config in configs.items():
        selection = (
            config.get("hfcs_implicate") if isinstance(config, dict) else getattr(config, "hfcs_implicate", None)
        )
        if selection is not None:
            if selection != 1:
                raise ValueError("Only first-implicate extraction is supported.")
            result[getattr(country, "value", str(country))] = 1
    return result


def consumption_data_signature(configuration, raw_data_path, *, single_hfcs_survey=False):
    """None preserves compatibility with genuinely legacy data pickles."""
    sources = configured_consumption_inputs(configuration, raw_data_path)
    extraction = configured_hfcs_extractions(configuration)
    if not sources and not extraction:
        return None
    return {
        "stage": CONSUMPTION_DATA_STAGE if sources else "legacy_first_implicate_individual_pensions_retirement_v3",
        "extraction": extraction,
        "legacy_history_selection": bool(single_hfcs_survey),
        "configuration": configuration if isinstance(configuration, dict) else configuration.model_dump(mode="json"),
        "sources": sources,
    }


def load_configured_consumption_sources(configuration, raw_data_path):
    inputs = configured_consumption_inputs(configuration, raw_data_path)
    result = {}
    for country, config in inputs.items():
        registration_path = Path(config["registration"]["path"])
        payload = registration_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != config["registration"]["sha256"]:
            raise ValueError("Registration changed while loading HFCS consumption inputs.")
        registration = HFCSConsumptionRegistration(**json.loads(payload))
        result[country] = read_hfcs_consumption_source(
            Path(config["source"]["path"]),
            wave=configuration.year,
            expected_sha256=config["source"]["sha256"],
            registration=registration,
        )
    return result


def consumption_cache_requires_rebuild(data, configuration, raw_data_path, *, single_hfcs_survey=False):
    return getattr(data, "household_consumption_signature", None) != consumption_data_signature(
        configuration, raw_data_path, single_hfcs_survey=single_hfcs_survey
    )


def require_completed_consumption_allocation(synthetic_country):
    if str(getattr(synthetic_country.population, "household_consumption_stage", "")).startswith("hfcs_sampling_only"):
        raise ValueError(
            "This HFCS sampling-only cache predates the completed opening consumption allocation. "
            "Rebuild it using the current data configuration, or select method: legacy."
        )


def preserve_consumption_cache(path):
    """Retain an exact content-addressed copy before changing data method."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    backup = path.with_name(f"{path.stem}.consumption-backup-{digest}{path.suffix}")
    if not backup.exists():
        shutil.copy2(path, backup)
    elif hashlib.sha256(backup.read_bytes()).hexdigest() != digest:
        raise ValueError("Existing consumption-cache backup has inconsistent contents.")
    return backup
