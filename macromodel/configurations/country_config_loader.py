"""Helpers for loading country configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .country_configuration import CountryConfiguration

_PARAMETER_FILE_KEY = "paper_parameter_file"
_PARAMETER_REF_KEY = "paper_parameter_ref"


def _is_country_config_map(payload: Mapping[str, Any]) -> bool:
    return bool(payload) and all(isinstance(key, str) and len(key) == 3 and key.isupper() for key in payload)


def _resolve_dot_path(payload: Mapping[str, Any], dot_path: str) -> Any:
    current: Any = payload
    for part in dot_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"Missing paper parameter section: {dot_path}")
        current = current[part]
    return current


def _resolve_parameter_references(payload: Any, base_dir: Path) -> Any:
    if isinstance(payload, list):
        return [_resolve_parameter_references(item, base_dir) for item in payload]
    if not isinstance(payload, Mapping):
        return payload

    resolved_children = {
        key: _resolve_parameter_references(value, base_dir)
        for key, value in payload.items()
        if key not in {_PARAMETER_FILE_KEY, _PARAMETER_REF_KEY}
    }
    has_file = _PARAMETER_FILE_KEY in payload
    has_ref = _PARAMETER_REF_KEY in payload
    if has_file != has_ref:
        raise ValueError(f"{_PARAMETER_FILE_KEY} and {_PARAMETER_REF_KEY} must be provided together.")
    if not has_file:
        return resolved_children
    if resolved_children:
        keys = ", ".join(sorted(str(key) for key in resolved_children))
        raise ValueError(f"Paper parameter references cannot define sibling parameter keys: {keys}")

    parameter_path = Path(str(payload[_PARAMETER_FILE_KEY]))
    if not parameter_path.is_absolute():
        parameter_path = base_dir / parameter_path
    if not parameter_path.exists():
        raise FileNotFoundError(f"Paper parameter file not found: {parameter_path}")
    with parameter_path.open() as handle:
        parameter_payload = yaml.safe_load(handle) or {}
    parameter_section = _resolve_dot_path(parameter_payload, str(payload[_PARAMETER_REF_KEY]))
    if not isinstance(parameter_section, Mapping):
        raise ValueError(f"Paper parameter section must be a mapping: {payload[_PARAMETER_REF_KEY]}")
    if _PARAMETER_FILE_KEY in parameter_section or _PARAMETER_REF_KEY in parameter_section:
        raise ValueError("Nested paper parameter references are not supported.")
    return dict(parameter_section)


def load_country_configuration(config_path: str | Path, country_iso3: str | None = None) -> CountryConfiguration:
    """Load a country configuration, resolving paper-parameter references first."""
    path = Path(config_path)
    with path.open() as handle:
        payload = yaml.safe_load(handle) or {}
    if country_iso3 is not None:
        if country_iso3 in payload:
            payload = payload[country_iso3]
        elif _is_country_config_map(payload):
            raise ValueError(f"Country {country_iso3!r} not found in {path}")
    elif _is_country_config_map(payload):
        if len(payload) != 1:
            raise ValueError(f"Country configuration file contains multiple countries; pass country_iso3 for {path}")
        payload = next(iter(payload.values()))
    resolved_payload = _resolve_parameter_references(payload, path.parent)
    return CountryConfiguration(**resolved_payload)
