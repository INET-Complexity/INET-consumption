"""Validated source-record handoff for HFCS consumption initialisation.

Source tables are private microdata. Keep them outside version control. The
caller supplies the calibration's pre-percentile validity and threshold-pool
masks, not its trimmed output. No INET-side income/tax re-estimation is done.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from macro_data.util.consumption_initialisation import (
    WinsorisedConsumptionRatios,
    winsorise_hfcs_consumption_ratios,
)

SOURCE_COLUMNS = ("source_id", "wave", "c_y", "survey_weight", "hard_valid", "threshold_pool")
# Optional annual HFCS income columns (calibration names), used for balance-sheet scaling.
INCOME_COLUMNS = ("disp_geom_avg_income", "disposable_income", "disp_non_property_income")


@dataclass(frozen=True)
class HFCSConsumptionRegistration:
    """Expected preparation facts registered independently of the handoff file.

    Derive these from the complete calibration preparation, never from the
    candidate file being validated. The registration must accompany its
    reviewed preparation manifest; file SHA alone establishes no completeness.
    """

    source_count: int
    source_ids_sha256: str
    preparation_manifest_sha256: str
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99


def source_ids_fingerprint(source_ids: np.ndarray) -> str:
    """Order-independent fingerprint of a source ID set, preserving strings."""
    ids = _identifiers(source_ids)
    if pd.Index(ids).has_duplicates:
        raise ValueError("Source identifiers must be unique for registration.")
    encoded = json.dumps(sorted(ids.tolist()), ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class HFCSConsumptionSource:
    """One source wave, with unique stable identifiers and winsorised ratios."""

    source_ids: np.ndarray
    wave: int
    winsorised: WinsorisedConsumptionRatios
    sha256: str
    survey_weights: np.ndarray
    incomes: dict = field(default_factory=dict)

    def ratios_for(self, source_ids: np.ndarray) -> np.ndarray:
        """Link in draw order, allowing replacement draws but never fallback.

        Hard-invalid source rows and unknown IDs are distinct errors. Neither
        receives an invented spending propensity. Repeated model draws retain
        their source ratio without applying source survey weights a second time.
        """
        requested = _identifiers(source_ids)
        indexer = pd.Index(self.source_ids).get_indexer(requested)
        if np.any(indexer < 0):
            raise ValueError(f"{np.count_nonzero(indexer < 0)} household draws have no HFCS source match.")
        if np.any(~self.winsorised.valid[indexer]):
            raise ValueError("Household draws include hard-invalid HFCS records; choose their treatment explicitly.")
        return self.winsorised.ratios[indexer].copy()

    def incomes_for(self, source_ids: np.ndarray, concept: str) -> np.ndarray:
        """Annual HFCS income on ``concept`` in draw order (NaN where the source value is missing)."""
        if concept not in self.incomes:
            raise ValueError(f"HFCS source has no income column {concept!r}.")
        indexer = pd.Index(self.source_ids).get_indexer(_identifiers(source_ids))
        if np.any(indexer < 0):
            raise ValueError(f"{np.count_nonzero(indexer < 0)} household draws have no HFCS source match.")
        return self.incomes[concept][indexer].copy()


def _identifiers(values: np.ndarray) -> np.ndarray:
    """Require explicit stable string identifiers, preserving leading zeros."""
    ids = np.asarray(values, dtype=object)
    if ids.ndim != 1 or any(not isinstance(x, str) or not x.strip() or x != x.strip() for x in ids):
        raise ValueError("source_id must be a vector of nonempty, unpadded stable strings.")
    return ids.copy()


def read_hfcs_consumption_source(
    path: Path, *, wave: int, expected_sha256: str, registration: HFCSConsumptionRegistration
) -> HFCSConsumptionSource:
    """Read a hash-pinned calibration source table and winsorise before draws.

    The Parquet schema is SOURCE_COLUMNS. c_y must be total consumption divided
    by disposable geometric-average income net of interest plus HMR imputed
    rent. The table must be exported from that calibration preparation. The
    file hash pins both values and masks; it is not inferred from filenames.
    All source records for the chosen wave must be present before percentiles
    are computed; do not export only the final retained sample.
    """
    if not isinstance(registration, HFCSConsumptionRegistration):
        raise ValueError("An independent HFCS preparation registration is required.")
    count = registration.source_count
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count <= 0:
        raise ValueError("Registered source_count must be a positive integer.")
    for label, digest_value in (
        ("source_ids_sha256", registration.source_ids_sha256),
        ("preparation_manifest_sha256", registration.preparation_manifest_sha256),
    ):
        if (
            not isinstance(digest_value, str)
            or len(digest_value) != 64
            or any(c not in "0123456789abcdef" for c in digest_value)
        ):
            raise ValueError(f"Registered {label} must be a lowercase SHA-256.")
    quantiles = np.asarray([registration.lower_quantile, registration.upper_quantile], dtype=float)
    if quantiles.shape != (2,) or not np.all(np.isfinite(quantiles)) or not 0 <= quantiles[0] < quantiles[1] <= 1:
        raise ValueError("Registered quantiles must be ordered scalar probabilities.")
    if not isinstance(wave, (int, np.integer)) or isinstance(wave, (bool, np.bool_)):
        raise ValueError("wave must be an integer survey year.")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("A 64-character source SHA-256 is required.")
    try:
        int(expected_sha256, 16)
    except ValueError as exc:
        raise ValueError("Source SHA-256 must be hexadecimal.") from exc
    path = Path(path)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256.lower():
        raise ValueError("HFCS consumption source SHA-256 mismatch.")
    # Read the same bytes that were hashed, even if the file changes meanwhile.
    from io import BytesIO

    frame = pd.read_parquet(BytesIO(payload))
    missing = set(SOURCE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"HFCS consumption source is missing columns: {sorted(missing)}")
    required_metadata = {
        "consumption_concept": "total",
        "income_concept": "net_of_interest_plus_imputed_rent",
        "income_scale_var": "disp_geom_avg_income",
        "preparation_manifest_sha256": registration.preparation_manifest_sha256,
        "preparation_stage": "before_percentile_trimming",
    }
    for key, expected in required_metadata.items():
        if frame.attrs.get(key) != expected:
            raise ValueError(f"HFCS consumption source requires {key}={expected!r}.")
    if not pd.api.types.is_integer_dtype(frame["wave"].dtype) or frame["wave"].isna().any():
        raise ValueError("Source wave must contain nonmissing integer survey years.")
    frame = frame.loc[frame["wave"].eq(wave)].copy()
    if frame.empty:
        raise ValueError("Requested wave is absent from the HFCS consumption source.")
    ids = _identifiers(frame["source_id"].to_numpy())
    if pd.Index(ids).has_duplicates:
        raise ValueError("HFCS source identifiers must be unique within the selected wave.")
    if len(frame) != registration.source_count:
        raise ValueError("HFCS source count differs from the registered complete source population.")
    if source_ids_fingerprint(ids) != registration.source_ids_sha256:
        raise ValueError("HFCS source ID set differs from the registered complete source population.")
    for column in ("hard_valid", "threshold_pool"):
        if frame[column].isna().any() or not pd.api.types.is_bool_dtype(frame[column].dtype):
            raise ValueError(f"{column} must be a nonmissing boolean source mask.")
    winsorised = winsorise_hfcs_consumption_ratios(
        c_y=frame["c_y"].to_numpy(dtype=float),
        survey_weights=frame["survey_weight"].to_numpy(dtype=float),
        hard_valid=frame["hard_valid"].to_numpy(dtype=bool),
        threshold_pool=frame["threshold_pool"].to_numpy(dtype=bool),
        lower_quantile=registration.lower_quantile,
        upper_quantile=registration.upper_quantile,
    )
    incomes = {
        name: pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float, copy=True)
        for name in INCOME_COLUMNS
        if name in frame.columns
    }
    return HFCSConsumptionSource(
        ids, int(wave), winsorised, digest, frame["survey_weight"].to_numpy(dtype=float, copy=True), incomes
    )
