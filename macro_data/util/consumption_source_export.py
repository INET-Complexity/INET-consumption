"""Export the calibration's prepared pre-percentile table as a model input.

Input frames must already contain the calibration C/Y, weights and masks using
original HFCS source IDs. This exporter performs no new estimation or imputation.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from macro_data.util.hfcs_consumption_source import read_hfcs_consumption_source


def export_consumption_source(frame, registration, destination: Path, *, wave: int):
    """Validate and export private Parquet/registration; return config fields.

    ``registration`` comes independently from the complete reviewed calibration
    preparation. Do not construct it from an arbitrarily truncated export frame.
    Only use an original-ID crosswalk verified against source weights/spending.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    registration_path = destination.with_suffix(".registration.json")
    with TemporaryDirectory(dir=destination.parent) as scratch:
        candidate = Path(scratch) / "source.parquet"
        frame.to_parquet(candidate, index=False)
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        read_hfcs_consumption_source(candidate, wave=wave, expected_sha256=digest, registration=registration)
        registration_payload = (json.dumps(asdict(registration), sort_keys=True, indent=2) + "\n").encode()
        # Publish only validated content. A partial/interrupted pair cannot pass
        # the independently pinned file hashes in the generated configuration.
        candidate.replace(destination)
        registration_path.write_bytes(registration_payload)
    return {
        "method": "hfcs_cy",
        "implicate": 1,
        "source_file": str(destination),
        "source_sha256": digest,
        "registration_file": str(registration_path),
        "registration_sha256": hashlib.sha256(registration_payload).hexdigest(),
    }
