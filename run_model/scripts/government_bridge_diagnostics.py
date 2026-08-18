"""Compatibility CLI for :mod:`src.diagnostics.government_bridge`."""

from __future__ import annotations

import sys
from pathlib import Path

RUN_MODEL_DIR = Path(__file__).resolve().parents[1]
if str(RUN_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL_DIR))

from src.diagnostics.government_bridge import main  # noqa: E402

if __name__ == "__main__":
    main()
