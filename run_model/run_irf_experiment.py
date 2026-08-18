"""Compatibility entry point for :mod:`src.experiments.irf`.

Reusable implementation lives under ``src``. Importing this legacy module
returns that implementation module so existing monkeypatch-based callers keep
working; executing the file retains the historical CLI command.
"""

from __future__ import annotations

import sys
from pathlib import Path

RUN_MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_MODEL_DIR.parent
for path in (str(REPO_ROOT), str(RUN_MODEL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import src.experiments.irf as _implementation  # noqa: E402

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
