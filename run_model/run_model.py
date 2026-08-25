"""Compatibility command for :mod:`src.workflows.single_run`."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

RUN_MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_MODEL_DIR.parent
for path in (str(REPO_ROOT), str(RUN_MODEL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

if __name__ == "__main__":
    runpy.run_module("src.workflows.single_run", run_name="__main__")
else:
    from src.workflows import single_run as _implementation

    sys.modules[__name__] = _implementation
