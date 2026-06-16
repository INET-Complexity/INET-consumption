"""Run Stage 0 household account diagnostics from a data pickle and model HDF5."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RUN_MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_MODEL_DIR.parent
for path in (str(REPO_ROOT), str(RUN_MODEL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.household_account_diagnostics import (  # noqa: E402
    run_household_account_diagnostics,
    write_household_account_outputs,
)

from macro_data import DataWrapper  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-pkl", required=True, type=Path, help="DataWrapper pickle with synthetic population.")
    parser.add_argument("--model-h5", required=True, type=Path, help="Saved model HDF5 file.")
    parser.add_argument("--country-code", required=True, help="Country ISO3 code, e.g. FRA.")
    parser.add_argument("--period", type=int, default=None, help="HDF5 row to diagnose. Defaults to the final row.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for diagnostic outputs.")
    parser.add_argument(
        "--parameter-file",
        type=Path,
        default=RUN_MODEL_DIR / "config" / "consumption_paper_parameters.yaml",
        help="Active consumption parameter registry.",
    )
    return parser.parse_args()


def main(
    *,
    data_pkl: str | Path,
    model_h5: str | Path,
    country_code: str,
    period: int | None = None,
    output_dir: str | Path,
    parameter_file: str | Path = RUN_MODEL_DIR / "config" / "consumption_paper_parameters.yaml",
) -> dict[str, Path]:
    data = DataWrapper.init_from_pickle(data_pkl)
    diagnostics = run_household_account_diagnostics(
        datawrapper=data,
        model_h5=model_h5,
        country_code=country_code,
        period=period,
        parameter_file=parameter_file,
    )
    return write_household_account_outputs(diagnostics, output_dir, country_code=country_code)


if __name__ == "__main__":
    args = _parse_args()
    outputs = main(
        data_pkl=args.data_pkl,
        model_h5=args.model_h5,
        country_code=args.country_code,
        period=args.period,
        output_dir=args.output_dir,
        parameter_file=args.parameter_file,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
