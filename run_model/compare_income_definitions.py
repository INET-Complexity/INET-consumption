"""Run paired legacy/reconciled income scenarios with one shared data cache.

Run from run_model, for example:
    python compare_income_definitions.py --seed 40 --t-max 50

The legacy arm retains the annualisation correction. This isolates income
reconciliation, rather than reproducing every historical bug or parameter set.
The existing config/.env and API environment settings are used by the workflow.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd
from config import SCENARIO_PRESETS
from src.notebook_state import run_notebook_workflow, write_run_manifest
from src.notebook_workflow import NotebookRunConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--t-max", type=int, default=50)
    parser.add_argument("--country", default="FRA")
    parser.add_argument("--output-dir", type=Path, default=Path("data/output_data/income_comparison"))
    args = parser.parse_args()
    output_dir = args.output_dir.resolve() / f"{args.country}_seed{args.seed}_t{args.t_max}"
    config = NotebookRunConfig(
        seed=args.seed,
        t_max=args.t_max,
        country_iso3=args.country,
        output_dir=output_dir,
        force_rebuild_data=False,
        run_benchmark=False,
    )
    frames = {}
    for mode in ("legacy", "reconciled"):
        name = f"calibrated_income_{mode}"
        # Each run loads the same cache into a fresh model. Never continue a
        # model constructed with the other income definition.
        state = run_notebook_workflow(
            replace(config, model_file_name=f"simulation_{args.country}_{mode}.h5"),
            scenario_name=name,
            scenario_overrides=SCENARIO_PRESETS[name],
            previous_state=None,
            write_manifest=False,
        )
        write_run_manifest(state, output_dir / f"manifest_{mode}.json")
        frame = state.df_scenario.copy()
        frame.to_csv(output_dir / f"macro_{mode}.csv")
        frames[mode] = frame
    pd.concat(frames, axis=1).to_csv(output_dir / "macro_comparison.csv")
    print(f"Paired model files, manifests and macro comparison saved in {output_dir}")


if __name__ == "__main__":
    main()
