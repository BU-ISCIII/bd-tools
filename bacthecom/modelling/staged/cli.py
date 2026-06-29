"""CLI for the staged mortality modelling pipeline."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from .config import runtime_defaults


def build_arg_parser() -> argparse.ArgumentParser:
    cfg = runtime_defaults()
    tcfg = cfg["training"]
    targets = cfg["targets"]
    default_out = Path.cwd() / cfg["paths"]["default_results_rel"]

    parser = argparse.ArgumentParser(
        description=(
            "Two-stage BACTHECOM mortality pipeline: "
            "Stage 1 predicts global mortality; Stage 2 predicts selected horizon "
            "on Stage-1-positive subset with leakage-safe OOF feature injection."
        )
    )
    parser.add_argument("--data", type=Path, required=True, help="Input CSV dataset path.")
    parser.add_argument("--results-dir", "--results_dir", dest="results_dir", type=Path, default=default_out)

    parser.add_argument("--target1", type=str, default=targets["target1"], help="Stage 1 binary target column.")
    parser.add_argument("--target2", type=str, required=True, help="Stage 2 binary target column (e.g. mortalidad_14_dias).")

    parser.add_argument("--algorithm", type=str, choices=["catb", "lgbm", "xgb"], default=tcfg["algorithm"], help="Model family for both stages.")
    parser.add_argument("--n-trials", "--n_trials", dest="n_trials", type=int, default=tcfg["n_trials"])
    parser.add_argument("--early-stopping-rounds", "--early_stopping_rounds", dest="early_stopping_rounds", type=int, default=tcfg["early_stopping_rounds"])
    parser.add_argument("--fast-search", "--fast_search", dest="fast_search", action="store_true", help="Narrow Optuna space for quicker development runs.")

    parser.add_argument("--testing-split-size", "--testing_split_size", dest="testing_split_size", type=float, default=tcfg["testing_split_size"])
    parser.add_argument("--training-n-splits", "--training_n_splits", dest="training_n_splits", type=int, default=tcfg["training_n_splits"])

    parser.add_argument("--study-name", "--study_name", dest="study_name", type=str, required=True)
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URI, e.g. sqlite:///optuna.db")
    parser.add_argument("--seed", type=int, default=tcfg["seed"])
    parser.add_argument("--n-cpus", "--n_cpus", dest="n_cpus", type=int, default=tcfg["n_cpus"])
    parser.add_argument("--job-id", "--job_id", dest="job_id", type=str, default="local")
    parser.add_argument("--array-id", "--array_id", dest="array_id", type=str, default="0")

    parser.add_argument("--utils-json", "--utils_json", dest="utils_json", type=Path, default=None, help="Optional JSON with {'drop_columns': [...]}.")
    parser.add_argument("--na-perc-limit", "--na_perc_limit", dest="na_perc_limit", type=float, default=80.0)
    parser.add_argument("--iqr-multiplier", "--iqr_multiplier", dest="iqr_multiplier", type=float, default=5.0)
    parser.add_argument("--max-corr", "--max_corr", dest="max_corr", type=float, default=0.95)

    parser.add_argument(
        "--stage2-use-predicted-positive-test",
        action="store_true",
        help="Also evaluate deployed cascade on Stage-1 predicted-positive test subset.",
    )
    return parser


def main() -> None:
    start = time.time()
    parser = build_arg_parser()
    args = parser.parse_args()
    print("Parsed args:", args)

    from .modelling import run_training

    output_folder = Path(args.results_dir)
    try:
        run_training(args)
    except Exception:
        if output_folder.exists():
            shutil.rmtree(output_folder)
        raise

    print(f"\nElapsed: {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
