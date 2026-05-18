"""CLI for the three-level modelling pipeline."""

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
    home = Path.cwd()
    default_db = home / cfg["paths"]["default_database_rel"]
    default_out = home / cfg["paths"]["default_output_rel"]

    parser = argparse.ArgumentParser(
        description=(
            "Three-level independent (no cascade): "
            "sepsis → resultado_hemo_grouped → resistente_cefalosporina"
        )
    )
    parser.add_argument("--database-file", "-db", type=Path, default=default_db)
    parser.add_argument("--output-dir", "-o", type=Path, default=default_out)
    parser.add_argument(
        "--sepsis-target", type=str, default=targets["sepsis_target"],
        help="Column name for the Level-1 binary target (default: sepsis).",
    )
    parser.add_argument(
        "--hemo-target", type=str, default=targets["hemo_target"],
        help=(
            "Column name for the Level-2 multiclass target (default: resultado_hemo_grouped). "
            "Raw resultado_hemo values are also supported; when more than two positive ``hemo`` labels "
            "exist, Stage 2 will automatically use multiclass modeling."
        ),
    )
    parser.add_argument(
        "--hemo-negative-label",
        type=str,
        default=targets["hemo_negative_label"],
        help="Label treated as hemoculture-negative for Level-2 Stage-1 gate.",
    )
    parser.add_argument(
        "--skip-l2-gate",
        action="store_true",
        help=(
            "Skip Level 2 hemoculture positive/negative gate and directly predict "
            "etiology classes (including NEGATIVE) in a single model."
        ),
    )
    parser.add_argument(
        "--l2-gate-proba-as-feature",
        action="store_true",
        help=(
            "When using the Level 2 binary gate, add the gate probability as an "
            "extra feature to the Stage 2 subtype model."
        ),
    )
    parser.add_argument("--hemo-coco-label", type=str, default=targets["hemo_coco_label"])
    parser.add_argument("--hemo-bacilo-label", type=str, default=targets["hemo_bacilo_label"])
    parser.add_argument("--bmr-target", type=str, default=targets["bmr_target"])
    parser.add_argument("--bmr-negative-label", type=str, default=targets["bmr_negative_label"])
    parser.add_argument("--cef-target", type=str, default=targets["cef_target"])
    parser.add_argument("--skip-l3-hemo-filter", action="store_true")
    parser.add_argument("--cef-multi-positive-label", type=str, default=targets["cef_multi_positive_label"])
    parser.add_argument("--weight-column", type=str, default=targets["weight_column"])
    parser.add_argument(
        "--model-type", type=str, choices=["xgb", "lgbm", "rf", "catb"], default=tcfg["model_type"]
    )
    parser.add_argument("--binary-trials", "-btrials", type=int, default=tcfg["binary_trials"])
    parser.add_argument("--cv-splits", type=int, default=tcfg["cv_splits"])
    parser.add_argument("--test-size", "-tsize", type=float, default=tcfg["test_size"])
    parser.add_argument("--random-state", type=int, default=tcfg["random_state"])
    parser.add_argument("--optuna-storage", type=str, default=tcfg["optuna_storage"])
    parser.add_argument("--optuna-study-prefix", type=str, default=tcfg["optuna_study_prefix"])
    parser.add_argument("--optuna-load-if-exists", action="store_true")
    parser.add_argument("--na-perc-limit", "-na", type=float, default=tcfg["na_perc_limit"])
    parser.add_argument("--max-features", type=int, default=tcfg["max_features"])
    parser.add_argument("--l3-min-positive", type=int, default=tcfg["l3_min_positive"])
    parser.add_argument("--max-corr", type=float, default=tcfg["max_corr"])
    parser.add_argument(
        "--no-impute", dest="impute_missing", action="store_false",
        help="Disable missing-value imputation (default: enabled).",
    )
    parser.set_defaults(
        impute_missing=bool(tcfg["impute_missing"]),
        skip_l2_gate=bool(tcfg["skip_l2_gate"]),
        skip_l3_hemo_filter=bool(tcfg["skip_l3_hemo_filter"]),
        skip_rfecv=bool(tcfg["skip_rfecv"]),
        optuna_load_if_exists=bool(tcfg["optuna_load_if_exists"]),
        l2_gate_proba_as_feature=bool(tcfg.get("l2_gate_proba_as_feature", False)),
        l3_gnb_gate_proba_as_feature=bool(tcfg.get("l3_gnb_gate_proba_as_feature", False)),
    )
    parser.add_argument("--skip-rfecv", action="store_true", help="Skip RFECV and use all features.")
    parser.add_argument("--skip-optuna", action="store_true", help="Disable Optuna and fit with fixed model defaults.")
    parser.add_argument("--skip-level1", action="store_true", help="Skip Level 1 sepsis training/prediction.")
    parser.add_argument("--skip-level2", action="store_true", help="Skip Level 2 hemoculture etiology training/prediction.")
    parser.add_argument("--skip-level3", action="store_true", help="Skip Level 3 resistance training/prediction.")
    parser.add_argument("--l3-use-gnb-gate", action="store_true", help="Enable optional Level-3 GNB binary gate before resistance model.")
    parser.add_argument(
        "--l3-gnb-gate-proba-as-feature",
        action="store_true",
        help=(
            "When using the optional Level 3 GNB gate, add the gate probability "
            "as an extra feature to the Level 3 resistance model instead of "
            "hard subsetting on GNB-positive rows."
        ),
    )
    parser.add_argument("--l3-gnb-positive-label", type=str, default="GNBSelected", help="Positive label in --hemo-target used as GNB-positive when --l3-use-gnb-gate is enabled.")
    return parser


def main() -> None:
    from .modelling import run_training

    start = time.time()
    parser = build_arg_parser()
    args = parser.parse_args()
    print("Parsed args:", args)

    output_folder = Path(args.output_dir)
    try:
        run_training(args)
    except Exception:
        if output_folder.exists():
            shutil.rmtree(output_folder)
        raise
    print(f"\nElapsed: {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
