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
            "Three-level modelling pipeline: "
            "Level 1 sepsis → Level 2 etiology gate/subtype → Level 3 resistance"
        )
    )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    path_group = parser.add_argument_group("Paths")

    path_group.add_argument(
        "--database-file",
        "-db",
        type=Path,
        default=default_db,
        help="Input preprocessed CSV file.",
    )

    path_group.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=default_out,
        help="Output directory for training artifacts.",
    )

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------
    target_group = parser.add_argument_group("Targets")

    target_group.add_argument(
        "--sepsis-target",
        type=str,
        default=targets["sepsis_target"],
        help="Column name for the Level 1 sepsis target.",
    )

    target_group.add_argument(
        "--etiology-target",
        type=str,
        default=targets["hemo_target"],
        help=(
            "Column name for the Level 2 etiology target. "
            "Can be binary or multiclass depending on configuration."
        ),
    )

    target_group.add_argument(
        "--etiology-gate-target",
        type=str,
        default=targets["hemo_gate_target"],
        help=(
            "Column name for the Level 2 binary etiology gate target. "
            "This decides which rows move to Stage 2."
        ),
    )

    target_group.add_argument(
        "--etiology-gate-negative-label",
        type=str,
        default=targets["hemo_gate_negative_label"],
        help="Label treated as gate-negative for the Level 2 binary etiology gate.",
    )

    target_group.add_argument(
        "--resistance-gate-target",
        type=str,
        default=targets.get("resistance_gate_target", targets["hemo_gate_target"]),
        help=(
            "Column name for the independent Level 3 resistance gate target. "
            "Defaults to the Level 2 gate target for backward compatibility."
        ),
    )

    target_group.add_argument(
        "--resistance-gate-negative-label",
        type=str,
        default=targets.get("resistance_gate_negative_label", targets["hemo_gate_negative_label"]),
        help="Label treated as gate-negative for the Level 3 resistance gate.",
    )

    target_group.add_argument(
        "--cef-target",
        type=str,
        default=targets["cef_target"],
        help="Column name for the cefalosporin-resistance target.",
    )

    target_group.add_argument(
        "--weight-column",
        type=str,
        default=targets["weight_column"],
        help="Column containing sample weights.",
    )

    # ------------------------------------------------------------------
    # Level 2 configuration
    # ------------------------------------------------------------------
    l2_group = parser.add_argument_group("Level 2 etiology gate configuration")

    l2_group.add_argument(
        "--skip-l2-gate",
        action="store_true",
        default=bool(tcfg.get("skip_l2_gate", False)),
        help=(
            "Skip the Level 2 etiology positive/negative gate and directly "
            "predict etiology classes, including NEGATIVE, in a single model."
        ),
    )

    l2_group.add_argument(
        "--l2-gate-proba-as-feature",
        action="store_true",
        default=bool(tcfg.get("l2_gate_proba_as_feature", False)),
        help=(
            "When using the Level 2 binary gate, add the gate probability as an "
            "extra feature to the Stage 2 subtype model."
        ),
    )

    l2_group.add_argument(
        "--level2-focus",
        nargs="+",
        default=tcfg.get("level2_focus", []),
        help=(
            "Optional foco allowlist for Level 2 etiology. "
            "Use one or more foco labels, e.g. --level2-focus urinario "
            "or --level2-focus urinario pulmonar."
        ),
    )

    l2_group.add_argument(
        "--set-gate-recall",
        type=float,
        default=tcfg.get("set_gate_recall", None),
        help=(
            "Optional recall target for Level 2 gate threshold selection. "
            "If unset, the default threshold strategy is used."
        ),
    )

    l3_group = parser.add_argument_group("Level 3 resistance configuration")

    l3_group.add_argument(
        "--set-resistance-gate-recall",
        type=float,
        default=tcfg.get("set_resistance_gate_recall", None),
        help=(
            "Optional recall target for the independent Level 3 resistance-gate "
            "threshold selection. If unset, the default threshold strategy is used."
        ),
    )

    l2_group.add_argument(
        "--etiology-stage2-mode",
        type=str,
        choices=["auto", "binary", "multiclass"],
        default=tcfg.get("hemo_stage2_mode", "auto"),
        help=(
            "Level 2 Stage 2 subtype mode, also used for direct etiology when "
            "--skip-l2-gate is set. "
            "auto = keep target classes and use binary estimator only if exactly "
            "2 labels; binary = GNB vs non-GNB; "
            "multiclass = predict all etiology classes."
        ),
    )

    l3_group.add_argument(
        "--resistance-mode",
        type=str,
        choices=["auto", "gated", "resistance_gate", "direct", "true_positive_only"],
        default=tcfg.get("resistance_mode", "auto"),
        help=(
            "Population used for Level 3 resistance modelling. "
            "auto = gated when the Level 2 gate is available, otherwise direct; "
            "gated = train on true culture-positive rows and test on rows "
            "predicted culture-positive by the existing Level 2 gate; "
            "resistance_gate = train an independent infected +/- gate for "
            "Level 3, then train resistance on true culture-positive rows and "
            "test resistance on rows predicted positive by that Level 3 gate; "
            "direct = train/test on all rows with a non-null resistance target, "
            "independent of Level 2; "
            "true_positive_only = train/test only on true culture-positive rows "
            "without using predicted gate output."
        ),
    )

    # ------------------------------------------------------------------
    # Model and training
    # ------------------------------------------------------------------
    training_group = parser.add_argument_group("Model and training")

    training_group.add_argument(
        "--model-type",
        type=str,
        choices=["xgb", "lgbm", "rf", "catb"],
        default=tcfg["model_type"],
        help="Model backend.",
    )

    training_group.add_argument(
        "--binary-trials",
        "-btrials",
        type=int,
        default=tcfg["binary_trials"],
        help="Number of Optuna trials for binary models.",
    )

    training_group.add_argument(
        "--cv-splits",
        type=int,
        default=tcfg["cv_splits"],
        help="Number of cross-validation splits.",
    )

    training_group.add_argument(
        "--test-size",
        "-tsize",
        type=float,
        default=tcfg["test_size"],
        help="Test-set proportion.",
    )

    training_group.add_argument(
        "--random-state",
        type=int,
        default=tcfg["random_state"],
        help="Random seed.",
    )

    training_group.add_argument(
        "--l3-min-positive",
        type=int,
        default=tcfg.get("l3_min_positive", 30),
        help="Minimum positive cases required for Level 3 modelling.",
    )

    # ------------------------------------------------------------------
    # Feature preprocessing and selection
    # ------------------------------------------------------------------
    feature_group = parser.add_argument_group("Feature preprocessing and selection")

    feature_group.add_argument(
        "--na-perc-limit",
        "-na",
        type=float,
        default=tcfg["na_perc_limit"],
        help="Maximum allowed missingness proportion before dropping a feature.",
    )

    feature_group.add_argument(
        "--max-features",
        type=int,
        default=tcfg["max_features"],
        help="Maximum number of selected features per model.",
    )

    feature_group.add_argument(
        "--max-corr",
        type=float,
        default=tcfg["max_corr"],
        help="Maximum allowed absolute feature correlation.",
    )

    feature_group.add_argument(
        "--no-impute",
        dest="impute_missing",
        action="store_false",
        default=bool(tcfg["impute_missing"]),
        help="Disable missing-value imputation. Default is controlled by config.",
    )

    feature_group.add_argument(
        "--skip-rfecv",
        action="store_true",
        default=bool(tcfg.get("skip_rfecv", False)),
        help="Skip SHAP RFECV and use all features after preprocessing.",
    )

    # ------------------------------------------------------------------
    # Optuna
    # ------------------------------------------------------------------
    optuna_group = parser.add_argument_group("Optuna")

    optuna_group.add_argument(
        "--optuna-storage",
        type=str,
        default=tcfg["optuna_storage"],
        help="Optuna storage URI.",
    )

    optuna_group.add_argument(
        "--optuna-study-prefix",
        type=str,
        default=tcfg["optuna_study_prefix"],
        help="Prefix for Optuna study names.",
    )

    optuna_group.add_argument(
        "--optuna-load-if-exists",
        action="store_true",
        default=bool(tcfg.get("optuna_load_if_exists", False)),
        help="Reuse existing Optuna studies if present.",
    )

    optuna_group.add_argument(
        "--skip-optuna",
        action="store_true",
        default=False,
        help="Disable Optuna and fit with fixed model defaults.",
    )

    # ------------------------------------------------------------------
    # Stage skipping
    # ------------------------------------------------------------------
    skip_group = parser.add_argument_group("Stage skipping")

    skip_group.add_argument(
        "--skip-level1",
        action="store_true",
        default=False,
        help="Skip Level 1 sepsis training/prediction.",
    )

    skip_group.add_argument(
        "--skip-level2",
        action="store_true",
        default=False,
        help="Skip Level 2 etiology training/prediction.",
    )

    skip_group.add_argument(
        "--skip-level3",
        action="store_true",
        default=False,
        help="Skip Level 3 resistance training/prediction.",
    )

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------
    artifact_group = parser.add_argument_group("Artifacts")

    artifact_group.add_argument(
        "--skip-shap-values",
        action="store_true",
        default=bool(tcfg.get("skip_shap_values", False)),
        help="Skip saving SHAP value artifacts.",
    )

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

    print(f"\\nElapsed: {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
