"""CLI for the three-level modelling pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from .config import runtime_defaults


def _proportion(value: str) -> float:
    """Parse a numeric proportion constrained to the closed interval [0, 1]."""
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(f"expected a value between 0 and 1, got {value}")
    return parsed


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


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
        "--database-file", "-db", type=Path, default=default_db,
        help="Input preprocessed CSV file.",
    )
    path_group.add_argument(
        "--output-dir", "-o", type=Path, default=default_out,
        help="Output directory for training artifacts.",
    )

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------
    target_group = parser.add_argument_group("Targets")

    target_group.add_argument(
        "--sepsis-target", type=str, default=targets["sepsis_target"],
        help="Column name for the Level 1 sepsis target.",
    )
    target_group.add_argument(
        "--etiology-target", type=str, default=targets["hemo_target"],
        help="Column name for the Level 2 etiology target.",
    )
    target_group.add_argument(
        "--etiology-gate-target", type=str, default=targets["hemo_gate_target"],
        help="Column name for the Level 2 binary etiology gate target.",
    )
    target_group.add_argument(
        "--etiology-gate-negative-label", type=str,
        default=targets["hemo_gate_negative_label"],
        help="Label treated as gate-negative for the Level 2 binary etiology gate.",
    )
    target_group.add_argument(
        "--resistance-gate-target", type=str,
        default=targets.get("resistance_gate_target", targets["hemo_gate_target"]),
        help="Column name for the independent Level 3 resistance gate target.",
    )
    target_group.add_argument(
        "--resistance-gate-negative-label", type=str,
        default=targets.get(
            "resistance_gate_negative_label", targets["hemo_gate_negative_label"]
        ),
        help="Label treated as gate-negative for the Level 3 resistance gate.",
    )
    target_group.add_argument(
        "--cef-target", type=str, default=targets["cef_target"],
        help="Column name for the cefalosporin-resistance target.",
    )
    target_group.add_argument(
        "--weight-column", type=str, default=targets["weight_column"],
        help="Column containing sample weights.",
    )

    # ------------------------------------------------------------------
    # Level 2 configuration
    # ------------------------------------------------------------------
    l2_group = parser.add_argument_group("Level 2 etiology gate configuration")

    l2_group.add_argument(
        "--skip-l2-gate", action="store_true",
        default=bool(tcfg.get("skip_l2_gate", False)),
        help="Skip the Level 2 positive/negative gate and directly predict etiology.",
    )
    l2_group.add_argument(
        "--l2-gate-proba-as-feature", action="store_true",
        default=bool(tcfg.get("l2_gate_proba_as_feature", False)),
        help="Add Level 2 gate probability as a feature to Stage 2 subtype modelling.",
    )
    l2_group.add_argument(
        "--level2-focus", nargs="+", default=tcfg.get("level2_focus", []),
        help="Optional foco allowlist for Level 2 etiology.",
    )

    l2_group.add_argument(
        "--etiology-stage2-mode", type=str,
        choices=["auto", "binary", "multiclass"],
        default=tcfg.get("hemo_stage2_mode", "auto"),
        help=(
            "Level 2 Stage 2 mode: auto, binary (GNB vs non-GNB), "
            "or multiclass. Recall-threshold optimisation applies only to binary mode."
        ),
    )

    # ------------------------------------------------------------------
    # Level 3 configuration
    # ------------------------------------------------------------------
    l3_group = parser.add_argument_group("Level 3 resistance configuration")

    l3_group.add_argument(
        "--skip-l3-gate", action="store_true",
        default=False,
        help="Skip the Level 3 resistance gate (force direct or true_positive_only modes).",
    )
    l3_group.add_argument(
        "--resistance-mode", type=str,
        choices=["auto", "resistance_gate", "direct", "true_positive_only"],
        default=tcfg.get("resistance_mode", "auto"),
        help=(
            "Population used for Level 3 resistance modelling. "
            "auto, resistance_gate, direct, or true_positive_only. "
            "Note: 'true_positive_only' is an oracle/hard-filter mode that uses "
            "true Level-2 positive rows only (no gate is trained) and is intended "
            "for benchmarking or diagnostic experiments."
        ),
    )
    # Alias for the cef/resistance target to match Level-2 naming parity.
    l3_group.add_argument(
        "--resistance-target", type=str, dest="cef_target",
        default=targets["cef_target"],
        help=(
            "Column name for the Level 3 resistance target (alias for --cef-target)."
        ),
    )
    l3_group.add_argument(
        "--level3-focus", nargs="+", default=tcfg.get("level3_focus", []),
        help="Optional foco allowlist for Level 3 resistance.",
    )

    # ------------------------------------------------------------------
    # Model and training
    # ------------------------------------------------------------------
    training_group = parser.add_argument_group("Model and training")

    training_group.add_argument(
        "--model-type", type=str, choices=["xgb", "lgbm", "rf", "catb"],
        default=tcfg["model_type"], help="Model backend.",
    )
    training_group.add_argument(
        "--binary-trials", "-btrials", type=_positive_int,
        default=tcfg["binary_trials"], help="Number of Optuna trials for binary models.",
    )
    training_group.add_argument(
        "--cv-splits", type=_positive_int, default=tcfg["cv_splits"],
        help="Number of cross-validation splits.",
    )
    training_group.add_argument(
        "--test-size", "-tsize", type=_proportion, default=tcfg["test_size"],
        help="Test-set proportion.",
    )
    training_group.add_argument(
        "--random-state", type=int, default=tcfg["random_state"],
        help="Random seed.",
    )
    # ------------------------------------------------------------------
    # Target-specific positive-class recall optimisation
    # ------------------------------------------------------------------
    recall_group = parser.add_argument_group(
        "Target-specific positive-class recall optimisation"
    )

    recall_group.add_argument(
        "--l1-min-recall", type=_proportion, default=None,
        metavar="RECALL",
        help=(
            "Optional Level 1 positive-class recall target. "
            "Threshold is selected from calibrated OOF probabilities."
        ),
    )
    recall_group.add_argument(
        "--l2-gate-min-recall", "--set-gate-recall",
        dest="l2_gate_min_recall", type=_proportion,
        default=tcfg.get("l2_gate_min_recall", tcfg.get("set_gate_recall")),
        metavar="RECALL",
        help=(
            "Optional Level 2 gate positive-class recall target (culture-positive)."
        ),
    )
    recall_group.add_argument(
        "--l2-etiology-min-recall", type=_proportion, default=None,
        metavar="RECALL",
        help=(
            "Optional Level 2 binary etiology positive-class recall target "
            "(e.g. GNB). Ignored for multiclass Stage 2."
        ),
    )
    recall_group.add_argument(
        "--l3-gate-min-recall", "--set-resistance-gate-recall",
        dest="l3_gate_min_recall", type=_proportion,
        default=tcfg.get(
            "l3_gate_min_recall", tcfg.get("set_resistance_gate_recall")
        ),
        metavar="RECALL",
        help=(
            "Optional Level 3 culture-positive gate recall target. "
            "This controls the first gate in resistance_gate mode."
        ),
    )
    recall_group.add_argument(
        "--l3-cef-min-recall", type=_proportion, default=None,
        metavar="RECALL",
        help=(
            "Optional Level 3 cefalosporin-resistance positive-class recall target. "
            "This specifically targets RESIST_CEFALOSPORINAS_3a_4a, not NEGATIVE."
        ),
    )

    # Backwards-compatible global option. Per-level options always take precedence.
    recall_group.add_argument(
        "--optimize-recall", action="store_true", default=False,
        help=(
            "Legacy global recall optimisation. Applies min_recall to every binary "
            "level that does not have its own --*-min-recall setting."
        ),
    )
    recall_group.add_argument(
        "--min-recall", type=_proportion, default=None, metavar="RECALL",
        help="Legacy global positive-class recall target; defaults to 0.80 when --optimize-recall is used.",
    )
    recall_group.add_argument(
        "--optimize-recall-at-specificity", type=_proportion, default=None,
        metavar="SPECIFICITY",
        help=(
            "Legacy alternative threshold strategy: maximise positive-class recall "
            "while maintaining at least this specificity. Per-level min-recall takes precedence."
        ),
    )

    # ------------------------------------------------------------------
    # Feature preprocessing and selection
    # ------------------------------------------------------------------
    feature_group = parser.add_argument_group("Feature preprocessing and selection")

    feature_group.add_argument(
        "--na-perc-limit", "-na", type=_proportion,
        default=tcfg["na_perc_limit"],
        help="Maximum allowed missingness proportion before dropping a feature.",
    )
    feature_group.add_argument(
        "--max-features", type=_positive_int, default=tcfg["max_features"],
        help="Maximum number of selected features per model.",
    )
    feature_group.add_argument(
        "--max-corr", type=_proportion, default=tcfg["max_corr"],
        help="Maximum allowed absolute feature correlation.",
    )
    feature_group.add_argument(
        "--no-impute", dest="impute_missing", action="store_false",
        default=bool(tcfg["impute_missing"]),
        help="Disable missing-value imputation.",
    )
    feature_group.add_argument(
        "--skip-rfecv", action="store_true",
        default=bool(tcfg.get("skip_rfecv", False)),
        help="Skip SHAP RFECV and use all features after preprocessing.",
    )

    # ------------------------------------------------------------------
    # Optuna
    # ------------------------------------------------------------------
    optuna_group = parser.add_argument_group("Optuna")

    optuna_group.add_argument(
        "--optuna-storage", type=str, default=tcfg["optuna_storage"],
        help="Optuna storage URI.",
    )
    optuna_group.add_argument(
        "--optuna-study-prefix", type=str, default=tcfg["optuna_study_prefix"],
        help="Prefix for Optuna study names.",
    )
    optuna_group.add_argument(
        "--optuna-load-if-exists", action="store_true",
        default=bool(tcfg.get("optuna_load_if_exists", False)),
        help="Reuse existing Optuna studies if present.",
    )
    optuna_group.add_argument(
        "--skip-optuna", action="store_true", default=False,
        help="Disable Optuna and fit with fixed model defaults.",
    )

    # ------------------------------------------------------------------
    # Stage skipping
    # ------------------------------------------------------------------
    skip_group = parser.add_argument_group("Stage skipping")
    skip_group.add_argument("--skip-level1", action="store_true", default=False,
                            help="Skip Level 1 sepsis training/prediction.")
    skip_group.add_argument("--skip-level2", action="store_true", default=False,
                            help="Skip Level 2 etiology training/prediction.")
    skip_group.add_argument("--skip-level3", action="store_true", default=False,
                            help="Skip Level 3 resistance training/prediction.")

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------
    artifact_group = parser.add_argument_group("Artifacts")
    artifact_group.add_argument(
        "--skip-shap-values", action="store_true",
        default=bool(tcfg.get("skip_shap_values", False)),
        help="Skip saving SHAP value artifacts.",
    )

    return parser


def main() -> None:
    from .modelling import run_training

    start = perf_counter()
    parser = build_arg_parser()
    args = parser.parse_args()
    run_training(args)
    print(f"\nElapsed: {(perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
