#!/usr/bin/env python3
"""Generate markdown reports for modelling analysis folders."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import sys
from typing import Any


SCALAR_TYPES = (str, int, float, bool)
IMPORTANT_KEYS = [
    "subset",
    "subset_output_dir",
    "output_dir",
    "binary_model",
    "multiclass_model",
    "binary_feature_count",
    "selected_feature_count",
    "rfecv_step",
    "rfecv_scoring",
    "binary_trials",
    "macro_f1",
    "binary_roc_auc",
    "binary_threshold",
    "threshold",
    "mode",
    "target",
    "target_variable",
]
NESTED_KEYS = [
    "subset_summary",
    "binary",
    "multiclass",
    "gate",
    "metrics",
    "params",
    "binary_params",
    "multiclass_params",
    "binary_gate_params",
    "stage1_gate",
    "stage2_subtype",
]


@dataclass
class ReportContext:
    folder: Path
    output: Path
    params: dict[str, str]
    sbatch_files: list[Path]
    result_jsons: list[Path]
    plot_files: list[Path]
    legacy_readme: Path | None
    script_files: list[Path]


@dataclass
class ResultsContext:
    folder: Path
    output: Path
    training_dirs: list[Path]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - best effort report generation
        return f"ERROR READING FILE: {exc}"


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_params(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def find_files(folder: Path, pattern: str) -> list[Path]:
    return sorted(path for path in folder.glob(pattern) if path.is_file())


def find_training_dirs(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_dir() and path.name.startswith("01-training"))


def collect_report_context(folder: Path) -> ReportContext:
    params_path = folder / "training.params"
    params = parse_params(params_path)
    sbatch_files = find_files(folder, "*.sbatch")
    result_jsons = sorted(path for path in (folder / "results").rglob("*.json") if path.is_file()) if (folder / "results").exists() else []
    plot_files = sorted(
        path
        for path in (folder / "graphs").rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
    ) if (folder / "graphs").exists() else []
    legacy_readme = folder / "README" if (folder / "README").exists() else None
    script_files = [path for path in [folder / "_01_run_model.sh", folder / "_02_plot_results.sh", folder / "_03_report.sh"] if path.exists()]
    return ReportContext(
        folder=folder,
        output=folder / "README.md",
        params=params,
        sbatch_files=sbatch_files,
        result_jsons=result_jsons,
        plot_files=plot_files,
        legacy_readme=legacy_readme,
        script_files=script_files,
    )


def collect_results_context(folder: Path) -> ResultsContext:
    return ResultsContext(
        folder=folder,
        output=folder / "RESULTS.md",
        training_dirs=find_training_dirs(folder),
    )


def md_code_block(language: str, content: str) -> str:
    return f"```{language}\n{content.rstrip()}\n```" if content.strip() else f"```{language}\n```"


def format_list(items: list[str]) -> str:
    if not items:
        return "- None"
    return "\n".join(f"- {item}" for item in items)


def value_to_text(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (str, int)):
        return str(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        if len(value) <= 8 and all(isinstance(item, SCALAR_TYPES) or item is None for item in value):
            return json.dumps(value, ensure_ascii=False)
        return f"list[{len(value)}]"
    if isinstance(value, dict):
        return f"dict[{len(value)}]"
    return str(value)


def render_json_value(name: str, value: Any, indent: int = 0, depth: int = 0, max_depth: int = 2) -> list[str]:
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(value, dict):
        if depth >= max_depth:
            lines.append(f"{pad}- {name}: dict[{len(value)}]")
            return lines
        lines.append(f"{pad}- {name}:")
        for sub_key, sub_value in value.items():
            lines.extend(render_json_value(str(sub_key), sub_value, indent + 1, depth + 1, max_depth))
        return lines
    if isinstance(value, list):
        if not value:
            lines.append(f"{pad}- {name}: []")
        elif len(value) <= 8 and all(isinstance(item, SCALAR_TYPES) or item is None for item in value):
            lines.append(f"{pad}- {name}: {json.dumps(value, ensure_ascii=False)}")
        else:
            lines.append(f"{pad}- {name}: list[{len(value)}]")
        return lines
    lines.append(f"{pad}- {name}: {value_to_text(value)}")
    return lines


def summarize_json_file(path: Path) -> str:
    payload = read_json(path)
    if payload is None:
        return "- Failed to parse JSON."
    lines = [f"### `{path.relative_to(path.parents[1])}`"]
    if isinstance(payload, dict):
        for key in IMPORTANT_KEYS:
            if key in payload:
                lines.extend(render_json_value(key, payload[key]))
        for key in NESTED_KEYS:
            if key in payload and isinstance(payload[key], (dict, list)):
                lines.extend(render_json_value(key, payload[key]))
        extra_keys = [key for key in payload.keys() if key not in IMPORTANT_KEYS and key not in NESTED_KEYS]
        for key in extra_keys[:20]:
            value = payload[key]
            if isinstance(value, (dict, list)) or isinstance(value, SCALAR_TYPES) or value is None:
                lines.extend(render_json_value(key, value))
    elif isinstance(payload, list):
        lines.append(f"- JSON array with {len(payload)} elements")
    else:
        lines.append(f"- JSON root type: {type(payload).__name__}")
    return "\n".join(lines)


def sbatch_command_body(path: Path) -> str:
    text = read_text(path)
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("CMD=("):
            command_lines: list[str] = []
            for command_line in lines[index + 1 :]:
                if command_line.strip() == ")":
                    break
                command_lines.append(command_line)
            return "\n".join(command_lines).rstrip()
    for index, line in enumerate(lines):
        if line.strip().startswith("python3 "):
            return "\n".join(lines[index:]).rstrip()
    return text.strip()


def sbatch_directives(path: Path) -> str:
    text = read_text(path)
    directives = [line for line in text.splitlines() if line.startswith("#SBATCH")]
    return "\n".join(directives)


def target_summary(params: dict[str, str]) -> list[str]:
    lines: list[str] = []
    model_type = params.get("MODEL_TYPE")
    if model_type:
        lines.append(f"- Model type: `{model_type}`")
    subpipeline = params.get("SUBPIPELINE")
    if subpipeline:
        lines.append(f"- Script family: `{subpipeline}`")
    if params.get("SEPSIS_TARGET"):
        lines.append(f"- Level 1 target: `{params['SEPSIS_TARGET']}`")
    if params.get("HEMO_GATE_TARGET"):
        lines.append(f"- Level 2 gate target: `{params['HEMO_GATE_TARGET']}`")
    if params.get("HEMO_SET_GATE_RECALL"):
        lines.append(f"- Level 2 gate recall target: `{params['HEMO_SET_GATE_RECALL']}`")
    if params.get("HEMO_TARGET"):
        hemo_line = f"- Level 2 etiology target: `{params['HEMO_TARGET']}`"
        if params.get("HEMO_STAGE2_MODE"):
            hemo_line += f" (Stage 2 mode: `{params['HEMO_STAGE2_MODE']}`)"
        lines.append(hemo_line)
    elif params.get("HEMO_STAGE2_MODE"):
        lines.append(f"- Level 2 Stage 2 mode: `{params['HEMO_STAGE2_MODE']}`")
    if params.get("RESISTENCIA_TARGET"):
        lines.append(f"- Level 3 resistance target: `{params['RESISTENCIA_TARGET']}`")
    if params.get("BMR_TARGET"):
        lines.append(f"- Level 3 gate/BMR target: `{params['BMR_TARGET']}`")

    run_levels = []
    for level in ("RUN_LEVEL1", "RUN_LEVEL2", "RUN_LEVEL3"):
        if level in params:
            run_levels.append(f"{level.lower()}={params[level]}")
    if run_levels:
        lines.append(f"- Enabled stages: `{', '.join(run_levels)}`")
    return lines


def parameter_summary(params: dict[str, str]) -> list[str]:
    ordered_keys = [
        "INPUT_DATA",
        "OUTPUT_DIR",
        "SCRIPT_NAME",
        "MODEL_TYPE",
        "BINARY_TRIALS",
        "CV_SPLITS",
        "TEST_SIZE",
        "RANDOM_STATE",
        "SEED_LIST",
        "NA_PERC_LIMIT",
        "MAX_FEATURES",
        "MAX_CORR",
        "HEMO_STAGE2_MODE",
        "HEMO_SET_GATE_RECALL",
        "SKIP_RFECV",
        "SKIP_OPTUNA",
        "OPTUNA_STORAGE",
        "OPTUNA_STUDY_PREFIX",
        "OPTUNA_LOAD_IF_EXISTS",
        "WEIGHT_COLUMN",
    ]
    lines: list[str] = []
    for key in ordered_keys:
        value = params.get(key)
        if value:
            lines.append(f"- `{key}`: `{value}`")
    return lines


def command_list(params: dict[str, str]) -> list[str]:
    commands = [
        "_01_run_model.sh -> submits the training job",
        "_02_plot_results.sh -> builds plots and figures",
        "_03_report.sh -> generates this markdown report",
    ]
    script_name = params.get("SCRIPT_NAME")
    if script_name:
        commands.append(f"Underlying model script: `{script_name}`")
    return commands


def format_metric(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    if value is None:
        return "NA"
    return str(value)


def link_image(base: Path, image: Path, title: str | None = None) -> str:
    rel = image.relative_to(base)
    alt = title or image.stem.replace("_", " ")
    return f"![{alt}]({rel.as_posix()})"


def list_image_paths(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
    )


def render_image_gallery(base: Path, images: list[Path]) -> list[str]:
    lines: list[str] = []
    for image in images:
        title = image.stem.replace("_", " ")
        lines.extend([f"#### `{image.name}`", "", link_image(base, image, title), ""])
    return lines


def render_metrics_block(title: str, metrics: list[tuple[str, Any]]) -> list[str]:
    lines = [f"#### {title}", ""]
    for key, value in metrics:
        lines.append(f"- {key}: `{format_metric(value)}`")
    lines.append("")
    return lines


def render_level1_section(base: Path, run_dir: Path, level_dir: Path) -> list[str]:
    summary_path = level_dir / "summary.json"
    data = read_json(summary_path) or {}
    images = list_image_paths(run_dir / "plots" / level_dir.name)

    lines = [f"### Level 1: `{level_dir.name}`", ""]
    if isinstance(data, dict):
        lines.extend(render_metrics_block(
            "Summary",
            [
                ("target", data.get("target")),
                ("macro F1", data.get("macro_f1")),
                ("ROC AUC", data.get("roc_auc")),
                ("PR AUC", data.get("pr_auc")),
                ("precision", data.get("precision")),
                ("recall", data.get("recall")),
                ("threshold", data.get("threshold")),
                ("classes", data.get("classes")),
                ("selected features", len(data.get("features", [])) if isinstance(data.get("features"), list) else data.get("features")),
            ],
        ))
    lines.extend(render_image_gallery(base, images))
    return lines


def render_level2_section(base: Path, run_dir: Path, level_dir: Path) -> list[str]:
    summary_path = level_dir / "summary.json"
    data = read_json(summary_path) or {}
    images = list_image_paths(run_dir / "plots" / level_dir.name)

    lines = [f"### Level 2: `{level_dir.name}`", ""]
    if isinstance(data, dict):
        lines.extend(render_metrics_block(
            "Run Summary",
            [
                ("mode", data.get("mode")),
                ("stage2 mode", data.get("stage2_mode") or (data.get("stage2_subtype", {}) if isinstance(data.get("stage2_subtype"), dict) else {}).get("stage2_mode")),
            ],
        ))
        stage1 = data.get("stage1_gate", {}) if isinstance(data.get("stage1_gate"), dict) else {}
        lines.extend(render_metrics_block(
            "Stage 1 Gate",
            [
                ("macro F1", stage1.get("macro_f1")),
                ("ROC AUC", stage1.get("roc_auc")),
                ("PR AUC", stage1.get("pr_auc")),
                ("precision", stage1.get("precision")),
                ("recall", stage1.get("recall")),
                ("threshold", stage1.get("threshold")),
                ("negative label", stage1.get("negative_label")),
                ("positive label", stage1.get("positive_label")),
            ],
        ))
        stage2 = data.get("stage2_subtype", {}) if isinstance(data.get("stage2_subtype"), dict) else {}
        lines.extend(render_metrics_block(
            "Stage 2 Subtype",
            [
                ("macro F1", stage2.get("macro_f1")),
                ("ROC AUC", stage2.get("roc_auc")),
                ("PR AUC", stage2.get("pr_auc")),
                ("precision", stage2.get("precision")),
                ("recall", stage2.get("recall")),
                ("stage2 mode", stage2.get("stage2_mode")),
                ("classes", stage2.get("classes")),
            ],
        ))
    lines.extend(render_image_gallery(base, images))
    return lines


def render_level3_section(base: Path, run_dir: Path, level_dir: Path) -> list[str]:
    summary_path = level_dir / "summary.json"
    data = read_json(summary_path) or {}
    images = list_image_paths(run_dir / "plots" / level_dir.name)

    lines = [f"### Level 3: `{level_dir.name}`", ""]
    if isinstance(data, dict):
        lines.extend(render_metrics_block(
            "Summary",
            [
                ("mode", data.get("mode")),
                ("target", data.get("target")),
                ("macro F1", data.get("macro_f1")),
                ("ROC AUC", data.get("roc_auc")),
                ("PR AUC", data.get("pr_auc")),
                ("precision", data.get("precision")),
                ("recall", data.get("recall")),
                ("threshold", data.get("threshold")),
                ("classes", data.get("classes")),
                ("selected features", len(data.get("features", [])) if isinstance(data.get("features"), list) else data.get("features")),
            ],
        ))
    lines.extend(render_image_gallery(base, images))
    return lines


def render_processing_section(base: Path, run_dir: Path) -> list[str]:
    processing_dir = run_dir / "processing"
    images = list_image_paths(run_dir / "plots" / "processing")
    ordered_groups = [
        ("QC", [
            "processing_imputed_features.png",
        ]),
        ("Dropping", [
            "processing_nan_dropped_features.png",
            "processing_correlation_dropped_features.png",
        ]),
        ("RFECV", [
            "processing_l1_rfecv_features_performance.png",
            "processing_l2_gate_rfecv_features_performance.png",
            "processing_l2_sub_rfecv_features_performance.png",
            "processing_l3_rfecv_features_performance.png",
        ]),
        ("IQR outliers", [
            "processing_iqr_outliers_train.png",
            "processing_iqr_outliers_test.png",
        ]),
    ]

    image_by_name = {image.name: image for image in images}
    lines = ["## Processing", ""]
    if not images:
        lines.append("- No processing plots found.")
        lines.append("")
        return lines

    if processing_dir.exists():
        lines.append(f"- Processing artifacts: `{processing_dir.relative_to(base)}`")
        lines.append("")

    for group_name, filenames in ordered_groups:
        group_images = [image_by_name[name] for name in filenames if name in image_by_name]
        if not group_images:
            continue
        lines.extend([f"### {group_name}", ""])
        lines.extend(render_image_gallery(base, group_images))

    remaining_names = {name for _, group in ordered_groups for name in group}
    remaining = [image for image in images if image.name not in remaining_names]
    if remaining:
        lines.extend(["### Other", ""])
        lines.extend(render_image_gallery(base, remaining))
    return lines


def render_results_report(ctx: ResultsContext) -> str:
    lines: list[str] = [
        "# Results",
        "",
        f"- Folder: `{ctx.folder.name}`",
        f"- Path: `{ctx.folder.resolve()}`",
        f"- Generated: `{datetime.now().astimezone().isoformat(timespec='seconds')}`",
        "",
    ]

    if not ctx.training_dirs:
        lines.extend(["- No `01-training*` directories found.", ""])
        return "\n".join(lines)

    for run_dir in ctx.training_dirs:
        lines.extend([
            f"## Training Run: `{run_dir.name}`",
            "",
        ])
        final_report = run_dir / "final_report.json"
        aggregate_report = run_dir / "aggregate_summary.json"
        report_data = read_json(final_report) or read_json(aggregate_report) or {}
        if isinstance(report_data, dict) and "args" in report_data:
            args = report_data.get("args", {}) if isinstance(report_data.get("args"), dict) else {}
            lines.extend(render_metrics_block(
                "Run Parameters",
                [
                    ("model type", args.get("model_type")),
                    ("sepsis target", args.get("sepsis_target")),
                    ("hemo gate target", args.get("hemo_gate_target")),
                    ("hemo target", args.get("hemo_target")),
                    ("stage 2 mode", args.get("hemo_stage2_mode")),
                    ("BMR target", args.get("bmr_target")),
                    ("cef target", args.get("cef_target")),
                    ("max features", args.get("max_features")),
                    ("skip RFECV", args.get("skip_rfecv")),
                ],
            ))

        lines.extend(render_processing_section(ctx.folder, run_dir))

        level1_dir = run_dir / "level1_sepsis"
        level2_dir = run_dir / "level2_hemo"
        level3_dir = run_dir / "level3_cefalosporina"
        if level1_dir.exists():
            lines.extend(render_level1_section(ctx.folder, run_dir, level1_dir))
        if level2_dir.exists():
            lines.extend(render_level2_section(ctx.folder, run_dir, level2_dir))
        if level3_dir.exists():
            lines.extend(render_level3_section(ctx.folder, run_dir, level3_dir))

    return "\n".join(lines)


def render_report(ctx: ReportContext) -> str:
    lines: list[str] = []
    folder_name = ctx.folder.name

    lines.extend([
        "# Analysis Report",
        "",
        f"- Folder: `{folder_name}`",
        f"- Path: `{ctx.folder.resolve()}`",
        f"- Generated: `{__import__('datetime').datetime.now().astimezone().isoformat(timespec='seconds')}`",
        "",
        "## Run Summary",
        "",
    ])
    lines.extend(target_summary(ctx.params) or ["- No run metadata found in `training.params`."])
    lines.append("")

    lines.extend(["## Parameters", ""])
    lines.extend(parameter_summary(ctx.params) or ["- No `training.params` file found."])
    lines.append("")

    lines.extend(["## Launch Scripts", ""])
    lines.extend(format_list(command_list(ctx.params)).splitlines())
    lines.append("")

    if ctx.legacy_readme is not None:
        lines.extend(["## Legacy README", "", md_code_block("text", read_text(ctx.legacy_readme)), ""])

    lines.extend(["## Scheduler Files", ""])
    if not ctx.sbatch_files:
        lines.append("- No `.sbatch` files found.")
    else:
        for sbatch_file in ctx.sbatch_files:
            lines.extend([
                f"### `{sbatch_file.name}`",
                "",
                "#### SBATCH Directives",
                "",
                md_code_block("bash", sbatch_directives(sbatch_file)),
                "",
                "#### Command Body",
                "",
                md_code_block("bash", sbatch_command_body(sbatch_file)),
                "",
            ])

    lines.extend(["## Results", ""])
    if not ctx.result_jsons:
        lines.append("- No JSON result files found.")
    else:
        for result_json in ctx.result_jsons:
            lines.append(summarize_json_file(result_json))
            lines.append("")

    lines.extend(["## Plots", ""])
    if not ctx.plot_files:
        lines.append("- No plot files found yet.")
    else:
        for plot_file in ctx.plot_files:
            lines.append(f"- `{plot_file.relative_to(ctx.folder)}`")
    lines.append("")

    lines.extend(["## Artifact Tree", ""])
    tree_lines = []
    try:
        import subprocess

        result = subprocess.run(
            ["tree", "-a", "-L", "3"],
            cwd=ctx.folder,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            tree_lines = result.stdout.splitlines()
    except Exception:
        tree_lines = []

    if tree_lines:
        lines.append(md_code_block("text", "\n".join(tree_lines)))
    else:
        fallback = sorted(str(path.relative_to(ctx.folder)) for path in ctx.folder.rglob("*") if path != ctx.output)
        lines.append(md_code_block("text", "\n".join(fallback)))

    lines.append("")
    return "\n".join(lines)


def write_report_pair(folder: Path, output: Path | None = None) -> tuple[Path, Path | None]:
    folder = folder.resolve()
    report_ctx = collect_report_context(folder)
    results_ctx = collect_results_context(folder)

    results_path = output.resolve() if output and output.name == results_ctx.output.name else results_ctx.output
    report_path = output.resolve() if output and output.name != results_ctx.output.name else report_ctx.output

    report_path.write_text(render_report(report_ctx), encoding="utf-8")
    results_path.write_text(render_results_report(results_ctx), encoding="utf-8")

    if report_path == results_path:
        return report_path, None
    return report_path, results_path


def make_report(folder: Path, output: Path | None = None) -> Path:
    report_path, _ = write_report_pair(folder, output)
    return report_path


def main(argv: list[str]) -> int:
    if len(argv) not in {1, 2, 3}:
        print("Usage: modelling_report.py [folder] [output]", file=sys.stderr)
        return 1

    folder = Path(argv[1]) if len(argv) >= 2 else Path(".")
    output = Path(argv[2]) if len(argv) == 3 else None

    if folder.is_file():
        print(f"Folder path expected, got file: {folder}", file=sys.stderr)
        return 1

    output_path, results_path = write_report_pair(folder, output)
    print(f"[OK] Wrote report: {output_path}")
    if results_path is not None:
        print(f"[OK] Wrote results report: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
