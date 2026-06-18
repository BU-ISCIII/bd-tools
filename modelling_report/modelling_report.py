#!/usr/bin/env python3
"""Generate markdown reports for modelling analysis folders."""

from __future__ import annotations

from dataclasses import dataclass
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
    if params.get("HEMO_TARGET"):
        lines.append(f"- Level 2 etiology target: `{params['HEMO_TARGET']}`")
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


def make_report(folder: Path, output: Path | None = None) -> Path:
    folder = folder.resolve()
    ctx = collect_report_context(folder)
    report_text = render_report(ctx)
    output_path = output.resolve() if output else ctx.output
    output_path.write_text(report_text, encoding="utf-8")
    return output_path


def main(argv: list[str]) -> int:
    if len(argv) not in {1, 2, 3}:
        print("Usage: modelling_report.py [folder] [output]", file=sys.stderr)
        return 1

    folder = Path(argv[1]) if len(argv) >= 2 else Path(".")
    output = Path(argv[2]) if len(argv) == 3 else None

    if folder.is_file():
        print(f"Folder path expected, got file: {folder}", file=sys.stderr)
        return 1

    output_path = make_report(folder, output)
    print(f"[OK] Wrote report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
