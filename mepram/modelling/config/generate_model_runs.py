#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import stat
import sys
from pathlib import Path
from typing import Any
import textwrap


SCOPES = ("sepsis", "etiology", "resistance")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ConfigError(ValueError):
    pass


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate MEPRAM SLURM jobs from JSON."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("model_runs.json"),
    )
    p.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.cwd(),
    )
    p.add_argument(
        "--scope",
        choices=("all",) + SCOPES,
        default="all",
    )
    p.add_argument(
        "--input-data",
        type=Path,
    )
    p.add_argument(
        "--overwrite-generated",
        action="store_true",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
    )
    return p.parse_args()


def q(v: Any) -> str:
    """Shell-quote a value using single quotes."""
    return "'" + str(v).replace("'", "'\"'\"'") + "'"


def executable(path: Path):
    path.chmod(
        path.stat().st_mode
        | stat.S_IXUSR
        | stat.S_IXGRP
        | stat.S_IXOTH
    )


def write(path: Path, text: str, overwrite: bool):
    if path.exists() and not overwrite:
        raise ConfigError(
            f"Generated file exists: {path}; use --overwrite-generated"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def merge(defaults: dict, override: dict) -> dict:
    out = dict(defaults)
    out.update(override)

    out["slurm"] = {
        **defaults.get("slurm", {}),
        **override.get("slurm", {}),
    }

    return out


def mounted(host: Path, bind: Path, mnt: Path) -> Path:
    try:
        return mnt / host.resolve().relative_to(bind.resolve())
    except ValueError as e:
        raise ConfigError(
            f"{host} is outside bind path {bind}"
        ) from e


def columns(csv_path: Path) -> set[str]:
    with csv_path.open(newline="") as fh:
        try:
            return {
                x.strip().strip('"')
                for x in next(csv.reader(fh))
            }
        except StopIteration as e:
            raise ConfigError(
                f"Empty CSV: {csv_path}"
            ) from e


def required(scope: str, r: dict) -> set[str]:
    req = {r["weight_column"]}

    if scope == "sepsis":
        req.add(r["sepsis_target"])

    elif scope == "etiology":
        req |= {
            r["sepsis_target"],
            r["etiology_target"],
        }

    if not r.get("skip_l2_gate", False):
        req.add(r["etiology_gate_target"])
    else:
        req.add(r["resistance_target"])

    return req


def validate(scope: str, r: dict):
    name = r.get("name")

    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise ConfigError(
            f"Invalid run name: {name!r}"
        )

    if not isinstance(r.get("enabled", True), bool):
        raise ConfigError(
            f"{name}: enabled must be boolean"
        )

    if scope == "etiology":
        if r["etiology_stage2_mode"] not in {
            "auto",
            "binary",
            "multiclass",
        }:
            raise ConfigError(
                f"{name}: invalid etiology_stage2_mode"
            )

        recall = float(
            r.get("min_recall_gate", 0.0)
        )

        if not 0 <= recall <= 1:
            raise ConfigError(
                f"{name}: min_recall_gate must be 0..1"
            )

        focus = r.get("level2_focus", [])

        if (
            not isinstance(focus, list)
            or not all(
                isinstance(x, str) and x
                for x in focus
            )
        ):
            raise ConfigError(
                f"{name}: level2_focus must be a string list"
            )

    if scope == "resistance":
        if r["resistance_mode"] not in {
            "auto",
            "direct",
            "resistance_gate",
            "true_positive_only",
        }:
            raise ConfigError(
                f"{name}: invalid resistance_mode"
            )


def per_run_study_prefix(
    template: str,
    *,
    run: str,
    output: str,
    model_type: str,
) -> str:
    """
    Return a study prefix that cannot collide with
    another configured run.
    """

    values = {
        "run": run,
        "output": output,
        "model_type": model_type,
    }

    try:
        prefix = str(template).format(**values).strip()
    except KeyError as e:
        raise ConfigError(
            f"Unknown study_prefix placeholder: {e.args[0]}"
        ) from e

    if not prefix:
        raise ConfigError(
            f"{run}: study_prefix cannot be empty"
        )

    if (
        "{run}" not in str(template)
        and "{output}" not in str(template)
    ):
        prefix = f"{prefix}_{run}"

    return prefix


def per_run_optuna_database(
    template: str,
    *,
    run: str,
    output: str,
    model_type: str,
) -> Path:
    """
    Resolve an isolated SQLite database path for one
    generated run.

    Existing scope-level filenames such as
    ``optuna_studies/etiology_optuna.db`` are treated as
    directory/suffix defaults; the output name replaces
    the shared filename.

    Templates may explicitly use {run}, {output},
    or {model_type}.
    """

    raw_template = (
        str(template).strip()
        or "optuna_studies/{output}.db"
    )

    values = {
        "run": run,
        "output": output,
        "model_type": model_type,
    }

    try:
        if "{" in raw_template:
            resolved = Path(
                raw_template.format(**values)
            )
        else:
            configured = Path(raw_template)

            suffix = configured.suffix or ".db"

            parent = (
                configured.parent
                if str(configured.parent) != "."
                else Path("optuna_studies")
            )

            resolved = parent / f"{output}{suffix}"

    except KeyError as e:
        raise ConfigError(
            f"Unknown optuna_database placeholder: {e.args[0]}"
        ) from e

    if resolved.is_absolute() or ".." in resolved.parts:
        raise ConfigError(
            f"Invalid optuna_database: {resolved}"
        )

    return resolved


def command(
    scope: str,
    r: dict,
    input_mnt: Path,
    output_mnt: Path,
    storage: str,
) -> list[str]:

    c = [
        "python3",
        "-m",
        "mepram.modelling.cli",
        "-db",
        str(input_mnt),
        "-o",
        str(output_mnt),

        "--model-type",
        str(r["model_type"]),

        "--binary-trials",
        str(r["binary_trials"]),

        "--cv-splits",
        str(r["cv_splits"]),

        "--test-size",
        str(r["test_size"]),

        "--random-state",
        str(r["random_state"]),

        "--na-perc-limit",
        str(r["na_perc_limit"]),

        "--max-features",
        str(r["max_features"]),

        "--max-corr",
        str(r["max_corr"]),

        "--weight-column",
        str(r["weight_column"]),

        "--optuna-storage",
        storage,

        "--optuna-study-prefix",
        str(r["study_prefix"]),
    ]

    if scope == "sepsis":

        c += [
            "--sepsis-target",
            r["sepsis_target"],
            "--skip-level2",
            "--skip-level3",
        ]

    elif scope == "etiology":

        c += [
            "--etiology-target",
            r["etiology_target"],

            "--etiology-gate-target",
            r["etiology_gate_target"],

            "--etiology-stage2-mode",
            r["etiology_stage2_mode"],

            "--cef-target",
            r["resistance_target"],

            "--resistance-mode",
            r["resistance_mode"],

            "--skip-level1",
            "--skip-level3",
        ]

    else:

        c += [
            "--etiology-target",
            r["etiology_target"],

            "--etiology-gate-target",
            r["etiology_gate_target"],

            "--etiology-stage2-mode",
            r["etiology_stage2_mode"],

            "--cef-target",
            r["resistance_target"],

            "--resistance-mode",
            r["resistance_mode"],

            "--skip-level1",
            "--skip-level2",
        ]

    if r.get("skip_l2_gate", False):
        c += ["--skip-l2-gate"]

    if float(r.get("min_recall_gate", 0)) > 0:
        c += [
            "--set-gate-recall",
            str(r["min_recall_gate"]),
        ]

    if r.get("l2_gate_proba_as_feature", False):
        c += [
            "--l2-gate-proba-as-feature"
        ]

    if r.get("level2_focus", []):
        c += [
            "--level2-focus",
            *r["level2_focus"],
        ]

    if r.get("optuna_load_if_exists", True):
        c += [
            "--optuna-load-if-exists"
        ]

    if r.get("skip_rfecv", False):
        c += [
            "--skip-rfecv"
        ]

    if r.get("skip_optuna", False):
        c += [
            "--skip-optuna"
        ]

    return c


def sbatch(
    scope: str,
    r: dict,
    cmd: list[str],
    bind: Path,
    mnt: Path,
    image: Path,
    pwd: Path,
) -> str:

    s = r["slurm"]

    args = "\n".join(
        f"    {q(x)}"
        for x in cmd
    )

    return f'''#!/bin/bash
#SBATCH --job-name={scope}_{r["name"]}
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time={s["time"]}
#SBATCH --ntasks={s["ntasks"]}
#SBATCH --cpus-per-task={s["cpus_per_task"]}
#SBATCH --mem={s["memory"]}
#SBATCH --partition={s["partition"]}

set -euo pipefail

mkdir -p logs

module load singularity

CMD=(
{args}
)

singularity exec \\
    --env SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" \\
    --env SLURM_JOB_ID="$SLURM_JOB_ID" \\
    --containall \\
    --bind {q(bind)}:{q(mnt)} \\
    --pwd {q(pwd)} \\
    {q(image)} \\
    "${{CMD[@]}}"
'''


def launcher(jobs: list[dict]) -> str:

    jf = "\n".join(
        f"  [{q(j['name'])}]={q(j['sbatch'])}"
        for j in jobs
    )

    od = "\n".join(
        f"  [{q(j['name'])}]={q(j['output'])}"
        for j in jobs
    )

    return f'''#!/bin/bash

set -euo pipefail

cd "$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

declare -A JOB_FILES=(
{jf}
)

declare -A OUTPUT_DIRS=(
{od}
)

ONLY=""
OVERWRITE=false
DRY=false
LIST=false

while [[ $# -gt 0 ]]; do

    case "$1" in

        --only)
            ONLY="${{2:?--only requires NAME}}"
            shift 2
            ;;

        --overwrite-results)
            OVERWRITE=true
            shift
            ;;

        --dry-run)
            DRY=true
            shift
            ;;

        --list)
            LIST=true
            shift
            ;;

        -h|--help)
            echo "Usage: $0 [--only NAME] [--overwrite-results] [--dry-run] [--list]"
            exit 0
            ;;

        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;

    esac

done

if $LIST; then
    printf '%s\\n' "${{!JOB_FILES[@]}}" | sort
    exit 0
fi

if [[ -n "$ONLY" && -z "${{JOB_FILES[$ONLY]+x}}" ]]; then
    echo "Unknown run: $ONLY" >&2
    exit 2
fi

mapfile -t NAMES < <(
    printf '%s\\n' "${{!JOB_FILES[@]}}" | sort
)

submitted=0
skipped=0

for name in "${{NAMES[@]}}"; do

    [[ -z "$ONLY" || "$name" == "$ONLY" ]] || continue

    out="${{OUTPUT_DIRS[$name]}}"
    job="${{JOB_FILES[$name]}}"

    if [[ -e "$out" ]]; then

        if $OVERWRITE; then

            echo "OVERWRITE $name: rm -rf -- $out"

            $DRY || rm -rf -- "$out"

        else

            echo "SKIP $name: $out exists"

            skipped=$((skipped + 1))

            continue

        fi

    fi

    echo "SUBMIT $name: sbatch $job"

    $DRY || sbatch "$job"

    submitted=$((submitted + 1))

done

echo "Submitted: $submitted"
echo "Skipped: $skipped"
'''


def main() -> int:

    a = parse_args()

    cfg = json.loads(
        a.config.resolve().read_text()
    )

    paths = cfg["paths"]
    defaults = cfg["defaults"]
    scopes = cfg["scopes"]

    root = a.workspace_root.resolve()

    if root.name in {
        v["directory"]
        for v in scopes.values()
    }:
        root = root.parent

    bind = Path(
        paths["bind_path"]
    ).resolve()

    mnt = Path(
        paths["mount_path"]
    )

    input_host = (
        a.input_data
        or Path(paths["input_data"])
    ).resolve()

    if not input_host.is_file():
        raise ConfigError(
            f"Input data not found: {input_host}"
        )

    cols = columns(input_host)

    chosen = (
        SCOPES
        if a.scope == "all"
        else (a.scope,)
    )

    total = 0

    for scope in chosen:

        sc = scopes[scope]

        d = root / sc["directory"]

        local_input = (
            d
            / "00-data"
            / input_host.name
        )

        jobs = []
        manifest = []

        seen = set()
        seen_databases = set()
        seen_studies = set()

        for raw in sc["runs"]:

            r = merge(
                defaults,
                raw,
            )

            validate(
                scope,
                r,
            )

            if r["name"] in seen:
                raise ConfigError(
                    f"Duplicate {scope} run: {r['name']}"
                )

            seen.add(
                r["name"]
            )

            if not r.get(
                "enabled",
                True,
            ):
                continue

            missing = sorted(
                required(
                    scope,
                    r,
                )
                - cols
            )

            if missing:
                raise ConfigError(
                    f"{scope}/{r['name']} "
                    f"missing columns: "
                    f"{', '.join(missing)}"
                )

            output = (
                r["name"]
                if r["name"].startswith("01-")
                else f"01-{r['name']}"
            )

            study_template = raw.get(
                "study_prefix",
                sc.get(
                    "study_prefix",
                    "{run}_{model_type}",
                ),
            )

            r["study_prefix"] = (
                per_run_study_prefix(
                    study_template,
                    run=r["name"],
                    output=output,
                    model_type=r["model_type"],
                )
            )

            if r["study_prefix"] in seen_studies:
                raise ConfigError(
                    f"Duplicate {scope} study prefix: "
                    f"{r['study_prefix']}"
                )

            seen_studies.add(
                r["study_prefix"]
            )

            database_template = raw.get(
                "optuna_database",
                sc.get(
                    "optuna_database",
                    "optuna_studies/{output}.db",
                ),
            )

            db_rel = (
                per_run_optuna_database(
                    database_template,
                    run=r["name"],
                    output=output,
                    model_type=r["model_type"],
                )
            )

            if db_rel in seen_databases:
                raise ConfigError(
                    f"Duplicate {scope} Optuna database: "
                    f"{db_rel}"
                )

            seen_databases.add(
                db_rel
            )

            timeout = int(
                r.get(
                    "sqlite_timeout_seconds",
                    120,
                )
            )

            if timeout < 1:
                raise ConfigError(
                    f"{r['name']}: "
                    f"sqlite_timeout_seconds "
                    f"must be >= 1"
                )

            db_host = d / db_rel

            storage = (
                "sqlite:////"
                + str(
                    mounted(
                        db_host,
                        bind,
                        mnt,
                    )
                ).lstrip("/")
                + f"?timeout={timeout}"
            )

            cmd = command(
                scope,
                r,
                mounted(
                    local_input,
                    bind,
                    mnt,
                ),
                mounted(
                    d / output,
                    bind,
                    mnt,
                ),
                storage,
            )

            rel = (
                Path("jobs")
                / f"{r['name']}.sbatch"
            )

            jobs.append(
                {
                    "name": r["name"],
                    "sbatch": str(rel),
                    "output": output,
                }
            )

            manifest.append(
                {
                    "name": r["name"],
                    "output_dir": output,
                    "sbatch": str(rel),
                    "optuna_database": str(db_rel),
                    "optuna_storage": storage,
                    "optuna_study_prefix": r[
                        "study_prefix"
                    ],
                    "command": cmd,
                    "configuration": r,
                }
            )

            if not a.validate_only:

                for x in (
                    d / "logs",
                    d / "00-data",
                    d / "optuna_studies",
                    d / "jobs",
                ):
                    x.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                db_host.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                db_host.touch(
                    exist_ok=True
                )

                # Singularity resolves the image before
                # applying bind mounts, so the .sif must
                # use its host path.
                #
                # Only paths consumed inside the container
                # (such as --pwd and CLI inputs) should be
                # translated to the mounted /mnt path.

                image_host = (
                    bind
                    / paths["singularity_image"]
                ).resolve()

                workdir_mounted = mounted(
                    bind
                    / paths["working_directory"],
                    bind,
                    mnt,
                )

                p = d / rel

                write(
                    p,
                    sbatch(
                        scope,
                        r,
                        cmd,
                        bind,
                        mnt,
                        image_host,
                        workdir_mounted,
                    ),
                    a.overwrite_generated,
                )

                executable(p)

        if not jobs:
            raise ConfigError(
                f"No enabled runs for {scope}"
            )

        if a.validate_only:
            print(
                f"VALID {scope}: "
                f"{len(jobs)} run(s)"
            )

            total += len(jobs)

            continue

        shutil.copy2(
            input_host,
            local_input,
        )

        # ------------------------------------------------------------------
        # Generated helper scripts
        # ------------------------------------------------------------------
        #
        # IMPORTANT:
        #
        # plot_results.py and report_results.py use package-relative imports
        # such as:
        #
        #     from .utils import ...
        #
        # Therefore they MUST be executed as Python modules:
        #
        #     python3 -m mepram.modelling.plot_results
        #
        # rather than:
        #
        #     python3 /path/to/plot_results.py
        #
        # PYTHONPATH points to the directory containing the `mepram` package.
        # ------------------------------------------------------------------

        mepram_pythonpath = bind / "DOC/bd-tools"

        plot_script = textwrap.dedent(
            f"""\
            #!/bin/bash
            set -euo pipefail

            D="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

            # Make the parent directory of the `mepram` package importable.
            export PYTHONPATH={q(mepram_pythonpath)}:${{PYTHONPATH:-}}

            python3 -m mepram.modelling.plot_results --run-dir "$D"
            """
        )

        report_script = textwrap.dedent(
            f"""\
            #!/bin/bash
            set -euo pipefail

            D="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

            # Make the parent directory of the `mepram` package importable.
            export PYTHONPATH={q(mepram_pythonpath)}:${{PYTHONPATH:-}}

            python3 -m mepram.modelling.report_results "$D" "$D/README.md"
            """
        )

        files = {
            "_01_run_models.sh": launcher(jobs),
            "_02_plots.sh": plot_script,
            "_03_report.sh": report_script,
        }

        for n, t in files.items():

            p = d / n

            write(
                p,
                t,
                a.overwrite_generated,
            )

            executable(p)

        write(
            d / "generated_runs.json",
            json.dumps(
                {
                    "scope": scope,
                    "input_data": str(input_host),
                    "runs": manifest,
                },
                indent=2,
            )
            + "\n",
            a.overwrite_generated,
        )

        print(
            f"GENERATED {scope}: "
            f"{len(jobs)} run(s) in {d}"
        )

        total += len(jobs)

    print(
        f"Complete: {total} enabled run(s)"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )

    except (
        ConfigError,
        KeyError,
        ValueError,
        json.JSONDecodeError,
        OSError,
    ) as e:

        print(
            f"ERROR: {e}",
            file=sys.stderr,
        )

        raise SystemExit(2)
