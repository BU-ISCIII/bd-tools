"""Generic CLI dispatcher for staged modelling projects."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from .projects import PROJECTS


def _bootstrap_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--project",
        type=str,
        default="sepsis_three_level",
        choices=sorted(PROJECTS.keys()),
        help="Project pipeline to run.",
    )
    return parser


def main() -> None:
    start = time.time()

    bootstrap = _bootstrap_parser()
    pre_args, remaining = bootstrap.parse_known_args()
    project = PROJECTS[pre_args.project]

    parser = project.build_parser()
    parser.add_argument(
        "--project",
        type=str,
        default=pre_args.project,
        choices=sorted(PROJECTS.keys()),
        help="Project pipeline to run.",
    )
    args = parser.parse_args(remaining)

    print("Selected project:", pre_args.project)
    print("Parsed args:", args)

    output_folder = Path(args.output_dir)
    try:
        project.run(args)
    except Exception:
        if output_folder.exists():
            shutil.rmtree(output_folder)
        raise

    print(f"\nElapsed: {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
