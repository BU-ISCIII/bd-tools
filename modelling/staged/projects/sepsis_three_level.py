"""Sepsis 3-level project adapter."""

from __future__ import annotations

import argparse

from three_level.cli import build_arg_parser as build_sepsis_parser

PROJECT_NAME = "sepsis_three_level"


def build_parser() -> argparse.ArgumentParser:
    return build_sepsis_parser()


def run(args: argparse.Namespace) -> None:
    from three_level.modelling import run_training

    run_training(args)
