import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modelling.cli import build_arg_parser


def test_level3_gate_args_are_available_and_parsed() -> None:
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "--resistance-gate-target",
            "culture_positive",
            "--resistance-gate-negative-label",
            "NEG",
            "--set-resistance-gate-recall",
            "0.9",
        ]
    )

    assert args.resistance_gate_target == "culture_positive"
    assert args.resistance_gate_negative_label == "NEG"
    assert args.set_resistance_gate_recall == 0.9
