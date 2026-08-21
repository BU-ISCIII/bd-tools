import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_benchmark.runner import _encode_binary_target


def test_encode_binary_target_maps_string_labels_to_0_1() -> None:
    y = pd.Series([
        "NEGATIVE",
        "RESIST_CEFALOSPORINAS_3a_4a",
        "NEGATIVE",
        "RESIST_CEFALOSPORINAS_3a_4a",
    ])

    encoded, classes, positive_label = _encode_binary_target(
        y,
        positive_label="RESIST_CEFALOSPORINAS_3a_4a",
    )

    assert encoded.tolist() == [0, 1, 0, 1]
    assert classes == [0, 1]
    assert positive_label == 1
