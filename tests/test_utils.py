"""Module to test bowline.utils."""

import pandas as pd
import pytest

from bowline.utils import detect_series_type


@pytest.mark.parametrize(
    "input_series, expected",
    [
        (pd.Series([0, 1, 1, 0]), "binary"),
        (pd.Series([0, 1, 2, 3]), "id"),
        (pd.Series([0.5, 1.2, 3.4, 4.4, 5.1, 5.1, 1.1]), "number"),
        (pd.Series([0] * 1000 + [1, 2, 3, 4, 5, 6, 7]), "number"),
        (
            pd.Series(
                [0] * 1000 + [1] * 999 + [2] * 999 + [3] * 999 + [4] * 999 + [5] * 999 + [6, 7]
            ),
            "number",
        ),
        (pd.Series(["married", "divorced", "single", "divorced"]), "category"),
        (pd.Series([0] * 1000 + [1, 2]), "category"),
        (pd.Series([1] * 1000 + [0, 2]), "category"),
    ],
)
def test_detect_series_type(input_series, expected):
    """Test detect_series_type()."""
    assert detect_series_type(input_series) == expected
