import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from load_test import percentile  # noqa: E402


def test_percentile_median_of_sorted_values():
    assert percentile([10, 20, 30, 40, 50], 0.5) == 30


def test_percentile_p0_and_p100_are_min_and_max():
    values = [5, 1, 9, 3]
    assert percentile(values, 0.0) == 1
    assert percentile(values, 1.0) == 9


def test_percentile_empty_list_is_zero():
    assert percentile([], 0.5) == 0.0
