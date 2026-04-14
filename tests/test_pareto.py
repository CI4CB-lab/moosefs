# tests/test_pareto.py
from typing import List, Tuple

import pytest

from moosefs.core.pareto import ParetoAnalysis


# ------------------------------------------------------------------ helpers
def _strip(rows: List[List]) -> List[List]:
    """keep the first 4 columns (name, dominate, dominated, scalar)."""
    return [r[:4] for r in rows]


def _lex(rows: List[List]) -> List[List]:
    """lexicographically sort by group-name only (for tie-cases)."""
    return sorted(rows, key=lambda r: r[0])


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def sample() -> Tuple[List[List[float]], List[str]]:
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    names = ["Group 1", "Group 2", "Group 3"]
    return data, names


@pytest.fixture
def same() -> Tuple[List[List[float]], List[str]]:
    data = [[5, 5, 5]] * 3
    names = ["Group 1", "Group 2", "Group 3"]
    return data, names


@pytest.fixture
def big() -> Tuple[List[List[float]], List[str]]:
    data = [
        [0, 0, 0],
        [2, 4, 6],
        [2, 4, 6],
        [2, 4, 7],
        [5, 7, 9],
        [2, 3, 4],
        [6, 8, 10],
        [3, 5, 7],
        [4, 5, 6],
        [13, 13, 13],
    ]
    names = [f"Group {i + 1}" for i in range(10)]
    return data, names


# ------------------------------------------------------------------ tests
def test_compute_dominance(sample):
    data, names = sample
    p = ParetoAnalysis(data, names)
    raw = _strip(p.get_results())[::-1]  # original order
    assert raw == [
        ["Group 1", 0, 2, -2],
        ["Group 2", 1, 1, 0],
        ["Group 3", 2, 0, 2],
    ]


def test_get_results_rank(sample):
    data, names = sample
    p = ParetoAnalysis(data, names)
    ranked = _strip(p.get_results())
    assert ranked == [
        ["Group 3", 2, 0, 2],
        ["Group 2", 1, 1, 0],
        ["Group 1", 0, 2, -2],
    ]


def test_all_equal_dominance(same):
    data, names = same
    p = ParetoAnalysis(data, names)
    ranked = _strip(p.get_results())

    # every scalar is 0; order is lexicographic by design
    assert ranked == [
        ["Group 1", 0, 0, 0],
        ["Group 2", 0, 0, 0],
        ["Group 3", 0, 0, 0],
    ]


def test_big_data(big):
    data, names = big
    p = ParetoAnalysis(data, names)
    ranked = _strip(p.get_results())

    expected = [
        ["Group 10", 9, 0, 9],
        ["Group 7", 8, 1, 7],
        ["Group 5", 7, 2, 5],
        ["Group 8", 5, 3, 2],
        ["Group 9", 4, 3, 1],
        ["Group 4", 4, 4, 0],
        ["Group 2", 2, 6, -4],
        ["Group 3", 2, 6, -4],
        ["Group 6", 1, 8, -7],
        ["Group 1", 0, 9, -9],
    ]
    # order of groups with equal scalar (-4) is lexicographic
    assert ranked == expected


# ------------------------------------------------------------------ utopia flag tests
def test_utopia_flag_not_set_when_pareto_decides(sample):
    """Pareto dominance produces a unique winner — utopia tie-break must NOT fire."""
    data, names = sample
    p = ParetoAnalysis(data, names)
    p.get_results()
    assert p.utopia_tiebreak_used is False
    assert p.n_tied == 0


def test_utopia_flag_set_when_all_tied(same):
    """All groups identical — Pareto cannot differentiate, utopia MUST fire."""
    data, names = same
    p = ParetoAnalysis(data, names)
    p.get_results()
    assert p.utopia_tiebreak_used is True
    assert p.n_tied == len(names)


def test_utopia_flag_set_with_partial_tie():
    """Top two groups tied on scalar, third clearly worse — utopia fires for the top tie only."""
    # Group A and Group B are incomparable (each better on one metric), Group C is dominated by both.
    data = [
        [10, 1],  # Group A: better on metric 0
        [1, 10],  # Group B: better on metric 1
        [0, 0],  # Group C: dominated by both
    ]
    names = ["Group A", "Group B", "Group C"]
    p = ParetoAnalysis(data, names)
    p.get_results()
    assert p.utopia_tiebreak_used is True
    assert p.n_tied == 2


def test_utopia_flag_not_set_for_big_data_with_clear_winner(big):
    """big fixture has a unique top group (Group 10, scalar=9) — no utopia needed."""
    data, names = big
    p = ParetoAnalysis(data, names)
    p.get_results()
    assert p.utopia_tiebreak_used is False
    assert p.n_tied == 0
