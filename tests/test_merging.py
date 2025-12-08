import pytest

from moosefs.core.feature import Feature
from moosefs.merging_strategies import (
    ArithmeticMeanMerger,
    BordaMerger,
    FrequencyBootstrapMerger,
    L2NormMerger,
    UnionOfIntersectionsMerger,
)


@pytest.fixture
def union_of_intersections():
    return UnionOfIntersectionsMerger()


@pytest.fixture
def borda_merger():
    return BordaMerger()


@pytest.fixture
def l2norm_merger():
    return L2NormMerger()


@pytest.fixture
def arithmetic_mean_merger():
    return ArithmeticMeanMerger()


@pytest.fixture
def frequency_merger():
    return FrequencyBootstrapMerger(num_bootstrap=5, min_freq=0.6, use_scores=True, fill=False)


def subset_to_Feature(subset):
    return [[Feature(name=str(i), score=i) for i in sublist] for sublist in subset]


def test_basic_functionality(union_of_intersections):
    subsets = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    subsets = subset_to_Feature(subsets)
    result = union_of_intersections.merge(subsets)
    assert isinstance(result, set)


def test_empty_input(union_of_intersections):
    subsets = []
    with pytest.raises(ValueError):
        union_of_intersections.merge(subsets)


def test_single_subset(union_of_intersections):
    subsets = [[1, 2, 3, 4]]
    subsets = subset_to_Feature(subsets)
    result = union_of_intersections.merge(subsets)
    assert result == {"1", "2", "3", "4"}


def test_multiple_subsets(union_of_intersections):
    subsets = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    subsets = subset_to_Feature(subsets)
    result = union_of_intersections.merge(subsets)
    assert result == {"2", "3", "4"}


def test_symmetry_property(union_of_intersections):
    subsets_1 = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    subsets_2 = [[3, 4, 5], [2, 3, 4], [1, 2, 3]]
    subsets_1 = subset_to_Feature(subsets_1)
    subsets_2 = subset_to_Feature(subsets_2)
    result_1 = union_of_intersections.merge(subsets_1)
    result_2 = union_of_intersections.merge(subsets_2)
    assert result_1 == result_2


def test_empty_output(union_of_intersections):
    subsets = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    subsets = subset_to_Feature(subsets)
    result = union_of_intersections.merge(subsets)
    assert result == set()


def test_merge_fill_full(union_of_intersections):
    subsets = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    subsets = subset_to_Feature(subsets)
    result = union_of_intersections.merge(subsets, num_features_to_select=9, fill=True)
    assert result == set([str(i) for i in range(1, 10)])


def test_merge_fill(union_of_intersections):
    subsets = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    subsets = subset_to_Feature(subsets)
    result = union_of_intersections.merge(subsets, num_features_to_select=4, fill=True)
    for i in ["2", "3", "4"]:
        assert i in result
    assert len(result) == 4


def test_borda_basic_functionality(borda_merger):
    subsets = [
        [Feature("A", 10), Feature("B", 8), Feature("C", 6)],
        [Feature("A", 9), Feature("B", 7), Feature("C", 5)],
        [Feature("A", 11), Feature("B", 6), Feature("C", 4)],
    ]
    result = borda_merger.merge(subsets, num_features_to_select=3)
    assert result == ["A", "B", "C"]


def test_borda_empty_input(borda_merger):
    subsets = []
    with pytest.raises(ValueError):
        borda_merger.merge(subsets, num_features_to_select=3)


def test_borda_single_score_list(borda_merger):
    subsets = [[Feature("A", 10), Feature("B", 8), Feature("C", 6)]]
    result = borda_merger.merge(subsets, num_features_to_select=3)
    assert result == ["A", "B", "C"]


def test_frequency_basic_threshold(frequency_merger):
    feature_names = ["a", "b", "c"]
    stats = {
        "m1": {"counts": [3, 1, 0], "score_sums": [3.0, 1.0, 0.0], "n_runs": 5},
        "m2": {"counts": [4, 2, 0], "score_sums": [4.0, 2.0, 0.0], "n_runs": 5},
    }
    merged = frequency_merger.merge(
        [],
        num_features_to_select=3,
        group=("m1", "m2"),
        feature_names=feature_names,
        bootstrap_stats=stats,
    )
    # feature "a" has freq 0.7, "b" 0.3, "c" 0.0 with min_freq=0.6
    assert merged == ["a"]


def test_frequency_fill_and_score_tiebreak(frequency_merger):
    feature_names = ["a", "b", "c"]
    stats = {
        "m1": {"counts": [3, 3, 0], "score_sums": [5.0, 4.0, 0.0], "n_runs": 5},
        "m2": {"counts": [3, 3, 0], "score_sums": [6.0, 1.0, 0.0], "n_runs": 5},
    }
    merged = frequency_merger.merge(
        [],
        num_features_to_select=2,
        group=("m1", "m2"),
        feature_names=feature_names,
        bootstrap_stats=stats,
        fill=True,
    )
    # "a" and "b" have equal freq=0.6; scores break tie to keep "a" first
    assert merged == ["a", "b"]


def test_frequency_fill_pads_when_under_min_freq():
    feature_names = ["a", "b", "c"]
    stats = {
        "m1": {"counts": [5, 1, 0], "score_sums": [5.0, 1.0, 0.0], "n_runs": 5},
        "m2": {"counts": [4, 0, 0], "score_sums": [4.0, 0.0, 0.0], "n_runs": 5},
    }
    merger = FrequencyBootstrapMerger(num_bootstrap=5, min_freq=0.8, use_scores=False, fill=True)
    merged = merger.merge(
        [],
        num_features_to_select=2,
        group=("m1", "m2"),
        feature_names=feature_names,
        bootstrap_stats=stats,
        fill=True,
    )
    # Only "a" clears the threshold; fill=True should backfill the next-best by frequency.
    assert merged == ["a", "b"]


def test_borda_multiple_scores(borda_merger):
    subsets = [
        [Feature("A", 10), Feature("B", 7), Feature("C", 5)],
        [Feature("A", 9), Feature("B", 8), Feature("C", 6)],
        [Feature("A", 8), Feature("B", 6), Feature("C", 7)],
    ]
    result = borda_merger.merge(subsets, num_features_to_select=3)
    assert result == ["A", "B", "C"]


def test_borda_k_features(borda_merger):
    subsets = [
        [Feature("A", 10), Feature("B", 8), Feature("C", 6)],
        [Feature("A", 9), Feature("B", 7), Feature("C", 5)],
        [Feature("A", 11), Feature("B", 6), Feature("C", 4)],
    ]
    result = borda_merger.merge(subsets, num_features_to_select=2)
    assert result == ["A", "B"]


def test_borda_symmetry_property(borda_merger):
    subsets = [
        [Feature("A", 10), Feature("B", 7), Feature("C", 6)],
        [Feature("A", 7), Feature("B", 10), Feature("C", 6)],
    ]
    result = borda_merger.merge(subsets, num_features_to_select=3)
    assert result == ["A", "B", "C"]


def test_borda_big(borda_merger):
    features1 = [
        Feature(name="A", score=8),
        Feature(name="B", score=3),
        Feature(name="C", score=6),
        Feature(name="D", score=9),
        Feature(name="E", score=1),
        Feature(name="F", score=5),
        Feature(name="G", score=7),
        Feature(name="H", score=2),
        Feature(name="I", score=4),
        Feature(name="J", score=10),
    ]

    features2 = [
        Feature(name="A", score=7),
        Feature(name="B", score=6),
        Feature(name="C", score=5),
        Feature(name="D", score=4),
        Feature(name="E", score=3),
        Feature(name="F", score=10),
        Feature(name="G", score=9),
        Feature(name="H", score=2),
        Feature(name="I", score=8),
        Feature(name="J", score=1),
    ]

    expected_result = ["G", "A", "F", "D", "I", "C", "J", "B", "E", "H"]

    subsets = [features1, features2]
    result = borda_merger.merge(subsets, num_features_to_select=10)
    assert result == expected_result


def test_l2norm_basic_functionality(l2norm_merger):
    subsets = [
        [Feature("A", 1), Feature("B", 2), Feature("C", 3)],
        [Feature("A", 3), Feature("B", 2), Feature("C", 1)],
    ]
    result = l2norm_merger.merge(subsets, num_features_to_select=3)
    assert result == ["A", "C", "B"]


def test_l2norm_empty_input(l2norm_merger):
    subsets = []
    with pytest.raises(ValueError):
        l2norm_merger.merge(subsets, num_features_to_select=3)


def test_l2norm_single_score_list(l2norm_merger):
    subsets = [[Feature("A", 10), Feature("B", 8), Feature("C", 6)]]
    result = l2norm_merger.merge(subsets, num_features_to_select=3)
    assert result == ["A", "B", "C"]


def test_l2norm_symmetry_property(l2norm_merger):
    subsets1 = [
        [Feature("A", 1), Feature("B", 2), Feature("C", 3)],
        [Feature("A", 3), Feature("B", 2), Feature("C", 1)],
    ]
    subsets2 = [
        [Feature("A", 3), Feature("B", 2), Feature("C", 1)],
        [Feature("A", 1), Feature("B", 2), Feature("C", 3)],
    ]
    result1 = l2norm_merger.merge(subsets1, num_features_to_select=3)
    result2 = l2norm_merger.merge(subsets2, num_features_to_select=3)
    assert result1 == result2


def test_l2norm_k_features(l2norm_merger):
    subsets = [
        [Feature("A", 1), Feature("B", 2), Feature("C", 3)],
        [Feature("A", 3), Feature("B", 2), Feature("C", 1)],
    ]
    result = l2norm_merger.merge(subsets, num_features_to_select=2)
    assert result == ["A", "C"]


def test_l2norm_big(l2norm_merger):
    features1 = [
        Feature(name="A", score=8),
        Feature(name="B", score=3),
        Feature(name="C", score=6),
        Feature(name="D", score=9),
        Feature(name="E", score=1),
        Feature(name="F", score=5),
        Feature(name="G", score=7),
        Feature(name="H", score=2),
        Feature(name="I", score=4),
        Feature(name="J", score=10),
    ]

    features2 = [
        Feature(name="A", score=7),
        Feature(name="B", score=6),
        Feature(name="C", score=5),
        Feature(name="D", score=4),
        Feature(name="E", score=3),
        Feature(name="F", score=10),
        Feature(name="G", score=9),
        Feature(name="H", score=2),
        Feature(name="I", score=8),
        Feature(name="J", score=1),
    ]

    expected_result = ["G", "F", "A", "J", "D", "I", "C", "B", "E", "H"]

    subsets = [features1, features2]
    result = l2norm_merger.merge(subsets, num_features_to_select=10)
    assert result == expected_result


def test_arithmetic_mean_basic_functionality(arithmetic_mean_merger):
    subsets = [
        [Feature("A", 10), Feature("B", 8), Feature("C", 6)],
        [Feature("A", 9), Feature("B", 7), Feature("C", 5)],
        [Feature("A", 8), Feature("B", 6), Feature("C", 7)],
    ]
    result = arithmetic_mean_merger.merge(subsets, num_features_to_select=3)
    assert result == ["A", "B", "C"]


def test_arithmetic_mean_empty_input(arithmetic_mean_merger):
    subsets = []
    with pytest.raises(ValueError):
        arithmetic_mean_merger.merge(subsets, num_features_to_select=3)


def test_arithmetic_mean_single_score_list(arithmetic_mean_merger):
    subsets = [[Feature("A", 10), Feature("B", 8), Feature("C", 6)]]
    result = arithmetic_mean_merger.merge(subsets, num_features_to_select=3)
    assert result == ["A", "B", "C"]


def test_arithmetic_mean_multiple_scores(arithmetic_mean_merger):
    subsets = [
        [Feature("A", 90), Feature("B", 70), Feature("C", 50)],
        [Feature("A", 80), Feature("B", 60), Feature("C", 40)],
        [Feature("A", 85), Feature("B", 65), Feature("C", 55)],
    ]
    result = arithmetic_mean_merger.merge(subsets, num_features_to_select=3)
    assert result == ["A", "B", "C"]


def test_arithmetic_mean_k_features(arithmetic_mean_merger):
    subsets = [
        [Feature("A", 10), Feature("B", 8), Feature("C", 6)],
        [Feature("A", 9), Feature("B", 7), Feature("C", 5)],
        [Feature("A", 11), Feature("B", 6), Feature("C", 4)],
    ]
    result = arithmetic_mean_merger.merge(subsets, num_features_to_select=2)
    assert result == ["A", "B"]


def test_arithmetic_mean_symmetry_property(arithmetic_mean_merger):
    subsets1 = [
        [Feature("A", 10), Feature("B", 7), Feature("C", 6)],
        [Feature("A", 7), Feature("B", 10), Feature("C", 6)],
    ]
    subsets2 = [
        [Feature("A", 7), Feature("B", 10), Feature("C", 6)],
        [Feature("A", 10), Feature("B", 7), Feature("C", 6)],
    ]
    result1 = arithmetic_mean_merger.merge(subsets1, num_features_to_select=3)
    result2 = arithmetic_mean_merger.merge(subsets2, num_features_to_select=3)
    assert result1 == result2


def test_arithmetic_mean_big(arithmetic_mean_merger):
    features1 = [
        Feature(name="A", score=8),
        Feature(name="B", score=3),
        Feature(name="C", score=6),
        Feature(name="D", score=9),
        Feature(name="E", score=1),
        Feature(name="F", score=5),
        Feature(name="G", score=7),
        Feature(name="H", score=2),
        Feature(name="I", score=4),
        Feature(name="J", score=10),
    ]

    features2 = [
        Feature(name="A", score=7),
        Feature(name="B", score=6),
        Feature(name="C", score=5),
        Feature(name="D", score=4),
        Feature(name="E", score=3),
        Feature(name="F", score=10),
        Feature(name="G", score=9),
        Feature(name="H", score=2),
        Feature(name="I", score=8),
        Feature(name="J", score=1),
    ]

    expected_result = ["G", "A", "F", "D", "I", "C", "J", "B", "E", "H"]

    subsets = [features1, features2]
    result = arithmetic_mean_merger.merge(subsets, num_features_to_select=10)
    assert result == expected_result
