import numpy as np
import pandas as pd
import pytest

from moosefs.core import Feature
from moosefs.feature_selectors import FeatureSelector, FStatisticSelector
from moosefs.merging_strategies import ArithmeticMeanMerger


class FixedScoreSelector(FeatureSelector):
    """Minimal selector for exercising the shared score contract."""

    def __init__(self, scores, num_features_to_select):
        super().__init__(task="classification", num_features_to_select=num_features_to_select)
        self.scores = scores

    def compute_scores(self, X, y):
        return self.scores


def test_f_statistic_does_not_select_a_constant_column():
    X = pd.DataFrame(
        {
            "signal": [0, 1, 2, 3, 5, 6, 7, 8],
            "constant": [4] * 8,
            "noise": [2, 1, 4, 5, 0, 7, 3, 6],
        }
    )
    y = np.array([0] * 4 + [1] * 4)

    with pytest.warns(UserWarning, match="constant"):
        scores, indices = FStatisticSelector("classification", 1).select_features(X, y)

    assert np.isnan(scores[1])
    assert X.columns[indices].tolist() == ["signal"]


def test_selection_excludes_undefined_scores_and_preserves_original_indices():
    scores = np.array([np.nan, 2.0, 0.0, -1.0, 2.0, -np.inf, np.inf])
    selector = FixedScoreSelector(scores, num_features_to_select=5)

    returned_scores, indices = selector.select_features(np.zeros((3, 7)), np.zeros(3))

    assert returned_scores is scores
    assert indices.tolist() == [6, 1, 4, 2, 3]


@pytest.mark.parametrize(
    ("scores", "num_features_to_select", "valid_count"),
    [
        ([np.nan, -np.inf], 1, 0),
        ([1.0, np.nan, -np.inf], 2, 1),
    ],
)
def test_selection_rejects_too_few_defined_scores(scores, num_features_to_select, valid_count):
    selector = FixedScoreSelector(scores, num_features_to_select)

    with pytest.raises(ValueError, match=rf"only {valid_count} have defined scores"):
        selector.select_features(np.zeros((2, len(scores))), np.zeros(2))


def test_positive_infinite_f_statistic_is_selected_for_perfect_separation():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    with pytest.warns(RuntimeWarning, match="divide by zero"):
        scores, indices = FStatisticSelector("classification", 1).select_features(X, y)

    assert np.isposinf(scores[0])
    assert indices.tolist() == [0]


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        ([[1.0, 2.0, 3.0]], "one-dimensional"),
        ([1.0, 2.0], "length .* number of input features"),
        (["high", "medium", "low"], "real numeric"),
    ],
)
def test_selection_rejects_invalid_score_arrays(scores, message):
    selector = FixedScoreSelector(scores, num_features_to_select=1)

    with pytest.raises(ValueError, match=message):
        selector.select_features(np.zeros((2, 3)), np.zeros(2))


def test_equal_scores_are_ordered_by_original_feature_index():
    selector = FixedScoreSelector([1.0, 3.0, 3.0, 3.0], num_features_to_select=3)

    _, indices = selector.select_features(np.zeros((2, 4)), np.zeros(2))

    assert indices.tolist() == [1, 2, 3]


def test_positive_infinite_scores_remain_valid_during_merging():
    subsets = [
        [Feature("perfect", np.inf), Feature("strong", 10.0), Feature("weak", 0.0)],
        [Feature("perfect", 2.0), Feature("strong", 1.0), Feature("weak", 0.0)],
    ]

    with np.errstate(invalid="raise"):
        result = ArithmeticMeanMerger().merge(subsets, num_features_to_select=3)

    assert result == ["perfect", "strong", "weak"]
