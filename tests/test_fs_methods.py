import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from moosefs.feature_selectors import (
    ElasticNetSelector,
    FStatisticSelector,
    LassoSelector,
    MutualInfoSelector,
    RandomForestSelector,
    SVMSelector,
    XGBoostSelector,
)


@pytest.fixture
def fake_data_classification():
    informative_features = [6, 11, 19]  # Indices of informative features

    # Generate synthetic data with make_classification
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=2024,
    )

    return X, y, informative_features


@pytest.fixture
def fake_data_regression():
    X, y = make_regression(n_samples=1000, n_features=100, n_informative=2, random_state=1)
    informative_features = [85, 32]
    return X, y, informative_features


def test_fake_data_classification(fake_data_classification):
    X, y, expected_informative_features = fake_data_classification
    assert len(expected_informative_features) == 3
    assert X.shape[0] == 1000
    assert X.shape[1] == 20
    assert all(idx in range(X.shape[1]) for idx in expected_informative_features)


def test_fake_data_regression(fake_data_regression):
    X, y, expected_informative_features = fake_data_regression
    assert len(expected_informative_features) == 2
    assert X.shape[0] == 1000
    assert X.shape[1] == 100
    assert all(idx in range(X.shape[1]) for idx in expected_informative_features)


def test_feature_selection_f_statistic_regression(fake_data_regression):
    X, y, expected_features = fake_data_regression
    selector = FStatisticSelector(task="regression", num_features_to_select=2)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 100
    assert len(selected_features) == 2
    assert set(selected_features) == set(expected_features)


def test_feature_selection_f_statistic_classification(fake_data_classification):
    X, y, expected_features = fake_data_classification
    selector = FStatisticSelector(task="classification", num_features_to_select=3)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 20
    assert len(selected_features) == 3
    assert set(selected_features) == set(expected_features)


def test_feature_selection_mutual_info_classification(fake_data_classification):
    X, y, expected_features = fake_data_classification
    selector = MutualInfoSelector(task="classification", num_features_to_select=3)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 20
    assert len(selected_features) == 3
    assert set(selected_features) == set(expected_features)


def test_feature_selection_mutual_info_regression(fake_data_regression):
    X, y, expected_features = fake_data_regression
    selector = MutualInfoSelector(task="regression", num_features_to_select=2)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 100
    assert len(selected_features) == 2
    assert set(selected_features) == set(expected_features)


def test_feature_selection_xgboost_classification(fake_data_classification):
    X, y, expected_features = fake_data_classification
    selector = XGBoostSelector(task="classification", num_features_to_select=3)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 20
    assert len(selected_features) == 3
    assert set(selected_features) == set(expected_features)


def test_feature_selection_xgboost_regression(fake_data_regression):
    X, y, expected_features = fake_data_regression
    selector = XGBoostSelector(task="regression", num_features_to_select=2)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 100
    assert len(selected_features) == 2
    assert set(selected_features) == set(expected_features)


def test_feature_selection_random_forest_classification(fake_data_classification):
    X, y, expected_features = fake_data_classification
    selector = RandomForestSelector(task="classification", num_features_to_select=3)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 20
    assert len(selected_features) == 3
    assert set(selected_features) == set(expected_features)


def test_feature_selection_random_forest_regression(fake_data_regression):
    X, y, expected_features = fake_data_regression
    selector = RandomForestSelector(task="regression", num_features_to_select=2)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 100
    assert len(selected_features) == 2
    assert set(selected_features) == set(expected_features)


def test_feature_selection_svm_classification(fake_data_classification):
    X, y, expected_features = fake_data_classification
    selector = SVMSelector(task="classification", num_features_to_select=3)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 20
    assert len(selected_features) == 3
    assert set(selected_features) == set(expected_features)


def test_feature_selection_svm_regression(fake_data_regression):
    X, y, expected_features = fake_data_regression
    selector = SVMSelector(task="regression", num_features_to_select=2)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 100
    assert len(selected_features) == 2
    assert set(selected_features) == set(expected_features)


"""
def test_feature_selection_mrmr_classification(fake_data_classification):
    X, y, expected_features = fake_data_classification
    selector = MRMRSelector(task="classification", num_features_to_select=3)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 20
    assert len(selected_features) == 3
    assert set(selected_features) == set(expected_features)


def test_feature_selection_mrrm_regression(fake_data_regression):
    X, y, expected_features = fake_data_regression
    selector = MRMRSelector(task="regression", num_features_to_select=2)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 100
    assert len(selected_features) == 2
    assert set(selected_features) == set(expected_features)
"""


def test_random_forest_kwargs_forwarded():
    # kwargs (random_state, model hyperparameters) used to be silently dropped.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 10))
    y = rng.integers(0, 2, size=80)

    scores1, _ = RandomForestSelector(task="classification", num_features_to_select=3, random_state=0).select_features(
        X, y
    )
    scores2, _ = RandomForestSelector(task="classification", num_features_to_select=3, random_state=0).select_features(
        X, y
    )
    assert np.array_equal(scores1, scores2)

    scores3, _ = RandomForestSelector(
        task="classification", num_features_to_select=3, random_state=0, n_estimators=5
    ).select_features(X, y)
    assert not np.array_equal(scores1, scores3)


def test_xgboost_kwargs_forwarded():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 10))
    y = rng.integers(0, 2, size=80)

    scores_small, _ = XGBoostSelector(
        task="classification", num_features_to_select=3, n_estimators=1, max_depth=1
    ).select_features(X, y)
    scores_large, _ = XGBoostSelector(
        task="classification", num_features_to_select=3, n_estimators=50, max_depth=6
    ).select_features(X, y)
    assert not np.array_equal(scores_small, scores_large)


def test_mutual_info_kwargs_forwarded():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    y = rng.integers(0, 2, size=100)

    scores1, _ = MutualInfoSelector(task="classification", num_features_to_select=2, random_state=7).select_features(
        X, y
    )
    scores2, _ = MutualInfoSelector(task="classification", num_features_to_select=2, random_state=7).select_features(
        X, y
    )
    assert np.array_equal(scores1, scores2)


def test_lasso_alpha_kwarg_persists_across_calls():
    # `alpha` used to be popped from kwargs, so a second call on the same
    # selector silently fell back to the default.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    y = 3 * X[:, 0] + rng.normal(scale=0.1, size=100)

    selector = LassoSelector(task="regression", num_features_to_select=2, alpha=1e6)
    scores_first, _ = selector.select_features(X, y)
    scores_second, _ = selector.select_features(X, y)

    # A huge alpha shrinks every coefficient to zero - on every call.
    assert np.allclose(scores_first, 0.0)
    assert np.array_equal(scores_first, scores_second)


def test_elastic_net_kwargs_persist_across_calls():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    y = 3 * X[:, 0] + rng.normal(scale=0.1, size=100)

    selector = ElasticNetSelector(task="regression", num_features_to_select=2, alpha=1e6)
    scores_first, _ = selector.select_features(X, y)
    scores_second, _ = selector.select_features(X, y)

    assert np.allclose(scores_first, 0.0)
    assert np.array_equal(scores_first, scores_second)


@pytest.mark.parametrize("selector_cls", [SVMSelector, LassoSelector, ElasticNetSelector])
def test_coefficient_selectors_are_scale_invariant(selector_cls):
    # Coefficient-based selectors standardize internally: blowing up the
    # scale of an informative feature must not change what gets selected
    # (unscaled, its coefficient would shrink by the same factor and the
    # feature would drop out of the top-k).
    X, y = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=3,
        n_redundant=0,
        shuffle=False,  # informative features are columns 0-2
        random_state=5,
    )

    selector = selector_cls(task="classification", num_features_to_select=3)
    _, idx_original = selector.select_features(X, y)

    X_blown = X.copy()
    X_blown[:, 0] *= 1e6
    _, idx_blown = selector.select_features(X_blown, y)

    assert set(map(int, idx_blown)) == set(map(int, idx_original))
    assert 0 in set(map(int, idx_blown))


def test_feature_selection_lasso_classification(fake_data_classification):
    X, y, expected_features = fake_data_classification
    selector = LassoSelector(task="classification", num_features_to_select=3)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 20
    assert len(selected_features) == 3
    assert set(selected_features) == set(expected_features)


def test_feature_selection_lasso_regression(fake_data_regression):
    X, y, expected_features = fake_data_regression
    selector = LassoSelector(task="regression", num_features_to_select=2)
    scores, selected_features = selector.select_features(X, y)
    assert len(scores) == 100
    assert len(selected_features) == 2
    assert set(selected_features) == set(expected_features)
