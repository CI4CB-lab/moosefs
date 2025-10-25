import numpy as np
import pandas as pd
import pytest

from moosefs.feature_selection_pipeline import FeatureSelectionPipeline


@pytest.fixture(params=["union_of_intersections_merger", "borda_merger"])
def pipeline_instance(request):
    num_samples = 100
    num_features = 5000
    feature_names = [f"Feature_{i + 1}" for i in range(num_features)]
    target_values = np.random.randint(0, 4, size=num_samples)
    data = pd.DataFrame(np.random.randn(num_samples, num_features), columns=feature_names)
    data["label"] = target_values

    fs_methods = [
        "f_statistic_selector",
        "random_forest_selector",
        "mutual_info_selector",
    ]
    merging_strategy = request.param
    num_repeats = 2
    task = "classification"
    random_state = 2024
    num_features_to_select = 300
    n_jobs = 1
    metrics = ["f1_score", "accuracy"]

    pipeline = FeatureSelectionPipeline(
        data=data,
        fs_methods=fs_methods,
        merging_strategy=merging_strategy,
        num_repeats=num_repeats,
        num_features_to_select=num_features_to_select,
        metrics=metrics,
        task=task,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return pipeline


def test_feature_selection_pipeline(pipeline_instance):
    best_features, best_repeat, best_group_name = pipeline_instance()

    assert best_features is not None, "Best features should not be None"
    assert best_repeat is not None, "Best repeat should not be None"
    assert best_group_name is not None, "Best group name should not be None"

    assert all(feature in pipeline_instance.data.columns for feature in best_features), (
        "All best features should exist in the original dataset."
    )

    assert 0 <= int(best_repeat) <= pipeline_instance.num_repeats, "Best repeat index must be within the valid range."

    assert best_group_name in pipeline_instance.subgroup_names, "Best group name must be in the defined subgroup names."


@pytest.mark.parametrize(
    "metrics",
    [
        ["f1_score", "accuracy"],
        ["f1_score", "accuracy", "logloss"],
    ],
)
@pytest.mark.parametrize("merging_strategy", ["union_of_intersections_merger", "borda_merger"])
def test_pipeline_with_varied_metrics(merging_strategy, metrics):
    num_samples = 100
    num_features = 5000
    feature_names = [f"Feature_{i + 1}" for i in range(num_features)]
    target_values = np.random.randint(0, 4, size=num_samples)
    data = pd.DataFrame(np.random.randn(num_samples, num_features), columns=feature_names)
    data["label"] = target_values

    fs_methods = [
        "f_statistic_selector",
        "random_forest_selector",
        "mutual_info_selector",
    ]
    num_repeats = 2
    task = "classification"
    random_state = 2024
    num_features_to_select = 300
    fill = True
    n_jobs = 1

    pipeline = FeatureSelectionPipeline(
        data=data,
        fs_methods=fs_methods,
        merging_strategy=merging_strategy,
        num_repeats=num_repeats,
        num_features_to_select=num_features_to_select,
        metrics=metrics,
        fill=fill,
        task=task,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    best_features, best_repeat, best_group_name = pipeline()

    assert best_features is not None, "Best features should not be None"
    assert len(best_features) == num_features_to_select, (
        "Fill is set to true, so the number of features should be equal to num_features_to_select."
    )
    assert 0 <= int(best_repeat) <= num_repeats, "Best repeat index must be within valid range."
    assert best_group_name in pipeline.subgroup_names, "Best group name must be a valid subgroup name."


@pytest.mark.parametrize("merging_strategy", ["union_of_intersections_merger", "borda_merger"])
def test_pipeline_requires_num_features(merging_strategy):
    """
    Verify that a ValueError is raised when num_features_to_select is missing.
    """
    num_samples = 100
    num_features = 5000
    feature_names = [f"Feature_{i + 1}" for i in range(num_features)]
    target_values = np.random.randint(0, 4, size=num_samples)
    data = pd.DataFrame(np.random.randn(num_samples, num_features), columns=feature_names)
    data["label"] = target_values

    fs_methods = [
        "f_statistic_selector",
        "random_forest_selector",
        "mutual_info_selector",
    ]
    num_repeats = 2
    task = "classification"
    random_state = 2024
    n_jobs = 1

    with pytest.raises(ValueError, match="num_features_to_select must be provided"):
        FeatureSelectionPipeline(
            data=data,
            fs_methods=fs_methods,
            merging_strategy=merging_strategy,
            num_repeats=num_repeats,
            num_features_to_select=None,  # Now mandatory for all cases
            metrics=["f1_score"],
            task=task,
            random_state=random_state,
            n_jobs=n_jobs,
        )
