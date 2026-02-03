import numpy as np
import pandas as pd
import pytest

from moosefs.feature_selection_pipeline import FeatureSelectionPipeline


@pytest.fixture
def pipeline_instance():
    # Define the data frame
    num_samples = 100
    num_features = 5000
    feature_names = [f"Feature_{i + 1}" for i in range(num_features)]
    target_values = np.random.randint(0, 4, size=num_samples)

    X = pd.DataFrame(np.random.randn(num_samples, num_features), columns=feature_names)
    y = pd.Series(target_values, name="class")

    # Define parameters
    fs_methods = [
        "f_statistic_selector",
        "mrmr_selector",
        "mutual_info_selector",
        "lasso_selector",
        "svm_selector",
    ]
    merging_strategy = "union_of_intersections_merger"
    num_repeats = 2
    task = "classification"
    random_state = 2024
    num_features_to_select = 300
    n_jobs = 1
    pipeline = FeatureSelectionPipeline(
        X=X,
        y=y,
        fs_methods=fs_methods,
        merging_strategy=merging_strategy,
        num_repeats=num_repeats,
        num_features_to_select=num_features_to_select,
        task=task,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    return pipeline


def test_compute_pareto_analysis(pipeline_instance):
    groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    names = ["Group 1", "Group 2", "Group 3"]
    best_group_name = pipeline_instance._compute_pareto(groups, names)
    assert best_group_name == "Group 3"


def test_generate_selector_ensembles(pipeline_instance):
    # Test with a minimum combination size of 3
    min_size = 3
    selector_sets = pipeline_instance._generate_selector_ensembles(min_group_size=min_size)

    # Calculate the expected number of combinations with 5 methods (size 3 and above)
    expected_combinations = 16  # Combinations of size 3 and size 4 and size 5
    assert len(selector_sets) == expected_combinations

    for ensemble in selector_sets:
        assert len(ensemble) >= min_size  # Ensure each ensemble has at least the minimum size
        for name in ensemble:
            assert isinstance(name, str)
            assert name in [
                "MutualInfo",
                "Lasso",
                "FStatistic",
                "SVM",
                "MRMR",
            ]

    # Test for the ValueError when min_size exceeds the number of methods
    with pytest.raises(ValueError):
        pipeline_instance._generate_selector_ensembles(min_group_size=6)

    for min_size, expected_combinations in zip([2, 4, 5], [26, 6, 1]):
        assert len(pipeline_instance._generate_selector_ensembles(min_group_size=min_size)) == expected_combinations
