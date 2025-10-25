import numpy as np
import pandas as pd

from moosefs.feature_selection_pipeline import FeatureSelectionPipeline


def _tiny_pipeline():
    # very small dataset to avoid heavy computations
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(20, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(rng.integers(0, 2, size=len(X)), name="label")

    return FeatureSelectionPipeline(
        X=X,
        y=y,
        fs_methods=[
            "f_statistic_selector",
            "variance_selector",
        ],
        merging_strategy="borda_merger",
        num_repeats=2,
        num_features_to_select=3,
        metrics=["accuracy"],
        task="classification",
        random_state=123,
        n_jobs=1,
    )


def test_replace_none_behaviour():
    pl = _tiny_pipeline()
    # one group has a None → whole row becomes -inf values
    metrics = [[0.1, 0.2], [None, 0.3]]
    cleaned = pl._replace_none(metrics)
    assert cleaned[0] == [0.1, 0.2]
    assert cleaned[1] == [-float("inf"), -float("inf")]


def test_calculate_means_and_extract_repeat_metrics():
    pl = _tiny_pipeline()
    # build artificial result_dicts for two metrics
    groups = [(0, ("A",)), (1, ("A",)), (0, ("B",)), (1, ("B",))]
    d1 = {groups[0]: 1.0, groups[1]: 3.0, groups[2]: 2.0, groups[3]: 4.0}
    d2 = {groups[0]: 10.0, groups[1]: 30.0, groups[2]: 20.0, groups[3]: 40.0}

    means = FeatureSelectionPipeline._calculate_means([d1, d2], [("A",), ("B",)])
    assert means == [[2.0, 20.0], [3.0, 30.0]]

    rows = pl._extract_repeat_metrics(("A",), d1, d2)
    # num_repeats=2 → two rows
    assert len(rows) == 2
    assert rows[0] == [1.0, 10.0]
    assert rows[1] == [3.0, 30.0]


def test_invalid_task_raises():
    X = pd.DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])
    y = pd.Series(np.random.randint(0, 2, size=10), name="label")
    try:
        FeatureSelectionPipeline(
            X=X,
            y=y,
            fs_methods=["f_statistic_selector", "variance_selector"],
            merging_strategy="borda_merger",
            num_repeats=1,
            num_features_to_select=2,
            task="not-a-valid-task",
        )
        assert False, "Expected ValueError for invalid task"
    except ValueError as e:
        assert "Task must be either" in str(e)


def test_data_argument_uses_last_column():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "target_name": [0, 1, 0],
        }
    )
    pipeline = FeatureSelectionPipeline(
        data=df,
        fs_methods=["f_statistic_selector", "variance_selector"],
        merging_strategy="borda_merger",
        num_repeats=1,
        num_features_to_select=2,
        task="classification",
    )
    assert pipeline.target_name == "target_name"
    assert list(pipeline.data.columns)[-1] == "target_name"
