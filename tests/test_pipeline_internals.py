import numpy as np
import pandas as pd

from moosefs.feature_selection_pipeline import FeatureSelectionPipeline
from moosefs.feature_selectors import FStatisticSelector
from moosefs.merging_strategies import ConsensusMerger
from moosefs.metrics.performance_metrics import Accuracy, F1Score


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


def test_user_instances_keep_configuration():
    # Instances passed to the pipeline must be used as-is: they used to be
    # re-instantiated with default arguments, silently discarding user
    # configuration (e.g. ConsensusMerger(k=3) became k=2).
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(20, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(rng.integers(0, 2, size=len(X)), name="label")

    merger = ConsensusMerger(k=3, fill=True)
    selector = FStatisticSelector(task="classification", num_features_to_select=4)

    pipeline = FeatureSelectionPipeline(
        X=X,
        y=y,
        fs_methods=[selector, "variance_selector"],
        merging_strategy=merger,
        num_repeats=2,
        num_features_to_select=3,
        metrics=["accuracy"],
        task="classification",
        random_state=0,
        n_jobs=1,
    )

    assert pipeline.merging_strategy is merger
    assert pipeline.merging_strategy.k == 3
    assert pipeline.merging_strategy.fill is True
    assert pipeline.fs_methods[0] is selector
    assert pipeline.fs_methods[0].num_features_to_select == 4


def test_metric_instances_accepted():
    # Metric instances used to be rejected by _load_class with a ValueError,
    # despite the documentation advertising instance support.
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series((X["f0"] > 0).astype(int).to_numpy(), name="label")

    accuracy = Accuracy()
    pipeline = FeatureSelectionPipeline(
        X=X,
        y=y,
        fs_methods=["f_statistic_selector", "variance_selector"],
        merging_strategy="union_of_intersections_merger",
        num_repeats=2,
        num_features_to_select=3,
        metrics=[accuracy, F1Score()],
        task="classification",
        random_state=0,
        n_jobs=1,
        fill=True,
    )
    assert pipeline.metrics[0] is accuracy

    features, ensemble = pipeline.run(verbose=False)
    assert len(features) > 0


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
