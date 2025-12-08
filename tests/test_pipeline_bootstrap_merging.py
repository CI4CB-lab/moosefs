import pandas as pd

from moosefs import FeatureSelectionPipeline
from moosefs.merging_strategies import BordaMerger, ConsensusMerger


class DummySelector:
    """Deterministic selector stub for testing bootstrap logic."""

    def __init__(self, name, indices, scores):
        self.name = name
        self._indices = list(indices)
        self._scores = list(scores)

    def select_features(self, X, y):
        n_features = X.shape[1]
        scores = [0.0] * n_features
        for idx, s in zip(self._indices, self._scores):
            if idx < n_features:
                scores[idx] = s
        return scores, self._indices


def _toy_data():
    X = pd.DataFrame(
        {
            "f0": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "f1": [4, 3, 2, 1, 0, 1, 2, 3, 4, 5],
            "f2": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], name="target")
    return pd.concat([X, y], axis=1)


def test_bootstrap_prefilter_set_based_merger():
    data = _toy_data()
    sel1 = DummySelector("s1", indices=[0, 1], scores=[1.0, 0.8])
    sel2 = DummySelector("s2", indices=[0, 1], scores=[0.9, 0.7])

    pipeline = FeatureSelectionPipeline(
        data=data,
        fs_methods=[sel1, sel2],
        merging_strategy=ConsensusMerger(k=2, fill=False),
        num_repeats=1,
        num_features_to_select=2,
        metrics=[],
        task="classification",
        bootstrap=True,
        bootstrap_num_samples=5,
        bootstrap_min_freq=0.5,
        fill=False,
    )

    merged, _, _ = pipeline.run(verbose=False)
    assert set(merged) == {"f0", "f1"}


def test_bootstrap_prefilter_rank_based_merger():
    data = _toy_data()
    sel1 = DummySelector("s1", indices=[0, 1], scores=[1.0, 0.5])
    sel2 = DummySelector("s2", indices=[0], scores=[0.9])

    pipeline = FeatureSelectionPipeline(
        data=data,
        fs_methods=[sel1, sel2],
        merging_strategy=BordaMerger(),
        num_repeats=1,
        num_features_to_select=2,
        metrics=[],
        task="classification",
        bootstrap=True,
        bootstrap_num_samples=3,
        bootstrap_min_freq=0.75,
        fill=False,
    )

    merged, _, _ = pipeline.run(verbose=False)
    assert merged == ["f0"]


def test_bootstrap_fill_backfills_to_requested_size():
    data = _toy_data()
    sel1 = DummySelector("s1", indices=[0], scores=[1.0])
    sel2 = DummySelector("s2", indices=[0], scores=[0.9])

    pipeline = FeatureSelectionPipeline(
        data=data,
        fs_methods=[sel1, sel2],
        merging_strategy=ConsensusMerger(k=2),
        num_repeats=1,
        num_features_to_select=2,
        metrics=[],
        task="classification",
        bootstrap=True,
        bootstrap_num_samples=3,
        bootstrap_min_freq=0.9,
        fill=True,
    )

    merged, _, _ = pipeline.run(verbose=False)
    assert set(merged) == {"f0", "f1"}
    assert len(merged) == 2
