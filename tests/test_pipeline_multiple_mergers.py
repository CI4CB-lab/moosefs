import pandas as pd

from moosefs import FeatureSelectionPipeline
from moosefs.merging_strategies.base_merger import MergingStrategy


class DummySelector:
    """Selector that marks provided indices as selected with given scores."""

    def __init__(self, name, scores, indices):
        self.name = name
        self._scores = list(scores)
        self._indices = list(indices)

    def select_features(self, X, y):
        return list(self._scores[: X.shape[1]]), list(self._indices)


class SortedMerger(MergingStrategy):
    """Simple deterministic merger for testing multiple strategies."""

    def __init__(self, name, reverse: bool = False):
        super().__init__("set-based")
        self.name = name
        self._reverse = reverse

    def merge(self, subsets, num_features_to_select=None, **kwargs):
        self._validate_input(subsets)
        names = sorted({f.name for subset in subsets for f in subset}, reverse=self._reverse)
        if num_features_to_select is not None:
            return names[:num_features_to_select]
        return names


def _toy_data():
    X = pd.DataFrame(
        {
            "f0": list(range(12)),
            "f1": [0, 1] * 6,
        }
    )
    y = pd.Series([0, 1] * 6, name="target")
    return pd.concat([X, y], axis=1)


def test_multiple_merging_strategies_expand_ensembles():
    data = _toy_data()
    sel1 = DummySelector("s1", scores=[2.0, 1.0], indices=[0, 1])
    sel2 = DummySelector("s2", scores=[1.0, 2.0], indices=[0, 1])

    merger_a = SortedMerger("KeepSorted", reverse=False)
    merger_b = SortedMerger("ReverseSorted", reverse=True)

    pipeline = FeatureSelectionPipeline(
        data=data,
        fs_methods=[sel1, sel2],
        merging_strategy=[merger_a, merger_b],
        num_repeats=1,
        num_features_to_select=1,
        metrics=[],
        task="classification",
        min_group_size=2,
        fill=False,
        random_state=42,
    )

    merged, best_repeat, best_ensemble = pipeline.run(verbose=False)

    assert pipeline._multiple_mergers is True
    expected_selectors = ("s1", "s2")
    assert pipeline.ensembles == [
        ("KeepSorted", expected_selectors),
        ("ReverseSorted", expected_selectors),
    ]

    assert (0, ("KeepSorted", expected_selectors)) in pipeline.merged_features
    assert (0, ("ReverseSorted", expected_selectors)) in pipeline.merged_features
    assert pipeline.merged_features[(0, ("KeepSorted", expected_selectors))] == ["f0"]
    assert pipeline.merged_features[(0, ("ReverseSorted", expected_selectors))] == ["f1"]

    assert best_repeat == 0
    assert best_ensemble in pipeline.ensembles
    assert merged in (["f0"], ["f1"])
