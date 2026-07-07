import numpy as np

from .base_merger import MergingStrategy


class ArithmeticMeanMerger(MergingStrategy):
    """Rank-based merging using the arithmetic mean of scores."""

    name = "ArithmeticMean"

    def __init__(self, **kwargs) -> None:
        # Keep taxonomy consistent with existing mergers
        super().__init__("rank-based")
        self.kwargs = kwargs

    def merge(
        self,
        subsets: list,
        num_features_to_select: int,
        **kwargs,
    ) -> list:
        """Return the top‑k feature names after arithmetic-mean aggregation.

        Subsets are aligned by feature name and each selector's scores are
        min-max normalized before averaging, so selectors may contribute
        different feature sets on different score scales. Features missing
        from a selector's subset get no credit from it.

        Args:
            subsets: Feature lists (one list per selector).
            num_features_to_select: Number of names to return.

        Returns:
            Feature names sorted by mean normalized score.
        """
        self._validate_input(subsets)

        feature_names, scores = self._aligned_scores(subsets)

        # Arithmetic mean of per-selector normalized scores
        scores_merged = self._normalize_scores(scores).mean(axis=1)

        # Lower score ⇒ higher rank (same convention as Borda)
        sorted_names = [feature_names[i] for i in np.argsort(-scores_merged, kind="stable")]
        return sorted_names[:num_features_to_select]
