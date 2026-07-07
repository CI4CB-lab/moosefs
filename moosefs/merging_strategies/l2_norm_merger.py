import numpy as np

from .base_merger import MergingStrategy


class L2NormMerger(MergingStrategy):
    """Rank-based merging using the L2-norm (RMS) of scores."""

    name = "L2Norm"

    def __init__(self, **kwargs) -> None:
        super().__init__("rank-based")
        self.kwargs = kwargs

    def merge(
        self,
        subsets: list,
        num_features_to_select: int,
        **kwargs,
    ) -> list:
        """Return the top‑k feature names after L2-norm aggregation.

        Subsets are aligned by feature name and each selector's scores are
        min-max normalized before aggregation, so selectors may contribute
        different feature sets on different score scales. Features missing
        from a selector's subset get no credit from it.

        Args:
            subsets: Feature lists (one list per selector).
            num_features_to_select: Number of names to return.

        Returns:
            Feature names sorted by aggregated L2 score.
        """
        self._validate_input(subsets)

        feature_names, scores = self._aligned_scores(subsets)

        # Euclidean norm (root-mean-square) of per-selector normalized scores
        scores_merged = np.linalg.norm(self._normalize_scores(scores), ord=2, axis=1)

        sorted_names = [feature_names[i] for i in np.argsort(-scores_merged, kind="stable")]
        return sorted_names[:num_features_to_select]
