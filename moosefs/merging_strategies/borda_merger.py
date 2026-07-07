import numpy as np
from ranky import borda

from .base_merger import MergingStrategy


class BordaMerger(MergingStrategy):
    """Rank-based merging using the Borda count method."""

    name = "Borda"

    def __init__(self, **kwargs) -> None:
        """Initialize a rank-based merger.

        Args:
            **kwargs: Forwarded to the Borda routine (if applicable).
        """
        super().__init__("rank-based")
        self.kwargs = kwargs

    def merge(self, subsets: list, num_features_to_select: int, **kwargs) -> list:
        """Merge by Borda and return top-k names.

        Subsets are aligned by feature name, so selectors may contribute
        different feature sets. A feature missing from a selector's subset is
        ranked worst for that selector.

        Args:
            subsets: Feature lists (one list per selector).
            num_features_to_select: Number of names to return.

        Returns:
            Feature names sorted by merged Borda scores.
        """
        self._validate_input(subsets)

        feature_names, scores = self._aligned_scores(subsets)

        # Missing features rank worst for that selector.
        scores = np.where(np.isnan(scores), -np.inf, scores)

        # Apply Borda count method
        scores_merged = borda(m=scores, **self.kwargs)

        # Sort based on Borda scores (lower score = higher rank)
        sorted_names = [feature_names[i] for i in np.argsort(scores_merged, kind="stable")]

        return list(sorted_names[:num_features_to_select])
