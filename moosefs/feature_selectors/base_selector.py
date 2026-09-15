from typing import Any

import numpy as np


class FeatureSelector:
    """Base class for feature selection.

    Subclasses must implement ``compute_scores`` returning one real numeric
    score per input feature. NaN and negative-infinite scores are undefined and
    are excluded from selection; positive infinity is a valid strongest score.
    """

    def __init__(self, task: str, num_features_to_select: int) -> None:
        """Initialize the selector.

        Args:
            task: Either "classification" or "regression".
            num_features_to_select: Number of top features to select.
        """
        self.task = task
        self.num_features_to_select = num_features_to_select

    def select_features(self, X: Any, y: Any) -> tuple:
        """Select top features using the computed scores.

        Args:
            X: Training samples, shape (n_samples, n_features).
            y: Targets, shape (n_samples,) or (n_samples, n_outputs).

        Returns:
            Tuple ``(scores, indices)``. Scores retain their original feature
            alignment, and indices contain the top-k original feature
            positions. Equal scores are ordered by increasing feature index.

        Raises:
            ValueError: If scores are not a one-dimensional real numeric array
                with one value per feature, or fewer than the requested number
                of defined scores are available. NaN and negative infinity are
                undefined; positive infinity remains selectable.
        """
        scores = self.compute_scores(X, y)
        score_array = np.asarray(scores)

        if score_array.ndim != 1:
            raise ValueError("Feature scores must be a one-dimensional array.")
        if score_array.shape[0] != X.shape[1]:
            raise ValueError(
                f"Feature scores length ({score_array.shape[0]}) must match "
                f"the number of input features ({X.shape[1]})."
            )
        if not np.issubdtype(score_array.dtype, np.number) or np.issubdtype(score_array.dtype, np.complexfloating):
            raise ValueError("Feature scores must contain real numeric values.")

        valid_mask = ~np.isnan(score_array) & ~np.isneginf(score_array)
        valid_indices = np.flatnonzero(valid_mask)
        if valid_indices.size < self.num_features_to_select:
            raise ValueError(
                f"Cannot select {self.num_features_to_select} features: only {valid_indices.size} have defined scores."
            )

        # Sort scores descending while preserving increasing original feature
        # index for ties. Positive infinity naturally ranks before finite scores.
        order = np.lexsort((-valid_indices, score_array[valid_indices]))[::-1]
        indices = valid_indices[order[: self.num_features_to_select]]
        return scores, indices

    def compute_scores(self, X: Any, y: Any) -> np.ndarray:
        """Compute per-feature scores (override in subclasses)."""
        raise NotImplementedError("Subclasses must implement compute_scores")
