import numpy as np

from ..core.feature import Feature


class MergingStrategy:
    """Abstract base for merging strategies.

    Strategies can be "set-based" or "rank-based" depending on how they merge
    the per-selector outputs.
    """

    def __init__(self, strategy_type: str) -> None:
        """Initialize the strategy.

        Args:
            strategy_type: Either "set-based" or "rank-based".
        """
        self.strategy_type = strategy_type

    def merge(self, data: list, num_features_to_select: int, **kwargs) -> list:
        """Merge input data according to the strategy.

        Subclasses must implement this method.

        Args:
            data: List of Feature lists (one list per selector) or a single list.
            num_features_to_select: Number of top features to return.
            **kwargs: Strategy-specific options.

        Returns:
            A list of merged features (or names depending on strategy).

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def is_set_based(self) -> bool:
        """Return True if the strategy is set-based."""
        return self.strategy_type == "set-based"

    def is_rank_based(self) -> bool:
        """Return True if the strategy is rank-based."""
        return self.strategy_type == "rank-based"

    @staticmethod
    def _aligned_scores(subsets: list) -> tuple:
        """Align subsets by feature name into a score matrix.

        Subsets may contain different features in different orders (e.g. each
        selector's top-k), so scores must be matched by name, never by
        position.

        Args:
            subsets: List of Feature lists (one list per selector).

        Returns:
            (names, matrix): feature names in first-appearance order across
            subsets, and a float matrix of shape (n_features, n_subsets) with
            NaN where a selector did not score the feature.
        """
        names: list = []
        positions: dict = {}
        for subset in subsets:
            for feature in subset:
                if feature.name not in positions:
                    positions[feature.name] = len(names)
                    names.append(feature.name)

        matrix = np.full((len(names), len(subsets)), np.nan)
        for col, subset in enumerate(subsets):
            for feature in subset:
                matrix[positions[feature.name], col] = feature.score
        return names, matrix

    @staticmethod
    def _normalize_scores(matrix: np.ndarray) -> np.ndarray:
        """Min-max normalize each column (selector) over its present values.

        Selector scores live on incomparable scales (importances, |coefs|,
        mutual information), so each column is scaled to [0, 1] before
        aggregation. Missing entries (NaN) become 0 so features a selector
        did not score get no credit from it.
        """
        normed = np.zeros_like(matrix)
        for col in range(matrix.shape[1]):
            present = ~np.isnan(matrix[:, col])
            if not present.any():
                continue
            values = matrix[present, col]
            min_v = values.min()
            span = values.max() - min_v
            if span == 0:
                span = 1.0
            normed[present, col] = (values - min_v) / span
        return normed

    def _validate_input(self, subsets: list) -> None:
        """Validate that ``subsets`` contains Feature objects.

        Args:
            subsets: A list of Feature or a list of Feature lists.

        Raises:
            ValueError: If empty or containing invalid types.
        """
        if not subsets:
            raise ValueError("Subsets cannot be empty.")

        if isinstance(subsets[0], list):  # List of lists case
            if not all(isinstance(sub, list) and sub for sub in subsets):
                raise ValueError("Subsets cannot contain empty lists.")
            if not all(isinstance(feature, Feature) for sub in subsets for feature in sub):
                raise ValueError("Subsets must contain Feature objects.")
        else:  # Single list case
            if not all(isinstance(feature, Feature) for feature in subsets):
                raise ValueError("Subsets must contain Feature objects.")
