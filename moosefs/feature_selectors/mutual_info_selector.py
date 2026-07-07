from typing import Any

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from .base_selector import FeatureSelector


class MutualInfoSelector(FeatureSelector):
    """Feature selector using mutual information scores."""

    name = "MutualInfo"

    def __init__(self, task: str, num_features_to_select: int, **kwargs: Any) -> None:
        """
        Args:
            task: ML task ('classification' or 'regression').
            num_features_to_select: Number of features to select.
            **kwargs: Additional arguments for mutual information function.
        """
        super().__init__(task, num_features_to_select)
        self.kwargs = kwargs

    def compute_scores(self, X: Any, y: Any) -> np.ndarray:
        """
        Computes mutual information scores.

        Args:
            X: Training samples.
            y: Target values.

        Returns:
            Mutual information scores for each feature.

        Raises:
            ValueError: If task is not 'classification' or 'regression'.
        """
        mutual_info_func = {
            "classification": mutual_info_classif,
            "regression": mutual_info_regression,
        }.get(self.task)
        if mutual_info_func is None:
            raise ValueError("Task must be 'classification' or 'regression'.")
        scores = mutual_info_func(X, y, **self.kwargs)
        return scores
