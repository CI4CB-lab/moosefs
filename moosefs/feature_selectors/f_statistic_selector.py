from typing import Any

import numpy as np
from sklearn.feature_selection import f_classif, f_regression

from .base_selector import FeatureSelector


class FStatisticSelector(FeatureSelector):
    """Feature selector using F-statistic scores."""

    name = "FStatistic"

    def __init__(self, task: str, num_features_to_select: int, **kwargs: Any) -> None:
        """
        Args:
            task: ML task ('classification' or 'regression').
            num_features_to_select: Number of features to select.
            **kwargs: Additional arguments for the scoring function.
        """
        super().__init__(task, num_features_to_select)
        self.kwargs = kwargs

    def compute_scores(self, X: Any, y: Any) -> np.ndarray:
        """
        Computes F-statistic scores.

        Args:
            X: Training samples.
            y: Target values.

        Returns:
            Feature-aligned F-statistic scores. Constant features may have NaN
            scores, while perfect separation may produce positive infinity;
            ``select_features`` handles both according to the base score
            contract.

        Raises:
            ValueError: If task is not 'classification' or 'regression'.
        """
        score_func = {"classification": f_classif, "regression": f_regression}.get(self.task)
        if score_func is None:
            raise ValueError("Task must be 'classification' or 'regression'.")
        scores, _ = score_func(X, y, **self.kwargs)
        return scores
