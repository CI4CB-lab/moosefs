from typing import Any

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from .base_selector import FeatureSelector


class SVMSelector(FeatureSelector):
    """Feature selector using SVM coefficients."""

    name = "SVM"

    def __init__(self, task: str, num_features_to_select: int, **kwargs: Any) -> None:
        """
        Args:
            task: ML task ('classification' or 'regression').
            num_features_to_select: Number of features to select.
            **kwargs: Additional arguments for the SVM model.
        """
        super().__init__(task, num_features_to_select)
        self.kwargs = kwargs

    def compute_scores(self, X: Any, y: Any) -> np.ndarray:
        """
        Computes feature importances using an SVM model.

        Args:
            X: Training samples.
            y: Target values.

        Returns:
            Feature importances derived from SVM model coefficients.

        Raises:
            ValueError: If task is not 'classification' or 'regression'.
        """
        model_cls = {"classification": SVC, "regression": SVR}.get(self.task)
        if model_cls is None:
            raise ValueError("Task must be 'classification' or 'regression'.")

        # Only remove `random_state` for SVR
        filtered_kwargs = (
            {k: v for k, v in self.kwargs.items() if k != "random_state"} if self.task == "regression" else self.kwargs
        )

        # Standardize features so coefficient magnitudes are comparable.
        model = make_pipeline(StandardScaler(), model_cls(kernel="linear", **filtered_kwargs))
        model.fit(X, y)
        scores = np.abs(model[-1].coef_[0])
        return scores
