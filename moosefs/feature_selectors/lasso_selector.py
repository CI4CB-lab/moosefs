from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .base_selector import FeatureSelector


class LassoSelector(FeatureSelector):
    """Feature selector using Lasso regression."""

    name = "Lasso"

    def __init__(self, task: str, num_features_to_select: int, **kwargs: Any) -> None:
        """
        Args:
            task: ML task ('classification' or 'regression').
            num_features_to_select: Number of features to select.
            **kwargs: Additional arguments for Lasso.
        """
        super().__init__(task, num_features_to_select)
        self.kwargs = kwargs

    def compute_scores(self, X: Any, y: Any) -> np.ndarray:
        """
        Computes feature scores using Lasso regression.

        Args:
            X: Training samples.
            y: Target values.

        Returns:
            Feature scores based on absolute Lasso coefficients.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        if isinstance(y, np.ndarray) and y.ndim == 2:
            y = y.ravel()

        # Standardize features so coefficient magnitudes are comparable;
        # default alpha is 0.05 if not provided in kwargs.
        model = make_pipeline(StandardScaler(), Lasso(**{"alpha": 0.05, **self.kwargs}))
        model.fit(X, y)
        scores = np.abs(model[-1].coef_)
        return scores
