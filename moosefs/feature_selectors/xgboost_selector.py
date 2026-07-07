from typing import Any

import numpy as np
from xgboost import XGBClassifier, XGBRegressor

from .base_selector import FeatureSelector


class XGBoostSelector(FeatureSelector):
    """Feature selector using XGBoost feature importance."""

    name = "XGBoost"

    def __init__(self, task: str, num_features_to_select: int, **kwargs: Any) -> None:
        """
        Args:
            task: ML task ('classification' or 'regression').
            num_features_to_select: Number of features to select.
            **kwargs: Additional arguments for the XGBoost model.
        """
        super().__init__(task, num_features_to_select)
        self.kwargs = kwargs

    def compute_scores(self, X: Any, y: Any) -> np.ndarray:
        """
        Computes feature importances using an XGBoost model.

        Args:
            X: Training samples.
            y: Target values.

        Returns:
            Feature importances from the trained XGBoost model.

        Raises:
            ValueError: If task is not 'classification' or 'regression'.
        """
        model_cls = {"classification": XGBClassifier, "regression": XGBRegressor}.get(self.task)
        if model_cls is None:
            raise ValueError("Task must be 'classification' or 'regression'.")
        model = model_cls(**self.kwargs)
        model.fit(X, y)
        scores = model.feature_importances_
        return scores
