from typing import Any, Optional

from joblib import hash as joblib_hash
import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


class BaseMetric:
    """Base class for computing evaluation metrics.

    Trains a small battery of models and aggregates per-model metric values.
    """

    def __init__(self, name: str, task: str) -> None:
        """Initialize the metric with a task type.

        Args:
            name: Human-readable metric name.
            task: Either "classification" or "regression".
        """
        if task not in {"classification", "regression"}:
            raise ValueError("Task must be 'classification' or 'regression'.")

        self.name = name
        self.task = task
        self.models = self._initialize_models()

    def model_signature(self) -> str:
        """Return a stable signature describing the internal model set."""
        signature_payload = {
            name: (
                f"{model.__class__.__module__}.{model.__class__.__qualname__}",
                model.get_params(deep=True),
            )
            for name, model in self.models.items()
        }
        return f"{self.task}:{joblib_hash(signature_payload)}"

    def _initialize_models(self) -> dict:
        """Initialize task-specific models.

        Returns:
            Mapping from model label to estimator instance.
        """
        # Keep inner models single-threaded to avoid nested parallelism.
        return {
            "classification": {
                "Random Forest": RandomForestClassifier(n_jobs=1),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Gradient Boosting": GradientBoostingClassifier(),
            },
            "regression": {
                "Random Forest": RandomForestRegressor(n_jobs=1),
                "Linear Regression": LinearRegression(),
                "Gradient Boosting": GradientBoostingRegressor(),
            },
        }[self.task]

    def train_and_predict(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
    ) -> dict:
        """Train all models and generate predictions.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_test: Test features.
            y_test: Test targets.

        Returns:
            Dict keyed by model name with predictions and optional probabilities.
        """
        results = {}

        for model_name, model in self.models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test) if self.task == "classification" else None
            results[model_name] = {
                "predictions": predictions,
                "probabilities": probabilities,
            }

        return results

    def compute(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
    ) -> float:
        """Compute the metric (implemented by subclasses)."""
        raise NotImplementedError("This method must be overridden in subclasses.")


class RegressionMetric(BaseMetric):
    """Base class for regression metrics."""

    def __init__(self, name: str) -> None:
        super().__init__(name, task="regression")

    def compute(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
    ) -> float:
        """Average the metric over the internal model set."""
        results = self.train_and_predict(X_train, y_train, X_test, y_test)
        return self.aggregate_from_results(y_test, results)

    def aggregate_from_results(self, y_test: np.ndarray, results: dict) -> float:
        """Aggregate metric value from cached prediction results."""
        return float(np.mean([self._metric_func(y_test, res["predictions"]) for res in results.values()]))

    def _metric_func(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Metric function to be overridden by subclasses."""
        raise NotImplementedError("This method must be overridden in subclasses.")


class R2Score(RegressionMetric):
    def __init__(self) -> None:
        super().__init__("R2 Score")

    def _metric_func(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return r2_score(y_true, y_pred)


class MeanAbsoluteError(RegressionMetric):
    def __init__(self) -> None:
        super().__init__("Mean Absolute Error")

    def _metric_func(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return -mean_absolute_error(y_true, y_pred)  # Return negative MAE


class MeanSquaredError(RegressionMetric):
    def __init__(self) -> None:
        super().__init__("Mean Squared Error")

    def _metric_func(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return -mean_squared_error(y_true, y_pred)  # Return negative MSE


class ClassificationMetric(BaseMetric):
    """Base class for classification metrics."""

    def __init__(self, name: str) -> None:
        super().__init__(name, task="classification")

    def compute(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
    ) -> float:
        """Average the metric over the internal model set."""
        results = self.train_and_predict(X_train, y_train, X_test, y_test)
        return self.aggregate_from_results(y_test, results)

    def aggregate_from_results(self, y_test: np.ndarray, results: dict) -> float:
        """Aggregate metric value from cached prediction results."""
        return float(
            np.mean(
                [self._metric_func(y_test, res["predictions"], res.get("probabilities")) for res in results.values()]
            )
        )

    def _metric_func(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> float:
        """Metric function to be overridden by subclasses."""
        raise NotImplementedError("This method must be overridden in subclasses.")


class LogLoss(ClassificationMetric):
    def __init__(self) -> None:
        super().__init__("Log Loss")

    def _metric_func(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> float:
        return -log_loss(y_true, y_proba)


class F1Score(ClassificationMetric):
    def __init__(self) -> None:
        super().__init__("F1 Score")

    def _metric_func(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: None = None) -> float:
        return f1_score(y_true, y_pred, average="macro")


class Accuracy(ClassificationMetric):
    def __init__(self) -> None:
        super().__init__("Accuracy")

    def _metric_func(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: None = None) -> float:
        return accuracy_score(y_true, y_pred)


class PrecisionScore(ClassificationMetric):
    def __init__(self) -> None:
        super().__init__("Precision Score")

    def _metric_func(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: None = None) -> float:
        return precision_score(y_true, y_pred, average="macro", zero_division=0)


class RecallScore(ClassificationMetric):
    def __init__(self) -> None:
        super().__init__("Recall Score")

    def _metric_func(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: None = None) -> float:
        return recall_score(y_true, y_pred, average="macro")
