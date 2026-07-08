import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from moosefs.metrics import (
    Accuracy,
    F1Score,
    LogLoss,
    MeanAbsoluteError,
    MeanSquaredError,
    PrecisionScore,
    R2Score,
    RecallScore,
)


# Test data for classification
@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Test data for regression
@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Classification Metrics Tests
def test_log_loss(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    metric = LogLoss()
    result = metric.compute(X_train, y_train, X_test, y_test)
    assert result < 1.0, f"Unexpected log loss: {result}"


def test_f1_score(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    metric = F1Score()
    result = metric.compute(X_train, y_train, X_test, y_test)
    assert 0.0 <= result <= 1.0, f"Unexpected Macro F1 Score: {result}"


def test_accuracy(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    metric = Accuracy()
    result = metric.compute(X_train, y_train, X_test, y_test)
    assert 0.0 <= result <= 1.0, f"Unexpected Accuracy: {result}"


def test_precision_score(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    metric = PrecisionScore()
    result = metric.compute(X_train, y_train, X_test, y_test)
    assert 0.0 <= result <= 1.0, f"Unexpected Precision Score: {result}"


def test_recall_score(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    metric = RecallScore()
    result = metric.compute(X_train, y_train, X_test, y_test)
    assert 0.0 <= result <= 1.0, f"Unexpected Recall Score: {result}"


def test_logistic_regression_metric_model_is_scaled():
    from sklearn.pipeline import Pipeline

    metric = Accuracy()
    model = metric.models["Logistic Regression"]
    assert isinstance(model, Pipeline)
    assert model.steps[0][0] == "standardscaler"


def test_model_signature_shared_across_metric_instances():
    # The fold-level model cache relies on identical signatures across
    # metric instances that use the same model battery.
    assert Accuracy().model_signature() == F1Score().model_signature()


# Regression Metrics Tests
def test_mean_squared_error(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    metric = MeanSquaredError()
    result = metric.compute(X_train, y_train, X_test, y_test)
    assert result < 0, "Result of compute by MSE should be negative"


def test_mean_absolute_error(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    metric = MeanAbsoluteError()
    result = metric.compute(X_train, y_train, X_test, y_test)
    assert result < 0, "Result of compute by MAE should be negative"


def test_r2_score(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    metric = R2Score()
    result = metric.compute(X_train, y_train, X_test, y_test)
    assert -1.0 <= result <= 1.0, f"Unexpected R2 Score: {result}"
