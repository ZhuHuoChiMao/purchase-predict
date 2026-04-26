from __future__ import annotations

from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def _as_1d(target: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(target, pd.DataFrame):
        return target.iloc[:, 0]
    return target


def train_model(  # noqa: PLR0913
    X_train: pd.DataFrame,
    y_train: pd.DataFrame | pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame | pd.Series,
    models: list[dict[str, Any]],
    random_state: int,
    mlflow_enabled: bool,
    mlflow_experiment_name: str,
    mlflow_tracking_uri: str | None,
) -> dict[str, Any]:
    """Train candidate models and return the best one by F1 score."""
    candidates = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
    }
    y_train_series = _as_1d(y_train)
    y_test_series = _as_1d(y_test)
    trained_models: list[dict[str, Any]] = []

    for spec in models:
        model_name = spec["name"]
        model_class = candidates[model_name]
        params = dict(spec.get("params", {}))
        if "random_state" in model_class().get_params():
            params.setdefault("random_state", random_state)
        model = model_class(**params)
        model.fit(X_train, y_train_series)
        predictions = model.predict(X_test)
        score = f1_score(y_test_series, predictions)
        trained_models.append(
            {
                "name": model_name,
                "model": model,
                "params": params,
                "score": score,
            }
        )

    best = max(trained_models, key=lambda item: item["score"])
    if mlflow_enabled:
        _log_training_to_mlflow(
            best,
            X_train,
            y_test_series,
            best["model"].predict(X_test),
            mlflow_experiment_name,
            mlflow_tracking_uri,
        )

    return {
        "model": best["model"],
        "name": best["name"],
        "params": best["params"],
        "score": best["score"],
    }


def evaluate_model(
    trained_model: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.DataFrame | pd.Series,
) -> dict[str, float | str]:
    """Compute classification metrics for the selected model."""
    model = trained_model["model"]
    y_test_series = _as_1d(y_test)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "model_name": trained_model["name"],
        "f1": float(f1_score(y_test_series, predictions)),
        "precision": float(precision_score(y_test_series, predictions, zero_division=0)),
        "recall": float(recall_score(y_test_series, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_series, probabilities)),
    }


def _log_training_to_mlflow(  # noqa: PLR0913
    artifact: dict[str, Any],
    X_train: pd.DataFrame,
    y_test: pd.Series,
    predictions: Any,
    experiment_name: str,
    tracking_uri: str | None,
) -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=artifact["name"]):
        mlflow.log_params(artifact["params"])
        mlflow.log_metric("f1", float(f1_score(y_test, predictions)))
        mlflow.sklearn.log_model(
            sk_model=artifact["model"],
            artifact_path="model",
            input_example=X_train.head(3),
        )
