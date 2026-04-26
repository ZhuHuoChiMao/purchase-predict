import pandas as pd

from purchase_predict.pipelines.training.nodes import evaluate_model, train_model

MIN_EXPECTED_F1 = 0.9


def test_train_model_returns_best_artifact():
    X = pd.DataFrame({"x1": [0, 0, 1, 1, 0, 1], "x2": [0, 1, 0, 1, 0, 1]})
    y = pd.Series([0, 0, 1, 1, 0, 1], name="purchased")
    models = [
        {
            "name": "random_forest",
            "params": {"n_estimators": 10, "max_depth": 3},
        }
    ]

    artifact = train_model(
        X,
        y,
        X,
        y,
        models=models,
        random_state=40,
        mlflow_enabled=False,
        mlflow_experiment_name="purchase_predict_test",
        mlflow_tracking_uri=None,
    )
    metrics = evaluate_model(artifact, X, y)

    assert artifact["name"] == "random_forest"
    assert metrics["f1"] >= MIN_EXPECTED_F1
