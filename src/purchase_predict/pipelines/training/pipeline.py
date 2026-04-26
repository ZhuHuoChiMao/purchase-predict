from kedro.pipeline import Node, Pipeline

from .nodes import evaluate_model, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                train_model,
                [
                    "X_train",
                    "y_train",
                    "X_test",
                    "y_test",
                    "params:models",
                    "params:random_state",
                    "params:mlflow_enabled",
                    "params:mlflow_experiment_name",
                    "params:mlflow_tracking_uri",
                ],
                "trained_model",
                name="train_model_node",
            ),
            Node(
                evaluate_model,
                ["trained_model", "X_test", "y_test"],
                "metrics",
                name="evaluate_model_node",
            ),
        ]
    )
