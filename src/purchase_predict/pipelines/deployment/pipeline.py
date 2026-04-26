from kedro.pipeline import Node, Pipeline

from .nodes import select_model_for_deployment


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                select_model_for_deployment,
                [
                    "trained_model",
                    "transform_pipeline",
                    "metrics",
                    "params:minimum_deployment_f1",
                    "params:mlflow_enabled",
                    "params:mlflow_registered_model_name",
                    "params:mlflow_model_alias",
                    "params:mlflow_tracking_uri",
                ],
                "deployment_model",
                name="select_model_for_deployment_node",
            )
        ]
    )
