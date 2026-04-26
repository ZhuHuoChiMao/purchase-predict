from kedro.pipeline import Node, Pipeline

from .nodes import load_csv_files


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                load_csv_files,
                ["params:raw_input_path", "params:raw_input_pattern"],
                "primary",
                name="load_csv_files_node",
            )
        ]
    )
