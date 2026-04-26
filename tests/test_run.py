from pathlib import Path

from kedro.framework.project import pipelines
from kedro.framework.startup import bootstrap_project


def test_project_pipelines_are_registered():
    bootstrap_project(Path.cwd())

    assert {"loading", "processing", "training", "deployment", "__default__"} <= set(
        pipelines.keys()
    )
