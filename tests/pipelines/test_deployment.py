import pytest

from purchase_predict.pipelines.deployment.nodes import select_model_for_deployment


def test_select_model_for_deployment_enforces_threshold():
    model = {"model": object()}
    transform_pipeline = {}

    assert (
        select_model_for_deployment(
            model,
            transform_pipeline,
            {"f1": 0.8},
            0.5,
            False,
            "purchase_predict",
            "production",
            None,
        )
        is model
    )
    with pytest.raises(ValueError):
        select_model_for_deployment(
            model,
            transform_pipeline,
            {"f1": 0.2},
            0.5,
            False,
            "purchase_predict",
            "production",
            None,
        )
