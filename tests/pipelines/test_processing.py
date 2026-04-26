import pandas as pd

from purchase_predict.pipelines.processing.nodes import encode_features, split_dataset


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": [1, 2, 3, 4],
            "brand": ["a", "b", None, "a"],
            "price": [10.0, 12.0, 13.0, 15.0],
            "user_id": [1, 2, 3, 4],
            "user_session": ["s1", "s2", "s3", "s4"],
            "purchased": [0, 1, 0, 1],
            "num_views_session": [1, 2, 3, 4],
            "num_views_product": [1, 1, 2, 2],
            "category": ["c1", "c1", "c2", None],
            "sub_category": ["s1", "s2", None, "s1"],
            "hour": [1, 2, 3, 4],
            "minute": [1, 2, 3, 4],
            "weekday": [1, 2, 3, 4],
            "duration": [10, 20, 30, 40],
            "num_prev_sessions": [0, 0, 1, 1],
            "num_prev_product_views": [0, 1, 1, 2],
        }
    )


def test_encode_features_drops_identifiers_and_creates_encoders():
    result = encode_features(_dataset())

    assert "user_id" not in result["features"].columns
    assert "user_session" not in result["features"].columns
    assert {"category", "sub_category", "brand"} <= set(result["transform_pipeline"])


def test_split_dataset_returns_expected_outputs():
    encoded = encode_features(_dataset())["features"]

    result = split_dataset(encoded, test_ratio=0.5)

    assert set(result) == {"X_train", "y_train", "X_test", "y_test"}
    assert "purchased" not in result["X_train"].columns
