from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import pandas as pd


class PurchasePredictModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper that bundles preprocessing with the estimator."""

    def __init__(self, model: Any, encoders: dict[str, Any]) -> None:
        self.model = model
        self.encoders = encoders

    def predict(self, context: Any, model_input: pd.DataFrame) -> Any:
        features = self._prepare_features(model_input.copy())
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(features)[:, 1]
        return self.model.predict(features)

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        features = features.drop(
            columns=[column for column in ["user_id", "user_session"] if column in features],
            errors="ignore",
        )
        for column in ["category", "sub_category", "brand"]:
            features[column] = features[column].astype("string").fillna("unknown")
            encoder = self.encoders[column]
            features[column] = [
                self._encode_unknown_safe(value, encoder) for value in features[column]
            ]

        features = features.fillna(0)
        if "weekday" in features:
            features["weekday"] = features["weekday"].astype(int)

        if hasattr(self.model, "feature_names_in_"):
            features = features[list(self.model.feature_names_in_)]
        return features

    @staticmethod
    def _encode_unknown_safe(value: str, encoder: Any) -> int:
        if value in encoder.classes_:
            return int(encoder.transform([value])[0])
        if "unknown" in encoder.classes_:
            return int(encoder.transform(["unknown"])[0])
        return 0
