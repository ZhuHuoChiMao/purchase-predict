from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from kedro_datasets.pickle import PickleDataset
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "data" / "06_models" / "deployment_model.pkl"
ENCODER_PATH = PROJECT_ROOT / "data" / "04_feature" / "transform_pipeline.pkl"
PURCHASE_THRESHOLD = 0.5
MODEL_URI_ENV = "PURCHASE_MODEL_URI"
MLFLOW_TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"

app = FastAPI(title="purchase-predict")


class PurchaseRequest(BaseModel):
    product_id: int
    brand: str | None = None
    price: float
    user_id: int
    user_session: str
    num_views_session: int = Field(ge=0)
    num_views_product: int = Field(ge=0)
    category: str | None = None
    sub_category: str | None = None
    hour: int = Field(ge=0, le=23)
    minute: int = Field(ge=0, le=59)
    weekday: int = Field(ge=0)
    duration: int = Field(ge=0)
    num_prev_sessions: int = Field(ge=0)
    num_prev_product_views: int = Field(ge=0)


def _load_pickle(path: Path) -> Any:
    return PickleDataset(filepath=str(path)).load()


def _encode_unknown_safe(value: str, encoder: Any) -> int:
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    if "unknown" in encoder.classes_:
        return int(encoder.transform(["unknown"])[0])
    return 0


def _prepare_features(payload: PurchaseRequest) -> pd.DataFrame:
    row = payload.model_dump()
    row.pop("user_id")
    row.pop("user_session")
    features = pd.DataFrame([row])
    encoders = _load_pickle(ENCODER_PATH)

    for column in ["category", "sub_category", "brand"]:
        value = features.loc[0, column]
        value = "unknown" if value is None else str(value)
        features.loc[0, column] = _encode_unknown_safe(value, encoders[column])

    return features.astype(
        {
            "product_id": "int64",
            "brand": "int64",
            "category": "int64",
            "sub_category": "int64",
            "weekday": "int64",
        }
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PurchaseRequest) -> dict[str, float | int]:
    model = _load_model()
    features = _prepare_model_input(payload)
    probability = _predict_probability(model, features)
    return {"purchased": int(probability >= PURCHASE_THRESHOLD), "probability": probability}


def _prepare_model_input(payload: PurchaseRequest) -> pd.DataFrame:
    if os.getenv(MODEL_URI_ENV):
        return pd.DataFrame([payload.model_dump()])
    return _prepare_features(payload)


def _load_model() -> Any:
    model_uri = os.getenv(MODEL_URI_ENV)
    if model_uri:
        tracking_uri = os.getenv(MLFLOW_TRACKING_URI_ENV)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        return mlflow.pyfunc.load_model(model_uri)

    artifact = _load_pickle(MODEL_PATH)
    return artifact["model"] if isinstance(artifact, dict) and "model" in artifact else artifact


def _predict_probability(model: Any, features: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(features)[:, 1][0])
    return float(model.predict(features)[0])
