# Deployment

This project can run locally with bundled model artifacts, or in Cloud Run/GKE
with a model loaded from MLflow.

Required values for cloud deployment:

- `PROJECT_ID`: Google Cloud project ID.
- `_REGION`: Artifact Registry and Cloud Run region.
- `_ARTIFACT_REPOSITORY`: Docker Artifact Registry repository.
- `_MLFLOW_TRACKING_URI`: MLflow tracking server URL.
- `_PURCHASE_MODEL_URI`: MLflow model URI, for example `models:/purchase_predict@production`.
- `_SERVICE_ACCOUNT`: optional Cloud Run service account email.

Current project values:

- `PROJECT_ID=project-3139c4ea-e811-47b0-aad`
- `_REGION=europe-west1`
- `_ARTIFACT_REPOSITORY=cloud-run-source-deploy`
- `_SERVICE_NAME=purchase-api`
- `_MLFLOW_TRACKING_URI=http://104.197.20.252:5000`
- `_PURCHASE_MODEL_URI=models:/purchase_predict@production`

Register a new model version in MLflow:

```bash
uv run kedro run --env cloud
```

Cloud Build example:

```bash
gcloud builds submit \
  --config cloudbuild.yaml \
  --project project-3139c4ea-e811-47b0-aad \
  --substitutions _REGION=europe-west1,_ARTIFACT_REPOSITORY=cloud-run-source-deploy,_SERVICE_NAME=purchase-api,_MLFLOW_TRACKING_URI=http://104.197.20.252:5000,_PURCHASE_MODEL_URI=models:/purchase_predict@production
```

Kubernetes example:

```bash
kubectl apply -f k8s/secret.example.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

For the full GKE workflow, see `deploy/GKE.md`.
