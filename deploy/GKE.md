# GKE Deployment

This deployment is independent from the existing Cloud Run service. Both can run
at the same time and use the same MLflow production model:

- `MLFLOW_TRACKING_URI=http://104.197.20.252:5000`
- `PURCHASE_MODEL_URI=models:/purchase_predict@production`

## Create a GKE cluster

```bash
gcloud container clusters create purchase-predict-cluster \
  --project project-3139c4ea-e811-47b0-aad \
  --region europe-west1 \
  --num-nodes 2 \
  --machine-type e2-standard-2 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 3
```

Connect `kubectl`:

```bash
gcloud container clusters get-credentials purchase-predict-cluster \
  --project project-3139c4ea-e811-47b0-aad \
  --region europe-west1
```

## Build and deploy to GKE

From the `purchase-predict` directory:

```bash
gcloud builds submit \
  --project project-3139c4ea-e811-47b0-aad \
  --config cloudbuild-gke.yaml \
  --substitutions _REGION=europe-west1,_CLUSTER_NAME=purchase-predict-cluster,_ARTIFACT_REPOSITORY=cloud-run-source-deploy
```

## Check deployment

```bash
kubectl get pods
kubectl get service purchase-predict-api
kubectl get hpa
```

Wait until `EXTERNAL-IP` is assigned:

```bash
kubectl get service purchase-predict-api --watch
```

Test the API:

```bash
curl -X POST http://EXTERNAL-IP/predict \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": 1001,
    "brand": "unknown",
    "price": 99.99,
    "user_id": 1,
    "user_session": "test-session",
    "num_views_session": 3,
    "num_views_product": 2,
    "category": "unknown",
    "sub_category": "unknown",
    "hour": 14,
    "minute": 30,
    "weekday": 2,
    "duration": 120,
    "num_prev_sessions": 1,
    "num_prev_product_views": 5
  }'
```

## Clean up

To stop GKE costs after the notebook validation:

```bash
gcloud container clusters delete purchase-predict-cluster \
  --project project-3139c4ea-e811-47b0-aad \
  --region europe-west1
```
