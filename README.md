# Food-11 Classification — MLOps Labs

An image classification project using PyTorch, MLflow, FastAPI, and Docker.

## Lab 1 — Data Versioning
- Git for source control
- DVC for dataset versioning

## Lab 2 — Model Training
- ResNet18 trained on Food-11
- MLflow experiment tracking
- Best validation accuracy: 80.30%
- Registered model: food11
- Model version: 1
- Model alias: champion

## Lab 3 — Docker Model Serving
- FastAPI inference service
- Docker containerization
- Image upload and prediction endpoint
- Health check endpoint

### Build the Docker image

docker build -t food11-api:latest .

### Start the MLflow server

Run from the repository root:

uv run mlflow server --host 0.0.0.0 --port 5000 --workers 1 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --allowed-hosts "localhost,127.0.0.1,host.docker.internal,host.docker.internal:5000,localhost:5000,127.0.0.1:5000"

### Run the inference container

Run from the repository root in PowerShell:

docker run --name food11-container -p 127.0.0.1:8001:8000 -v "${PWD}/mlruns:/models:ro" -e MODEL_URI=/models/1/models/m-862f468a32214f1c81ccbb135f1621db/artifacts -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 food11-api:latest

The model artifact directory is mounted from the local MLflow store. This setup requires the registered model artifacts to exist on the host.

### API Endpoints

Health check:
http://127.0.0.1:8001/health

Interactive API documentation:
http://127.0.0.1:8001/docs

Prediction:
POST http://127.0.0.1:8001/predict

Upload a food image using the multipart form field named "file".

Example response:

{
  "predicted_class": "Dairy product",
  "confidence": 0.3284
}

### Verification

The API was successfully tested inside Docker:
- GET /health returned HTTP 200
- POST /predict returned HTTP 200
- Food image classification completed successfully

### Deployment Note

The tested Lab 3 setup used bind mounts for both the model artifacts and updated source code. A fully self-contained deployment has not yet been verified.
