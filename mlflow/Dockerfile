FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir "mlflow>=3.16.1"

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:////app/mlflow.db", "--default-artifact-root", "/app/mlartifacts", "--serve-artifacts"]
