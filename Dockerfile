FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY requirements-docker.txt ./

RUN uv pip install --system --no-cache \
    --index https://pypi.org/simple \
    --index https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    -r requirements-docker.txt

COPY src/ ./src/

ENV PYTHONPATH=/app/src
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

EXPOSE 8000

CMD ["uvicorn", "food11.serve:app", "--host", "0.0.0.0", "--port", "8000"]
