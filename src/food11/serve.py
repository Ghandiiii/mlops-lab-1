
import os
from contextlib import asynccontextmanager
from io import BytesIO

import mlflow
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import transforms


CLASS_NAMES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@asynccontextmanager
async def lifespan(app: FastAPI):
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = os.getenv("MODEL_URI", "models:/food11@champion")
    model = mlflow.pytorch.load_model(model_uri)
    model = model.to(DEVICE)
    model.eval()

    app.state.model = model
    app.state.device = DEVICE

    yield


app = FastAPI(
    title="Food-11 Prediction API",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Please upload an image file.",
        )

    contents = await file.read()

    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    tensor = PREPROCESS(image).unsqueeze(0).to(app.state.device)

    with torch.inference_mode():
        outputs = app.state.model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = probabilities.max(dim=1)

    index = int(predicted_idx.item())

    return {
        "predicted_class": CLASS_NAMES[index],
        "confidence": round(float(confidence.item()), 4),
    }

