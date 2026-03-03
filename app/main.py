"""
PneumoScan AI — FastAPI backend for pneumonia detection from chest X-rays.

Endpoints:
    GET  /           → Serve the frontend SPA
    GET  /health     → Health check (model status, uptime)
    POST /predict    → Upload an X-ray and get NORMAL / PNEUMONIA / UNCERTAIN
    POST /gradcam    → Upload an X-ray and get a Grad-CAM heatmap overlay
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.model_loader import get_model
from app.predict import ALLOWED_EXTENSIONS, predict_image, preprocess_image
from app.gradcam import generate_gradcam

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------
_start_time = time.time()


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("Starting PneumoScan AI …")
    get_model()  # warm-up
    logger.info("Model ready.")
    yield


app = FastAPI(
    title="PneumoScan AI",
    description="Deep-learning powered pneumonia detection from chest X-rays.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the frontend dev server (or any origin during dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend assets (CSS, JS)
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    raw_probability: float


class GradCAMResponse(BaseModel):
    prediction: str
    confidence: float
    raw_probability: float
    gradcam_base64: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the main HTML page."""
    return FileResponse(str(_frontend_dir / "index.html"))


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health_check():
    """Return service health and model status."""
    try:
        model = get_model()
        loaded = model is not None
    except Exception:
        loaded = False

    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


def _validate_upload(file: UploadFile) -> None:
    """Validate that the uploaded file is an acceptable image."""
    if file.content_type not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. "
            f"Accepted: {', '.join(ALLOWED_EXTENSIONS)}",
        )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(file: UploadFile = File(...)):
    """
    Upload a chest X-ray image and receive a pneumonia prediction.

    Returns the predicted label, confidence percentage, and raw sigmoid output.
    """
    _validate_upload(file)
    contents = await file.read()

    try:
        img_array = preprocess_image(contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    model = get_model()
    result = predict_image(model, img_array)

    logger.info("Prediction: %s (%.2f%%) — file: %s", result.label, result.confidence, file.filename)

    return PredictionResponse(
        prediction=result.label,
        confidence=result.confidence,
        raw_probability=result.raw_probability,
    )


@app.post("/gradcam", response_model=GradCAMResponse, tags=["explainability"])
async def gradcam(file: UploadFile = File(...)):
    """
    Upload a chest X-ray and receive a Grad-CAM heatmap + prediction.

    The heatmap highlights the regions the model focused on,
    providing visual explainability for the prediction.
    """
    _validate_upload(file)
    contents = await file.read()

    try:
        img_array = preprocess_image(contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    model = get_model()
    result = predict_image(model, img_array)

    try:
        heatmap_b64 = generate_gradcam(model, img_array)
    except Exception as exc:
        logger.error("Grad-CAM failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {exc}")

    return GradCAMResponse(
        prediction=result.label,
        confidence=result.confidence,
        raw_probability=result.raw_probability,
        gradcam_base64=heatmap_b64,
    )

