"""
Image preprocessing and prediction utilities.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

IMG_SIZE = 224
ALLOWED_EXTENSIONS = {"image/jpeg", "image/png", "image/bmp", "image/tiff"}

# Thresholds for classification confidence
PNEUMONIA_THRESHOLD = 0.75
NORMAL_THRESHOLD = 0.25


@dataclass
class PredictionResult:
    """Structured prediction output."""
    label: str
    confidence: float
    raw_probability: float


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """
    Read raw bytes → PIL Image → normalised NumPy array ready for inference.

    Returns:
        np.ndarray of shape (1, 224, 224, 3) with values in [0, 1].

    Raises:
        ValueError: If the image cannot be processed.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Unable to open image: {exc}") from exc

    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(model, img_array: np.ndarray) -> PredictionResult:
    """
    Run inference and return a structured result.

    Uses a three-tier threshold to separate NORMAL / UNCERTAIN / PNEUMONIA.
    """
    prediction = float(model.predict(img_array, verbose=0)[0][0])

    if prediction >= PNEUMONIA_THRESHOLD:
        label = "PNEUMONIA"
        confidence = prediction
    elif prediction <= NORMAL_THRESHOLD:
        label = "NORMAL"
        confidence = 1.0 - prediction
    else:
        label = "UNCERTAIN"
        confidence = max(prediction, 1.0 - prediction)

    return PredictionResult(
        label=label,
        confidence=round(confidence * 100, 2),
        raw_probability=round(prediction, 4),
    )

