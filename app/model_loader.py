"""
Model loader module — handles loading and caching the trained Keras model.
"""

import logging
import os
from pathlib import Path

import tensorflow as tf

logger = logging.getLogger(__name__)

# Resolve model path relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = os.getenv("MODEL_PATH", str(_PROJECT_ROOT / "model" / "pneumonia_model.keras"))

_model: tf.keras.Model | None = None


def load_model() -> tf.keras.Model:
    """Load the Keras model from disk (singleton pattern)."""
    global _model
    if _model is None:
        logger.info("Loading model from %s …", MODEL_PATH)
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Run training/train.py first or set MODEL_PATH env var."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully — input shape: %s", _model.input_shape)
    return _model


def get_model() -> tf.keras.Model:
    """Return the cached model instance."""
    return load_model()

