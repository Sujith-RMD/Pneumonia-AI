"""
Unit & integration tests for PneumoScan AI.

Run with:  pytest tests/ -v
"""

import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Ensure app package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────
@pytest.fixture
def dummy_image_bytes() -> bytes:
    """Create a minimal 224×224 RGB PNG in memory."""
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


@pytest.fixture
def dummy_jpeg_bytes() -> bytes:
    """Create a minimal JPEG image."""
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


# ──────────────────────────────────────────────
# Unit tests — preprocessing
# ──────────────────────────────────────────────
class TestPreprocessImage:
    def test_output_shape(self, dummy_image_bytes):
        from app.predict import preprocess_image

        result = preprocess_image(dummy_image_bytes)
        assert result.shape == (1, 224, 224, 3)

    def test_output_range(self, dummy_image_bytes):
        from app.predict import preprocess_image

        result = preprocess_image(dummy_image_bytes)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_dtype_float(self, dummy_image_bytes):
        from app.predict import preprocess_image

        result = preprocess_image(dummy_image_bytes)
        assert result.dtype == np.float32

    def test_invalid_bytes_raises(self):
        from app.predict import preprocess_image

        with pytest.raises(ValueError, match="Unable to open image"):
            preprocess_image(b"not-an-image")


# ──────────────────────────────────────────────
# Unit tests — prediction logic
# ──────────────────────────────────────────────
class TestPredictImage:
    def _make_mock_model(self, sigmoid_value: float):
        mock = MagicMock()
        mock.predict.return_value = np.array([[sigmoid_value]])
        return mock

    def test_pneumonia(self):
        from app.predict import predict_image

        model = self._make_mock_model(0.90)
        result = predict_image(model, np.zeros((1, 224, 224, 3)))
        assert result.label == "PNEUMONIA"
        assert result.confidence == 90.0

    def test_normal(self):
        from app.predict import predict_image

        model = self._make_mock_model(0.10)
        result = predict_image(model, np.zeros((1, 224, 224, 3)))
        assert result.label == "NORMAL"
        assert result.confidence == 90.0

    def test_uncertain(self):
        from app.predict import predict_image

        model = self._make_mock_model(0.50)
        result = predict_image(model, np.zeros((1, 224, 224, 3)))
        assert result.label == "UNCERTAIN"

    def test_boundary_pneumonia(self):
        from app.predict import predict_image

        model = self._make_mock_model(0.75)
        result = predict_image(model, np.zeros((1, 224, 224, 3)))
        assert result.label == "PNEUMONIA"

    def test_boundary_normal(self):
        from app.predict import predict_image

        model = self._make_mock_model(0.25)
        result = predict_image(model, np.zeros((1, 224, 224, 3)))
        assert result.label == "NORMAL"


# ──────────────────────────────────────────────
# Integration tests — FastAPI endpoints
# ──────────────────────────────────────────────
class TestAPI:
    @pytest.fixture(autouse=True)
    def _setup_client(self, dummy_jpeg_bytes):
        """Patch model loading and create a test client."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.92]])
        mock_model.input_shape = (None, 224, 224, 3)
        mock_model.layers = []

        with patch("app.model_loader.load_model", return_value=mock_model):
            with patch("app.model_loader._model", mock_model):
                with patch("app.model_loader.get_model", return_value=mock_model):
                    from httpx import ASGITransport, AsyncClient
                    from app.main import app

                    self.app = app
                    self.mock_model = mock_model
                    self.jpeg_bytes = dummy_jpeg_bytes
                    yield

    @pytest.mark.anyio
    async def test_health_endpoint(self):
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(transport=ASGITransport(app=self.app), base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "uptime_seconds" in data

    @pytest.mark.anyio
    async def test_predict_valid_image(self):
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(transport=ASGITransport(app=self.app), base_url="http://test") as client:
            resp = await client.post(
                "/predict",
                files={"file": ("test.jpg", self.jpeg_bytes, "image/jpeg")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] == "PNEUMONIA"
        assert "confidence" in data
        assert "raw_probability" in data

    @pytest.mark.anyio
    async def test_predict_invalid_file_type(self):
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(transport=ASGITransport(app=self.app), base_url="http://test") as client:
            resp = await client.post(
                "/predict",
                files={"file": ("test.txt", b"hello", "text/plain")},
            )
        assert resp.status_code == 400
