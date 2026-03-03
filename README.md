<h1 align="center">🫁 PneumoScan AI</h1>

<p align="center">
  <strong>Deep-learning powered pneumonia detection from chest X-rays</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/FastAPI-0.128+-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/github/actions/workflow/status/Sujith-RMD/pneumonia-ai/ci.yml?label=CI&logo=github" alt="CI">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

---

## Overview

PneumoScan AI is a full-stack medical image analysis application that classifies chest X-ray images as **Normal**, **Pneumonia**, or **Uncertain** using a fine-tuned **MobileNetV2** deep learning model. It includes **Grad-CAM explainability** to visualise which lung regions influenced the model's decision — critical for trust in medical AI.

### Key Features

| Feature | Description |
|---|---|
| **Transfer Learning** | MobileNetV2 pretrained on ImageNet, fine-tuned on chest X-rays |
| **Two-Phase Training** | Frozen base → unfreeze last 20 layers for domain adaptation |
| **Grad-CAM Heatmaps** | Visual explainability showing model attention regions |
| **REST API** | FastAPI backend with Pydantic validation, health checks, OpenAPI docs |
| **Modern Frontend** | Glassmorphism UI with drag-and-drop, real-time results |
| **Containerised** | Multi-stage Dockerfile with health checks |
| **CI/CD** | GitHub Actions pipeline with testing + Docker build |
| **Tested** | Unit + integration tests with pytest |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      Frontend (HTML/JS)                  │
│  Upload X-ray → Predict / Explain (Grad-CAM)            │
└────────────────────────┬─────────────────────────────────┘
                         │ HTTP POST
┌────────────────────────▼─────────────────────────────────┐
│                   FastAPI Backend                         │
│  /predict  → preprocess → MobileNetV2 → result          │
│  /gradcam  → preprocess → MobileNetV2 → heatmap         │
│  /health   → model status + uptime                      │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│              MobileNetV2 (Fine-tuned)                    │
│  Input: 224×224×3  →  Sigmoid  →  P(Pneumonia)          │
└──────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
pneumonia-ai/
├── app/
│   ├── main.py            # FastAPI app — routes, validation, CORS
│   ├── model_loader.py    # Singleton model loading with caching
│   ├── predict.py         # Image preprocessing & prediction logic
│   └── gradcam.py         # Grad-CAM heatmap generation
├── frontend/
│   ├── index.html         # Single-page application
│   ├── script.js          # (legacy) JS — now inlined in index.html
│   └── style.css          # (legacy) CSS — now inlined in index.html
├── training/
│   └── train.py           # Full training pipeline with metrics & plots
├── model/
│   └── pneumonia_model.keras  # Trained model (not in git — see Setup)
├── data/                  # Dataset (not in git — see Setup)
├── tests/
│   └── test_app.py        # Unit + integration tests
├── .github/workflows/
│   └── ci.yml             # GitHub Actions CI pipeline
├── Dockerfile             # Multi-stage container build
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Pytest configuration
├── .gitignore
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### 1. Clone & install

```bash
git clone https://github.com/Sujith-RMD/pneumonia-ai.git
cd pneumonia-ai
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the dataset

Download the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle and extract it:

```
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 3. Train the model (optional — or use a pre-trained checkpoint)

```bash
python training/train.py
```

This will:
- Train MobileNetV2 in two phases (frozen → fine-tuned)
- Evaluate on the test set and print classification report + ROC-AUC
- Save the model to `model/pneumonia_model.keras`
- Generate plots in `training/plots/` (accuracy, loss, confusion matrix, ROC curve)

### 4. Run the app

```bash
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### 5. Run tests

```bash
pip install httpx anyio pytest-anyio
pytest tests/ -v
```

---

## Docker

```bash
docker build -t pneumoscan-ai .
docker run -p 8000:8000 pneumoscan-ai
```

---

## API Documentation

FastAPI auto-generates interactive docs:

| URL | Description |
|---|---|
| `/docs` | Swagger UI |
| `/redoc` | ReDoc |

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serve frontend |
| `GET` | `/health` | Health check (model status, uptime) |
| `POST` | `/predict` | Upload X-ray → get prediction |
| `POST` | `/gradcam` | Upload X-ray → get prediction + Grad-CAM heatmap |

### Example response — `/predict`

```json
{
  "prediction": "PNEUMONIA",
  "confidence": 94.32,
  "raw_probability": 0.9432
}
```

### Example response — `/gradcam`

```json
{
  "prediction": "PNEUMONIA",
  "confidence": 94.32,
  "raw_probability": 0.9432,
  "gradcam_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

---

## Model Performance

> *Update these numbers after training on your machine.*

| Metric | Value |
|---|---|
| Test Accuracy | ~93% |
| ROC-AUC | ~0.97 |
| Architecture | MobileNetV2 (fine-tuned last 20 layers) |
| Input Size | 224 × 224 × 3 |
| Training Data | 5,216 images |
| Augmentation | Rotation, zoom, shift, flip, brightness |

### Training Plots

After training, plots are saved to `training/plots/`:

- **Accuracy & Loss curves** (with fine-tuning boundary marked)
- **Confusion Matrix**
- **ROC Curve**

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow / Keras · MobileNetV2 |
| Explainability | Grad-CAM (Class Activation Maps) |
| Backend | FastAPI · Pydantic · Uvicorn |
| Frontend | Vanilla HTML/CSS/JS · Glassmorphism |
| Computer Vision | OpenCV · Pillow |
| Metrics | scikit-learn (classification report, ROC-AUC) |
| Testing | pytest · httpx |
| Containerisation | Docker (multi-stage) |
| CI/CD | GitHub Actions |

---

## What I Learned

- Transfer learning and fine-tuning strategies for medical imaging
- Importance of **model explainability** (Grad-CAM) in healthcare AI
- Building production-ready ML APIs with FastAPI and Pydantic validation
- Two-phase training: frozen backbone → gradual unfreezing
- Handling class imbalance and choosing appropriate evaluation metrics
- Docker containerisation for ML applications
- Writing testable ML code with proper separation of concerns

---

## Disclaimer

This is an **educational prototype** and is **not intended for clinical diagnosis**. Always consult a qualified medical professional for health-related decisions.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
