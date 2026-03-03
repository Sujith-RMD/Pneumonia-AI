"""
Training script for PneumoScan AI — Chest X-ray Pneumonia Classifier.

Architecture : MobileNetV2 (transfer learning) + fine-tuning
Dataset      : Kaggle Chest X-Ray Images (Pneumonia)
Strategy     : Two-phase training → frozen base → unfreeze last-20 layers
Outputs      : Saved model  →  model/pneumonia_model.keras
               Plots        →  training/plots/  (accuracy, loss, confusion matrix, ROC)
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras import layers, models  # type: ignore[import-untyped]
from tensorflow.keras.applications import MobileNetV2  # type: ignore[import-untyped]
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore[import-untyped]
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore[import-untyped]

# ==========================
# CONFIG
# ==========================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10  # frozen base
EPOCHS_PHASE2 = 10  # fine-tuning
LEARNING_RATE_FT = 1e-5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = str(PROJECT_ROOT / "data" / "train")
VAL_DIR = str(PROJECT_ROOT / "data" / "val")
TEST_DIR = str(PROJECT_ROOT / "data" / "test")
MODEL_DIR = str(PROJECT_ROOT / "model")
PLOT_DIR = str(Path(__file__).resolve().parent / "plots")

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# ==========================
# DATA GENERATORS
# ==========================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=25,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    shear_range=0.2,
)

val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

# ==========================
# MODEL ARCHITECTURE
# ==========================
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # Freeze base model

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs=base_model.input, outputs=output)

# ==========================
# CALLBACKS
# ==========================
early_stop = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
)

# ==========================
# PHASE 1 — TRAIN (frozen base)
# ==========================
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print("\n[Phase 1] Training with frozen base …\n")
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE1,
    callbacks=[early_stop, reduce_lr],
)

# ==========================
# PHASE 2 — FINE-TUNING
# ==========================
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE_FT),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print("\n[Phase 2] Fine-tuning last 20 layers …\n")
history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE2,
    callbacks=[early_stop, reduce_lr],
)

# ==========================
# EVALUATE ON TEST SET
# ==========================
loss, acc = model.evaluate(test_generator)
print(f"\n{'='*50}")
print(f"Test Accuracy : {acc * 100:.2f}%")
print(f"Test Loss     : {loss:.4f}")
print(f"{'='*50}\n")

# Predictions for metrics
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator, verbose=0).ravel()
y_pred = (y_pred_probs >= 0.5).astype(int)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# ROC-AUC
roc_auc = roc_auc_score(y_true, y_pred_probs)
print(f"ROC-AUC Score: {roc_auc:.4f}\n")

# ==========================
# SAVE MODEL
# ==========================
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(os.path.join(MODEL_DIR, "pneumonia_model.keras"))
print("Model saved to model/pneumonia_model.keras")

# ==========================
# GENERATE & SAVE PLOTS
# ==========================
os.makedirs(PLOT_DIR, exist_ok=True)


def _merge_histories(h1, h2):
    """Combine two Keras History objects."""
    merged = {}
    for key in h1.history:
        merged[key] = h1.history[key] + h2.history.get(key, [])
    return merged


history = _merge_histories(history1, history2)

# 1. Accuracy curve
plt.figure(figsize=(10, 5))
plt.plot(history["accuracy"], label="Train Accuracy")
plt.plot(history["val_accuracy"], label="Val Accuracy")
plt.axvline(x=len(history1.history["accuracy"]) - 0.5, color="gray", linestyle="--", label="Fine-tune start")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "accuracy.png"), dpi=150)
plt.close()

# 2. Loss curve
plt.figure(figsize=(10, 5))
plt.plot(history["loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.axvline(x=len(history1.history["loss"]) - 0.5, color="gray", linestyle="--", label="Fine-tune start")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "loss.png"), dpi=150)
plt.close()

# 3. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# 4. ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "roc_curve.png"), dpi=150)
plt.close()

print(f"\nPlots saved to {PLOT_DIR}/")
print("Training complete.")

