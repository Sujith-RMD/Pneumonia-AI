"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for model explainability.

Generates heatmaps showing which regions of the chest X-ray the model
focused on when making its prediction — crucial for medical AI trust.
"""

from __future__ import annotations

import io
import base64
import logging

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

logger = logging.getLogger(__name__)


def generate_gradcam(
    model: tf.keras.Model,
    img_array: np.ndarray,
    last_conv_layer_name: str | None = None,
    alpha: float = 0.4,
) -> str:
    """
    Generate a Grad-CAM heatmap overlaid on the input image.

    Args:
        model: Trained Keras model.
        img_array: Preprocessed image array of shape (1, 224, 224, 3).
        last_conv_layer_name: Name of the last convolutional layer. Auto-detected if None.
        alpha: Overlay transparency (0 = only image, 1 = only heatmap).

    Returns:
        Base64-encoded PNG string of the heatmap overlay.
    """
    # Auto-detect last conv layer if not provided
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
            # For functional models wrapping another model (e.g., MobileNetV2)
            if hasattr(layer, "layers"):
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, tf.keras.layers.Conv2D):
                        last_conv_layer_name = sub_layer.name
                        # Build a reference through the outer model
                        last_conv_layer_name = f"{layer.name}/{sub_layer.name}"
                        break
                if last_conv_layer_name:
                    break

    if last_conv_layer_name is None:
        raise ValueError("Could not auto-detect a convolutional layer for Grad-CAM.")

    # Handle nested model (e.g., MobileNetV2 inside a functional model)
    if "/" in last_conv_layer_name:
        outer_name, inner_name = last_conv_layer_name.split("/", 1)
        outer_layer = model.get_layer(outer_name)
        last_conv_layer = outer_layer.get_layer(inner_name)
        # Build grad model through the nested architecture
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[
                outer_layer(model.input)
                if not hasattr(outer_layer, "get_layer")
                else _get_nested_output(model, outer_layer, inner_name),
                model.output,
            ],
        )
    else:
        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[last_conv_layer.output, model.output],
        )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv outputs by the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU and normalise
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Overlay on original image
    original = np.uint8(img_array[0] * 255)
    overlay = cv2.addWeighted(original, 1 - alpha, heatmap_color, alpha, 0)

    # Encode to base64 PNG
    img_pil = Image.fromarray(overlay)
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _get_nested_output(
    model: tf.keras.Model,
    outer_layer: tf.keras.layers.Layer,
    inner_layer_name: str,
) -> tf.Tensor:
    """Extract intermediate output from a nested model (e.g., MobileNetV2 base)."""
    inner_model = outer_layer
    intermediate_model = tf.keras.Model(
        inputs=inner_model.input,
        outputs=inner_model.get_layer(inner_layer_name).output,
    )
    return intermediate_model(model.input)
