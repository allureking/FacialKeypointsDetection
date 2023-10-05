"""Keras/TensorFlow CNN model for facial keypoint detection.

Architecture:
    Conv2D(32, 3x3) -> LeakyReLU(0.01) -> MaxPool(2x2) -> Dropout(0.05)
    Conv2D(64, 3x3) -> LeakyReLU(0.01) -> MaxPool(2x2) -> Dropout(0.01)
    Conv2D(128, 3x3) -> LeakyReLU(0.01) -> MaxPool(2x2) -> Dropout(0.15)
    Flatten -> Dense(500) -> LeakyReLU -> Dropout(0.5)
    Dense(500) -> LeakyReLU -> Dropout(0.5) -> Dense(30)

Training uses a two-phase strategy:
  Phase 1: Adam optimizer for fast initial convergence
  Phase 2: SGD with momentum for stable fine-tuning near the optimum
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorflow.keras.models import Sequential

logger = logging.getLogger(__name__)


def build_cnn(num_outputs: int = 30) -> "Sequential":
    """Build the CNN model architecture.

    Args:
        num_outputs: Number of output values (default 30 for 15 keypoints).

    Returns:
        Compiled-ready Keras Sequential model (not yet compiled).
    """
    from tensorflow.keras.layers import (
        Conv2D, Dense, Dropout, Flatten, LeakyReLU, MaxPooling2D,
    )
    from tensorflow.keras.models import Sequential

    model = Sequential([
        # Block 1: 32 filters
        Conv2D(32, (3, 3), input_shape=(96, 96, 1)),
        LeakyReLU(alpha=0.01),
        MaxPooling2D(2, 2),
        Dropout(0.05),

        # Block 2: 64 filters
        Conv2D(64, (3, 3)),
        LeakyReLU(alpha=0.01),
        MaxPooling2D(2, 2),
        Dropout(0.01),

        # Block 3: 128 filters
        Conv2D(128, (3, 3)),
        LeakyReLU(alpha=0.01),
        MaxPooling2D(2, 2),
        Dropout(0.15),

        # Fully connected
        Flatten(),
        Dense(500),
        LeakyReLU(alpha=0.01),
        Dropout(0.5),
        Dense(500),
        LeakyReLU(alpha=0.01),
        Dropout(0.5),

        # Output: 30 keypoint coordinates
        Dense(num_outputs),
    ])

    logger.info("Built CNN model with %d parameters", model.count_params())
    return model
