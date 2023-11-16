"""Visualization utilities for keypoints, training curves, and model comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def show_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """Display a single face image with keypoints overlaid as red dots.

    Args:
        image: (96, 96) grayscale image (values in [0, 1] or [0, 255]).
        keypoints: (30,) array of x, y coordinates (even=x, odd=y).
        title: Optional plot title.
        save_path: If given, save figure to this path instead of showing.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap="gray")
    ax.scatter(keypoints[0::2], keypoints[1::2], s=20, marker=".", c="red")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    title: str = "Training Curves",
    save_path: Optional[str] = None,
) -> None:
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Val Loss", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def save_sample_predictions(
    images: np.ndarray,
    predictions: np.ndarray,
    output_dir: str,
    num_samples: int = 8,
    prefix: str = "pred",
) -> list[str]:
    """Save individual prediction images for portfolio display.

    Args:
        images: (N, 96, 96) array.
        predictions: (N, 30) array of keypoint coordinates.
        output_dir: Directory to save images.
        num_samples: Number of samples to save.
        prefix: Filename prefix.

    Returns:
        List of saved file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(min(num_samples, len(images))):
        path = str(out / f"{prefix}_{i + 1:02d}.png")
        show_keypoints(images[i], predictions[i], save_path=path)
        paths.append(path)
    return paths


def save_prediction_grid(
    images: np.ndarray,
    predictions: np.ndarray,
    save_path: str,
    rows: int = 2,
    cols: int = 4,
) -> None:
    """Save a grid of face images with keypoints.

    Args:
        images: (N, 96, 96) array.
        predictions: (N, 30) keypoint coordinates.
        save_path: Output image path.
        rows: Number of grid rows.
        cols: Number of grid columns.
    """
    n = min(rows * cols, len(images))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    for idx, ax in enumerate(axes.flat):
        if idx < n:
            ax.imshow(images[idx], cmap="gray")
            kp = predictions[idx]
            ax.scatter(kp[0::2], kp[1::2], s=12, marker=".", c="red")
        ax.set_axis_off()
    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compare_models(
    image: np.ndarray,
    cnn_keypoints: np.ndarray,
    resnet_keypoints: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Side-by-side comparison of CNN vs ResNet predictions on one face."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    for ax, kp, label in [
        (ax1, cnn_keypoints, "CNN (Keras)"),
        (ax2, resnet_keypoints, "ResNet (PyTorch)"),
    ]:
        ax.imshow(image, cmap="gray")
        ax.scatter(kp[0::2], kp[1::2], s=20, marker=".", c="red")
        ax.set_title(label, fontsize=11)
        ax.set_axis_off()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
