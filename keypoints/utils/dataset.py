"""Dataset loading, preprocessing, and framework-specific data preparation."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

IMAGE_SIZE = 96

KEYPOINT_COLUMNS: list[str] = [
    "left_eye_center_x", "left_eye_center_y",
    "right_eye_center_x", "right_eye_center_y",
    "left_eye_inner_corner_x", "left_eye_inner_corner_y",
    "left_eye_outer_corner_x", "left_eye_outer_corner_y",
    "right_eye_inner_corner_x", "right_eye_inner_corner_y",
    "right_eye_outer_corner_x", "right_eye_outer_corner_y",
    "left_eyebrow_inner_end_x", "left_eyebrow_inner_end_y",
    "left_eyebrow_outer_end_x", "left_eyebrow_outer_end_y",
    "right_eyebrow_inner_end_x", "right_eyebrow_inner_end_y",
    "right_eyebrow_outer_end_x", "right_eyebrow_outer_end_y",
    "nose_tip_x", "nose_tip_y",
    "mouth_left_corner_x", "mouth_left_corner_y",
    "mouth_right_corner_x", "mouth_right_corner_y",
    "mouth_center_top_lip_x", "mouth_center_top_lip_y",
    "mouth_center_bottom_lip_x", "mouth_center_bottom_lip_y",
]


def _parse_images(image_strings: pd.Series) -> np.ndarray:
    """Parse space-separated pixel strings into a float32 image array.

    Returns array of shape (N, 96, 96) normalized to [0, 1].
    """
    images = np.array(
        [np.fromstring(s, sep=" ").astype(np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE)
         for s in image_strings]
    )
    return images / 255.0


def load_training_data(
    csv_path: str,
    fillna: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess training data from the Kaggle CSV.

    Args:
        csv_path: Path to training.csv.
        fillna: If True, forward-fill missing keypoint values (for CNN).
                If False, preserve NaNs (for ResNet with NaN-aware loss).

    Returns:
        images: float32 array (N, 96, 96) normalized to [0, 1].
        keypoints: float32 array (N, 30), may contain NaNs if fillna=False.
    """
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d training samples from %s", len(df), csv_path)

    if fillna:
        df.fillna(method="ffill", inplace=True)

    images = _parse_images(df["Image"])
    keypoints = df.drop(columns=["Image"]).to_numpy().astype(np.float32)

    return images, keypoints


def load_test_data(csv_path: str) -> np.ndarray:
    """Load test images from the Kaggle CSV.

    Returns float32 array (N, 96, 96) normalized to [0, 1].
    """
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d test samples from %s", len(df), csv_path)
    return _parse_images(df["Image"])


def load_lookup_table(csv_path: str) -> pd.DataFrame:
    """Load the Kaggle IdLookupTable for submission mapping."""
    return pd.read_csv(csv_path)


def split_data(
    images: np.ndarray,
    keypoints: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into train and validation sets.

    Returns (X_train, X_val, y_train, y_val).
    """
    return train_test_split(
        images, keypoints, test_size=test_size, random_state=random_state
    )


def prepare_keras_data(
    images: np.ndarray,
    keypoints: Optional[np.ndarray] = None,
    normalize_targets: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Reshape images for Keras CNN and optionally normalize targets.

    Args:
        images: (N, 96, 96) float array.
        keypoints: (N, 30) float array, or None for test data.
        normalize_targets: If True, divide keypoints by IMAGE_SIZE to [0, 1].

    Returns:
        images_4d: (N, 96, 96, 1) for Keras Conv2D input.
        keypoints_norm: Normalized keypoints, or None.
    """
    images_4d = images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    if keypoints is None:
        return images_4d, None
    kp = keypoints / IMAGE_SIZE if normalize_targets else keypoints.copy()
    return images_4d, kp


# ---------------------------------------------------------------------------
# PyTorch Dataset classes (guarded import)
# ---------------------------------------------------------------------------

try:
    import torch
    from torch.utils.data import Dataset

    class FacialKeypointsDataset(Dataset):
        """PyTorch Dataset for labeled facial keypoint images."""

        def __init__(self, images: np.ndarray, keypoints: np.ndarray) -> None:
            self.images = images
            self.keypoints = keypoints

        def __len__(self) -> int:
            return len(self.images)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            img = self.images[idx].reshape(1, IMAGE_SIZE, IMAGE_SIZE)
            kp = self.keypoints[idx]
            return (
                torch.tensor(img, dtype=torch.float32),
                torch.tensor(kp, dtype=torch.float32),
            )

    class FacialKeypointsTestDataset(Dataset):
        """PyTorch Dataset for unlabeled test images."""

        def __init__(self, images: np.ndarray) -> None:
            self.images = images

        def __len__(self) -> int:
            return len(self.images)

        def __getitem__(self, idx: int) -> torch.Tensor:
            img = self.images[idx].reshape(1, IMAGE_SIZE, IMAGE_SIZE)
            return torch.tensor(img, dtype=torch.float32)

except ImportError:
    pass
