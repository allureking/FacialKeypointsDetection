#!/usr/bin/env python3
"""Inference and Kaggle submission generation for both models.

Usage:
    python predict.py --model resnet --weights best_model.pth [--config PATH]
    python predict.py --model cnn --weights sgd_best.h5 [--config PATH]
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from keypoints.config import load_config
from keypoints.utils.dataset import (
    IMAGE_SIZE, KEYPOINT_COLUMNS, load_lookup_table, load_test_data,
    prepare_keras_data,
)


def predict_resnet(weights_path: str, test_images: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Load a ResNet model and generate predictions.

    Returns predictions array of shape (N, 30) in pixel coordinates.
    """
    import torch
    from keypoints.models.resnet import ResNet

    model = ResNet()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    from keypoints.utils.dataset import FacialKeypointsTestDataset
    from torch.utils.data import DataLoader

    loader = DataLoader(
        FacialKeypointsTestDataset(test_images), batch_size=16, shuffle=False,
    )
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch.to(device))
            all_preds.append(outputs.cpu().numpy())

    predictions = np.vstack(all_preds)
    return np.clip(predictions, 0, IMAGE_SIZE)


def predict_cnn(weights_path: str, test_images: np.ndarray) -> np.ndarray:
    """Load a Keras CNN model and generate predictions.

    Returns predictions array of shape (N, 30) in pixel coordinates.
    """
    from tensorflow.keras.models import load_model

    model = load_model(weights_path)
    X_test, _ = prepare_keras_data(test_images)
    predictions = model.predict(X_test)
    # CNN was trained with targets normalized to [0, 1], scale back
    return np.clip(predictions * IMAGE_SIZE, 0, IMAGE_SIZE)


def generate_submission(
    predictions: np.ndarray,
    lookup_path: str,
    output_path: str = "submission.csv",
) -> None:
    """Create a Kaggle-format submission CSV from predictions."""
    lookup = load_lookup_table(lookup_path)
    preds_df = pd.DataFrame(predictions, columns=KEYPOINT_COLUMNS)

    locations = []
    for _, row in lookup.iterrows():
        image_id = row["ImageId"] - 1  # zero-indexed
        locations.append(preds_df.loc[image_id, row["FeatureName"]])

    submission = lookup[["RowId"]].copy()
    submission["Location"] = locations
    submission.to_csv(output_path, index=False)
    logging.getLogger(__name__).info("Submission saved to %s (%d rows)", output_path, len(submission))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions and Kaggle submission")
    parser.add_argument("--model", required=True, choices=["resnet", "cnn"])
    parser.add_argument("--weights", required=True, help="Path to model weights")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--output", default="submission.csv", help="Output CSV path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config(args.config)
    test_images = load_test_data(cfg.data.test_csv)

    if args.model == "resnet":
        predictions = predict_resnet(args.weights, test_images)
    else:
        predictions = predict_cnn(args.weights, test_images)

    generate_submission(predictions, cfg.data.lookup_csv, args.output)


if __name__ == "__main__":
    main()
