#!/usr/bin/env python3
"""Training entry point for the Keras CNN model.

Two-phase training strategy:
  Phase 1: Adam optimizer (lr=0.0005) with Huber loss and early stopping
  Phase 2: SGD fine-tuning (lr=0.001, momentum=0.9) with ReduceLROnPlateau

Usage:
    python train_cnn.py [--config PATH] [--debug]
"""

from __future__ import annotations

import argparse
import logging

from keypoints.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Keras CNN model")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    cfg = load_config(args.config)
    logger.info("Starting CNN training with config: %s", cfg.cnn)

    # Lazy import to avoid loading TensorFlow when not needed
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.losses import Huber
    from tensorflow.keras.optimizers import SGD, Adam

    from keypoints.models.cnn import build_cnn
    from keypoints.utils.dataset import (
        load_training_data, prepare_keras_data, split_data,
    )

    # --- Data Loading ---
    images, keypoints = load_training_data(cfg.data.training_csv, fillna=True)
    X_train, X_val, y_train, y_val = split_data(
        images, keypoints,
        test_size=cfg.cnn.val_split,
        random_state=cfg.data.random_seed,
    )
    X_train, y_train = prepare_keras_data(X_train, y_train, normalize_targets=True)
    X_val, y_val = prepare_keras_data(X_val, y_val, normalize_targets=True)
    logger.info("Train: %d samples, Val: %d samples", len(X_train), len(X_val))

    # --- Build Model ---
    model = build_cnn(num_outputs=cfg.data.num_outputs)

    # --- Phase 1: Adam ---
    logger.info("Phase 1: Adam optimizer (lr=%.4f)", cfg.cnn.adam.learning_rate)
    model.compile(
        optimizer=Adam(learning_rate=cfg.cnn.adam.learning_rate),
        loss=Huber(delta=cfg.cnn.huber_delta),
        metrics=["mae"],
    )
    history_adam = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.cnn.adam.epochs,
        batch_size=cfg.cnn.batch_size,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=cfg.cnn.adam.patience,
                restore_best_weights=True,
            ),
        ],
        verbose=1,
    )
    model.save("adam_best.h5")
    logger.info("Phase 1 complete. Best val_loss: %.6f",
                min(history_adam.history["val_loss"]))

    # --- Phase 2: SGD Fine-tuning ---
    logger.info("Phase 2: SGD optimizer (lr=%.4f, momentum=%.1f)",
                cfg.cnn.sgd.learning_rate, cfg.cnn.sgd.momentum)
    model.compile(
        optimizer=SGD(
            learning_rate=cfg.cnn.sgd.learning_rate,
            momentum=cfg.cnn.sgd.momentum,
        ),
        loss=Huber(delta=cfg.cnn.huber_delta),
        metrics=["mae"],
    )
    history_sgd = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.cnn.sgd.epochs,
        batch_size=cfg.cnn.batch_size,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=cfg.cnn.sgd.patience,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=cfg.cnn.sgd.reduce_lr_factor,
                patience=cfg.cnn.sgd.reduce_lr_patience,
                verbose=1,
                min_lr=cfg.cnn.sgd.min_lr,
            ),
        ],
        verbose=1,
    )
    model.save("sgd_best.h5")
    logger.info("Phase 2 complete. Best val_loss: %.6f",
                min(history_sgd.history["val_loss"]))
    logger.info("Training finished. Models saved: adam_best.h5, sgd_best.h5")


if __name__ == "__main__":
    main()
