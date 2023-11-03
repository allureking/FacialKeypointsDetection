#!/usr/bin/env python3
"""Training entry point for the PyTorch ResNet model.

Single-phase training with Adam, StepLR scheduling, early stopping,
and NaN-aware MSE loss for partially-labeled data.

Usage:
    python train_resnet.py [--config PATH] [--debug]
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from keypoints.config import load_config
from keypoints.models.resnet import ResNet
from keypoints.utils.dataset import (
    FacialKeypointsDataset, load_training_data, split_data,
)
from keypoints.utils.losses import MSELossIgnoreNan


def train_one_epoch(
    model: ResNet,
    loader: DataLoader,
    criterion: MSELossIgnoreNan,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, keypoints in loader:
        images, keypoints = images.to(device), keypoints.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, keypoints)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
    return total_loss / total_samples


def validate(
    model: ResNet,
    loader: DataLoader,
    criterion: MSELossIgnoreNan,
    device: torch.device,
) -> float:
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, keypoints in loader:
            images, keypoints = images.to(device), keypoints.to(device)
            outputs = model(images)
            loss = criterion(outputs, keypoints)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
    return total_loss / total_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PyTorch ResNet model")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Starting ResNet training with config: %s", cfg.resnet)

    # --- Data Loading ---
    images, keypoints = load_training_data(cfg.data.training_csv, fillna=False)
    X_train, X_val, y_train, y_val = split_data(
        images, keypoints,
        test_size=cfg.data.test_split,
        random_state=cfg.data.random_seed,
    )
    train_loader = DataLoader(
        FacialKeypointsDataset(X_train, y_train),
        batch_size=cfg.resnet.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        FacialKeypointsDataset(X_val, y_val),
        batch_size=cfg.resnet.batch_size, shuffle=False,
    )
    logger.info("Train: %d samples, Val: %d samples", len(X_train), len(X_val))

    # --- Model, Loss, Optimizer ---
    model = ResNet(num_outputs=cfg.data.num_outputs).to(device)
    criterion = MSELossIgnoreNan()
    optimizer = optim.Adam(model.parameters(), lr=cfg.resnet.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.resnet.step_lr_size, gamma=cfg.resnet.step_lr_gamma,
    )

    # --- Training Loop ---
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, cfg.resnet.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        logger.info(
            "Epoch %d/%d — Train: %.4f | Val: %.4f",
            epoch, cfg.resnet.epochs, train_loss, val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= cfg.resnet.patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    logger.info("Training finished. Best val_loss: %.4f", best_val_loss)

    # Save loss history for visualization
    np.savez(
        "resnet_history.npz",
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
    )
    logger.info("Loss history saved to resnet_history.npz")


if __name__ == "__main__":
    main()
