"""Custom loss functions for training with partially-labeled data."""

from __future__ import annotations

import torch
import torch.nn as nn


class MSELossIgnoreNan(nn.Module):
    """MSE loss that ignores NaN values in the target.

    When training data has missing keypoint annotations, this loss computes
    MSE only over the finite (non-NaN) target values. This allows training
    on all 7,049 samples rather than discarding the ~5,000 that have
    partial labels.

    The original approach replaced NaN targets with the prediction value
    (making their gradient zero). This implementation uses explicit masking
    for clarity.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(target)
        count = mask.sum()
        if count == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return ((pred[mask] - target[mask]) ** 2).sum() / count
