"""Unit tests for model architectures and loss functions."""

import pytest
import numpy as np
import torch

from keypoints.models.resnet import build_resnet
from keypoints.utils.losses import MSELossIgnoreNan


class TestResNet:
    def test_output_shape(self):
        model = build_resnet(num_keypoints=15)
        x = torch.randn(2, 1, 96, 96)
        out = model(x)
        assert out.shape == (2, 30)

    def test_single_sample(self):
        model = build_resnet(num_keypoints=15)
        x = torch.randn(1, 1, 96, 96)
        out = model(x)
        assert out.shape == (1, 30)

    def test_gradient_flow(self):
        model = build_resnet(num_keypoints=15)
        x = torch.randn(1, 1, 96, 96, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_parameter_count(self):
        model = build_resnet(num_keypoints=15)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert params > 100_000


class TestNaNAwareLoss:
    def test_no_nans(self):
        loss_fn = MSELossIgnoreNan()
        pred = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[1.0, 2.0, 3.0]])
        loss = loss_fn(pred, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_all_nans(self):
        loss_fn = MSELossIgnoreNan()
        pred = torch.tensor([[1.0, 2.0]])
        target = torch.tensor([[float("nan"), float("nan")]])
        loss = loss_fn(pred, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_partial_nans(self):
        loss_fn = MSELossIgnoreNan()
        pred = torch.tensor([[1.0, 5.0]])
        target = torch.tensor([[1.0, float("nan")]])
        loss = loss_fn(pred, target)
        # Only the first element contributes: (1-1)^2 / 1 = 0
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_partial_nans_with_error(self):
        loss_fn = MSELossIgnoreNan()
        pred = torch.tensor([[3.0, 5.0]])
        target = torch.tensor([[1.0, float("nan")]])
        loss = loss_fn(pred, target)
        # (3-1)^2 / 1 = 4.0
        assert loss.item() == pytest.approx(4.0, abs=1e-6)

    def test_differentiable(self):
        loss_fn = MSELossIgnoreNan()
        pred = torch.tensor([[2.0, 3.0]], requires_grad=True)
        target = torch.tensor([[1.0, float("nan")]])
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None


class TestDataset:
    def test_pytorch_dataset_shapes(self):
        from keypoints.utils.dataset import FacialKeypointsDataset

        images = np.random.rand(5, 96, 96).astype(np.float32)
        keypoints = np.random.rand(5, 30).astype(np.float32)
        dataset = FacialKeypointsDataset(images, keypoints)

        assert len(dataset) == 5
        img, kp = dataset[0]
        assert img.shape == (1, 96, 96)
        assert kp.shape == (30,)

    def test_prepare_keras_data(self):
        from keypoints.utils.dataset import prepare_keras_data

        images = np.random.rand(10, 96, 96).astype(np.float32)
        keypoints = np.random.rand(10, 30).astype(np.float32) * 96

        images_4d, kp_norm = prepare_keras_data(images, keypoints, normalize_targets=True)
        assert images_4d.shape == (10, 96, 96, 1)
        assert kp_norm.max() <= 1.0

    def test_split_data(self):
        from keypoints.utils.dataset import split_data

        images = np.random.rand(100, 96, 96).astype(np.float32)
        keypoints = np.random.rand(100, 30).astype(np.float32)

        X_train, X_val, y_train, y_val = split_data(images, keypoints, test_size=0.2)
        assert len(X_train) == 80
        assert len(X_val) == 20


class TestConfig:
    def test_load_default(self):
        from keypoints.config import load_config

        config = load_config("config/default.yaml")
        assert config.data.image_size == 96
        assert config.data.num_keypoints == 15
        assert config.resnet.lr > 0
