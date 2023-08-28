"""Centralized configuration with typed dataclasses and YAML loading."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


@dataclass(frozen=True)
class DataConfig:
    image_size: int = 96
    num_keypoints: int = 15
    num_outputs: int = 30
    test_split: float = 0.2
    random_seed: int = 42
    training_csv: str = "data/training.csv"
    test_csv: str = "data/test.csv"
    lookup_csv: str = "data/IdLookupTable.csv"


@dataclass(frozen=True)
class CNNAdamConfig:
    learning_rate: float = 0.0005
    epochs: int = 250
    patience: int = 20


@dataclass(frozen=True)
class CNNSGDConfig:
    learning_rate: float = 0.001
    momentum: float = 0.9
    epochs: int = 150
    patience: int = 20
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 10
    min_lr: float = 1e-6


@dataclass(frozen=True)
class CNNConfig:
    batch_size: int = 64
    val_split: float = 0.1
    loss: str = "huber"
    huber_delta: float = 1.0
    adam: CNNAdamConfig = field(default_factory=CNNAdamConfig)
    sgd: CNNSGDConfig = field(default_factory=CNNSGDConfig)


@dataclass(frozen=True)
class ResNetConfig:
    batch_size: int = 16
    learning_rate: float = 0.0001
    epochs: int = 100
    patience: int = 5
    step_lr_size: int = 5
    step_lr_gamma: float = 0.1
    loss: str = "mse_ignore_nan"


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    resnet: ResNetConfig = field(default_factory=ResNetConfig)


def _build_nested(cls, raw: dict):
    """Recursively build a frozen dataclass from a dict, handling nested configs."""
    kwargs = {}
    for f in cls.__dataclass_fields__.values():
        if f.name not in raw:
            continue
        val = raw[f.name]
        if hasattr(f.type, "__dataclass_fields__"):
            val = _build_nested(f.type, val)
        kwargs[f.name] = val
    return cls(**kwargs)


def load_config(path: Optional[str] = None) -> AppConfig:
    """Load configuration from a YAML file.

    Falls back to config/default.yaml if no path is given.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG
    if not config_path.exists():
        logger.warning("Config %s not found, using defaults", config_path)
        return AppConfig()

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    logger.info("Loaded config from %s", config_path)
    return _build_nested(AppConfig, raw)
