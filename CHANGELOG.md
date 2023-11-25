# Changelog

## [1.0.0] - 2026-02-26

### Added
- Modular Python package with `keypoints` namespace
- CNN model (Keras/TensorFlow): 3 conv blocks with LeakyReLU, two-phase training (Adam + SGD)
- ResNet model (PyTorch): 6-stage, 12 residual blocks with batch normalization
- Custom `MSELossIgnoreNan` for training with partially-labeled data
- Shared dataset loading and preprocessing utilities for both frameworks
- Centralized YAML configuration with typed frozen dataclasses
- Training entry points with `--config` and `--debug` CLI flags
- Unified prediction script with Kaggle submission generation
- Visualization utilities: keypoint overlay, training curves, prediction grids, model comparison
- Architecture documentation with pipeline and block diagrams

### Changed
- Merged code from two separate repositories into a single modular package
- Extracted all hardcoded hyperparameters to `config/default.yaml`
- Replaced `print()` statements with Python `logging` module
- Added type hints and docstrings throughout
- PyTorch and TensorFlow are optional dependencies with lazy imports
