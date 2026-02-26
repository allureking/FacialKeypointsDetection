# Changelog

## [1.1.0] - 2026-02-26

### Added
- pytest test suite covering model output shapes, gradient flow, NaN-aware loss edge cases, dataset utilities, and config loading
- Sample prediction images (`assets/sample_predictions_resnet.png`)
- Training curve visualizations (`assets/training_curves.png`)
- Model comparison overlay (`assets/model_comparison.png`)

### Changed
- Rewrote README with badges (Python, PyTorch, TensorFlow, License, Kaggle RMSE), embedded images, architecture diagrams, and code examples
- Added parameter counts and strategy details to results table
- Added references section (He et al. 2016, Kaggle)

### Fixed
- Deprecated `df.fillna(method="ffill")` replaced with `df.ffill()` for pandas 2.x compatibility

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
