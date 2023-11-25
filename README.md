# Facial Keypoints Detection

CNN and ResNet models for detecting 15 facial keypoints (30 x,y coordinates) on 96×96 grayscale images. Built for the [Kaggle Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection) competition.

## Results

| Model | Framework | Kaggle Score (RMSE) | Training Strategy |
|-------|-----------|-------------------|-------------------|
| ResNet | PyTorch | **2.10** | Adam + StepLR + NaN-aware MSE |
| CNN | TensorFlow/Keras | 2.55 | Two-phase: Adam → SGD + Huber |

## Features

- **Dual-framework architecture** — PyTorch ResNet and Keras CNN sharing data loading and configuration
- **NaN-aware loss** — Custom `MSELossIgnoreNan` trains on all 7,049 samples including partially-labeled ones
- **Two-phase CNN training** — Adam for fast convergence, then SGD with ReduceLROnPlateau for fine-tuning
- **Deep residual learning** — 6-stage ResNet (32→512 channels) with batch normalization and skip connections
- **Centralized YAML config** — All hyperparameters in `config/default.yaml` with typed dataclass validation

## Project Structure

```
├── config/default.yaml         # Hyperparameters for both models
├── keypoints/
│   ├── config.py               # Frozen dataclasses + YAML loader
│   ├── models/
│   │   ├── cnn.py              # Keras CNN (3 conv blocks + 2 FC layers)
│   │   └── resnet.py           # PyTorch ResNet (12 residual blocks)
│   └── utils/
│       ├── dataset.py          # Data loading, preprocessing, PyTorch Dataset
│       ├── losses.py           # MSELossIgnoreNan
│       └── visualization.py    # Keypoint plotting, training curves
├── train_cnn.py                # CNN training entry point
├── train_resnet.py             # ResNet training entry point
└── predict.py                  # Inference + Kaggle submission
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/allureking/FacialKeypointsDetection.git
cd FacialKeypointsDetection

# Install with both frameworks
pip install -e ".[all]"

# Or install for one framework only
pip install -e ".[pytorch]"    # PyTorch ResNet only
pip install -e ".[tensorflow]" # Keras CNN only
```

### Download Data

Download the [Kaggle dataset](https://www.kaggle.com/c/facial-keypoints-detection/data) and place files in `data/`:
```
data/
├── training.csv
├── test.csv
└── IdLookupTable.csv
```

### Train

```bash
# Train ResNet (PyTorch)
python train_resnet.py --config config/default.yaml

# Train CNN (Keras/TensorFlow)
python train_cnn.py --config config/default.yaml
```

### Predict

```bash
# Generate Kaggle submission
python predict.py --model resnet --weights best_model.pth
python predict.py --model cnn --weights sgd_best.h5
```

## How It Works

### Data Preprocessing

Training images are stored as space-separated pixel strings in CSV format. Each is parsed into a 96×96 float32 array and normalized to [0, 1]. Of the 7,049 training samples, only ~2,140 have all 15 keypoints labeled — the rest have partial annotations.

### CNN (Keras)

Three convolutional blocks (32→64→128 filters) with LeakyReLU and progressive dropout, followed by two 500-unit dense layers. Training uses Huber loss in two phases:

1. **Adam** (lr=0.0005, patience=20) for fast initial convergence
2. **SGD** (lr=0.001, momentum=0.9) with `ReduceLROnPlateau` for fine-tuning

Keypoint targets are normalized to [0, 1] by dividing by 96.

### ResNet (PyTorch)

Custom 6-stage ResNet with 12 residual blocks progressing from 32 to 512 channels. Each block uses batch normalization and ReLU with 1×1 convolution shortcuts for dimension matching. Trained with Adam (lr=0.0001), StepLR scheduling (step=5, gamma=0.1), and early stopping.

The custom `MSELossIgnoreNan` computes loss only over finite target values, allowing all 7,049 samples to contribute proportional gradients rather than discarding partially-labeled data.

## Configuration

All hyperparameters are centralized in `config/default.yaml` and loaded into frozen dataclasses at startup:

```yaml
resnet:
  batch_size: 16
  learning_rate: 0.0001
  epochs: 100
  patience: 5
  step_lr_size: 5
  step_lr_gamma: 0.1
```

## Acknowledgments

CNN implementation developed in collaboration with Felix-hyy for the USF 276DS course final project. ResNet architecture inspired by the Advanced Machine Learning course at USF.

## License

MIT
