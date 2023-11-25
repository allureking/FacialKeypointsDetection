# Architecture

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│              Facial Keypoints Detection Pipeline              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: 96×96 Grayscale Image (from CSV)                    │
│  ├─ Pixel normalization (0-255 → 0-1)                       │
│  ├─ NaN handling: forward-fill (CNN) or mask (ResNet)        │
│  └─ Train/Val split (80/20 ResNet, 90/10 CNN)               │
│                                                              │
│  ┌──────────────────┐    ┌──────────────────────────┐        │
│  │  CNN (Keras)     │    │  ResNet (PyTorch)        │        │
│  │                  │    │                          │        │
│  │  Conv2D(32)      │    │  Stem: Conv(1→32)+BN+Pool│        │
│  │  Conv2D(64)      │    │  Stage 1: ResBlock×2 (32)│        │
│  │  Conv2D(128)     │    │  Stage 2: ResBlock×2 (64)│        │
│  │  Dense(500)      │    │  Stage 3: ResBlock×2(128)│        │
│  │  Dense(500)      │    │  Stage 4: ResBlock×2(256)│        │
│  │  Dense(30)       │    │  Stage 5: ResBlock×2(512)│        │
│  │                  │    │  AvgPool → Linear(512→30)│        │
│  │  LeakyReLU       │    │  BatchNorm + ReLU        │        │
│  │  Dropout         │    │  1×1 conv shortcuts      │        │
│  │  Huber loss      │    │  NaN-aware MSE loss      │        │
│  │  Adam → SGD      │    │  Adam + StepLR           │        │
│  └────────┬─────────┘    └─────────────┬────────────┘        │
│           │                            │                     │
│           └────────────┬───────────────┘                     │
│                        │                                     │
│  Output: 30 values (x,y for 15 keypoints)                   │
│  └─ Kaggle submission CSV via IdLookupTable                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Residual Block

```
Input ─────────────────────────────────────┐
  │                                        │
  ├─ Conv2d(3×3) → BatchNorm → ReLU       │ (shortcut)
  │                                        │
  ├─ Conv2d(3×3) → BatchNorm              Conv1×1 (if dims differ)
  │                                        │
  └───────────── + ────────────────────────┘
                 │
               ReLU
                 │
              Output
```

## NaN-Aware Loss

The Kaggle training set has 7,049 samples, but only ~2,140 have all 15 keypoints labeled. Naive approaches either:
- Drop rows with any NaN (lose ~70% of data)
- Fill NaN with mean (introduce bias)

Our `MSELossIgnoreNan` masks NaN targets so each sample contributes loss only for its available keypoints:

```
loss = Σ(pred[mask] - target[mask])² / count(mask)
```

## Two-Phase CNN Training

```
Phase 1: Adam (lr=0.0005)          Phase 2: SGD (lr=0.001, momentum=0.9)
┌──────────────────────┐           ┌──────────────────────────────────┐
│ Fast convergence     │           │ Stable fine-tuning               │
│ Adaptive per-param   │──────────>│ Uniform gradient steps           │
│ Huber loss δ=1.0     │           │ ReduceLROnPlateau (factor=0.5)   │
│ Early stop p=20      │           │ Early stop p=20                  │
└──────────────────────┘           └──────────────────────────────────┘
```

Adam's adaptive learning rates enable rapid initial convergence, but can oscillate near minima. SGD with momentum provides more stable fine-tuning for the final performance push.
