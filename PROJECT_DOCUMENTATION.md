# Brain Tumor Detection Project - Code Documentation

**Last Updated:** May 12, 2026  
**Language:** Python 3.8+ | TensorFlow 2.10+  
**Project Type:** Medical Image Classification (Deep Learning)

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Datasets & Classes](#datasets--classes)
4. [Core Components](#core-components)
5. [Models](#models)
6. [Usage & Execution](#usage--execution)
7. [Key Implementation Details](#key-implementation-details)
8. [Configuration Reference](#configuration-reference)

---

## 📊 Project Overview

### Purpose

Multi-class brain tumor classification system using deep learning on MRI images. Classifies images into 4 categories:

- **notumor** (label 0) - Healthy brain scan
- **meningioma** (label 1) - Benign meninges tumor
- **pituitary** (label 2) - Pituitary gland tumor
- **glioma** (label 3) - Malignant glial tumor

### Technologies

- **Framework:** TensorFlow/Keras
- **Vision:** OpenCV
- **ML Utils:** Scikit-learn, NumPy
- **Visualization:** Matplotlib
- **Architecture:** CNN (baseline) + Transfer Learning (EfficientNetV2B0)

### Key Requirement

- **Confidence Threshold:** 70% (cases below → manual review)
- **Target Metrics:** F2-Score (recall emphasis), Sensitivity > 95% per class, ROC-AUC

---

## 📁 Project Structure

```
brain_tumor_project/
├── main.py                              # Entry point - orchestrates training
├── evaluation.py                        # Final evaluation metrics & visualizations
│
├── src/
│   ├── model.py                        # Baseline CNN (Encoder-Decoder architecture)
│   ├── preprocessing.py                # Data loading & caching (128×128)
│   ├── TransferLearned.py              # Transfer learning model & fine-tuning
│   └── preprocessing_tl.py             # Transfer learning preprocessing (224×224)
│
├── data/
│   ├── Training/                       # ~3,500 images (split 80/20)
│   │   ├── notumor/
│   │   ├── meningioma/
│   │   ├── pituitary/
│   │   └── glioma/
│   ├── Testing/                        # ~1,000 images (final evaluation only)
│   │   ├── notumor/
│   │   ├── meningioma/
│   │   ├── pituitary/
│   │   └── glioma/
│   ├── cache/                          # Cached 128×128 preprocessed .npy files
│   └── cache_tl/                       # Cached 224×224 preprocessed .npy files
│
├── models/
│   ├── cnn_baseline.keras              # Saved baseline CNN model
│   ├── transfer_efficientnetb0.keras   # Transfer learning model
│   └── transfer_efficientnetv2b0.keras # EfficientNetV2B0 model
│
└── results/                            # Training results & visualizations
    ├── cnn_baseline_TIMESTAMP/
    └── transfer_efficientnetv2b0_TIMESTAMP/
```

---

## 🗂️ Datasets & Classes

### Class Mapping

```python
CLASS_MAPPING = {
    'notumor':    0,
    'meningioma': 1,
    'pituitary':  2,
    'glioma':     3
}
```

### Data Split Strategy

**From Training/ folder:**

- 80% (≈3,520 images) → Training set (model learns)
- 20% (≈880 images) → Validation set (EarlyStopping monitoring)

**Testing/ folder:**

- ≈1,000 images (pristine, untouched during training)
- Used ONLY for final evaluation (test_accuracy, test_loss)

**Prevents data leakage:** Stratified split preserves class distribution

### Preprocessing

**Baseline CNN (128×128):**

- Load image → BGR to RGB → Resize to 128×128
- Normalize: divide by 255 → [0, 1] range
- Optional augmentation: Rotation (8°), Zoom (10%), Flip (horizontal), Contrast (10%)
- Cache: `data/cache/*.npy`

**Transfer Learning (224×224):**

- Load image → BGR to RGB → Resize to 224×224
- **No normalization** (applied by `preprocess_input()`)
- Augmentation: Rotation (10°), Zoom (15%), Flip (horizontal), Contrast (15%)
- Cache: `data/cache_tl/*.npy`

---

## 🏗️ Core Components

### 1. Main Entry Point (`main.py`)

**Functions:**

- `lancer_entrainement()` - Train baseline CNN with class weighting
- `lancer_evaluation_seule()` - Run evaluation without retraining
- `lancer_transfer_learning()` - Train transfer learning model with fine-tuning
- `main()` - Menu-driven interface

**Class Weighting Strategy:**

```python
class_weight = {
    0: 1.0,      # notumor (balanced)
    1: 2.5,      # meningioma (harder class - boosted)
    2: 1.0,      # pituitary (balanced)
    3: 1.8       # glioma (harder class - boosted)
}
```

### 2. Baseline Preprocessing (`src/preprocessing.py`)

**Key Functions:**

- `charger_image(chemin_image)` - Load single image, resize, normalize
- `charger_dataset(chemin_dossier)` - Load all images from folder
- `obtenir_donnees(chemin_train, chemin_test, forcer_recalcul=True)` - Main data loader with caching

**Features:**

- ✅ Smart caching: first run preprocesses (2 min), subsequent runs load cache (2 sec)
- ✅ Stratified 80/20 split preserves class distribution
- ✅ One-hot encoding for labels
- ✅ Shuffle with `random_state=42` for reproducibility

**Returns:**

```python
X_train, y_train, y_train_raw, X_val, y_val, X_test, y_test, y_test_raw
```

### 3. Transfer Learning Preprocessing (`src/preprocessing_tl.py`)

**Differences from baseline:**

- Image size: 224×224 (EfficientNetV2B0 requirement)
- NO normalization (handled by `preprocess_input()`)
- Cache location: `data/cache_tl/`

### 4. Evaluation Module (`evaluation.py`)

**Key Functions:**

- `preparer_predictions(model, X_test, y_test, y_test_raw)` - Get predictions & confidence
- `afficher_score_global(model, X_test, y_test)` - Final accuracy/loss
- `afficher_matrice_confusion(y_test_raw, y_pred, dossier_resultats)` - Confusion matrix
- `afficher_sensibilite(y_test_raw, y_pred)` - Per-class recall
- `afficher_roc_auc(y_test, y_proba)` - ROC curves per class
- `executer_evaluation_complete(X_test, y_test, y_test_raw, chemin_modele)` - Full evaluation pipeline

**Outputs:**

- Confusion matrix heatmap (normalized %)
- ROC-AUC curves (per class)
- Confidence distribution histogram
- Per-class metrics (precision, recall, F1, F2)
- Manual review flag (confidence < 70%)

**Configuration:**

```python
SEUIL_CONFIANCE = 0.7  # 70% threshold for manual review
NOMS_CLASSES = ['notumor', 'meningioma', 'pituitary', 'glioma']
```

---

## 🧠 Models

### Model 1: Baseline CNN

**File:** `src/model.py`

**Architecture Type:** Encoder-Decoder-Dense

```
Input: 128×128×3 (RGB image)
    ↓
[Optional] Data Augmentation
    ↓
ENCODER (progressive compression)
├── Conv2D(32, 3×3) + BatchNorm + MaxPool(2×2) → 64×64
├── Conv2D(64, 3×3) + BatchNorm + MaxPool(2×2) → 32×32
└── Conv2D(128, 3×3) + BatchNorm + MaxPool(2×2) → 16×16 [Bottleneck]
    ↓
DECODER (progressive expansion)
├── UpSampling(2×2) + Conv2D(64) + BatchNorm → 32×32
└── UpSampling(2×2) + Conv2D(32) + BatchNorm → 64×64
    ↓
CLASSIFICATION HEAD
├── GlobalAveragePooling2D() → 32-dim vector
├── Dropout(0.5)
├── Dense(128) + ReLU + L2(1e-4)
├── Dropout(0.4)
└── Dense(4) + Softmax → [P(notumor), P(meningioma), P(pituitary), P(glioma)]
    ↓
Output: 4 class probabilities
```

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Input Size | 128×128×3 |
| Learning Rate | 0.0005 |
| Optimizer | Adam |
| Loss | categorical_crossentropy |
| Max Epochs | 30 |
| Batch Size | 32 |
| Dropout | 0.5, 0.4 |
| L2 Regularization | 1e-4 |
| Augmentation | Enabled (Rotation 8°, Zoom 10%, Flip, Contrast 10%) |
| Early Stopping | patience=4 |
| LR Scheduler | ReduceLROnPlateau (patience=2, factor=0.5) |

**Key Features:**

- ✅ Encoder captures hierarchical features
- ✅ Decoder enriches representation before classification
- ✅ BatchNormalization stabilizes training
- ✅ Class weighting addresses harder classes
- ✅ Callbacks prevent overfitting

### Model 2: Transfer Learning (EfficientNetV2B0)

**File:** `src/TransferLearned.py`

**Architecture:**

```
Input: 224×224×3
    ↓
[Optional] Data Augmentation
    ↓
EfficientNetV2B0 Backbone (ImageNet pretrained, FROZEN)
├── Stem + MBConv blocks
├── Output: 7×7×1280 feature maps
└── Trainable: False (Phase 1)
    ↓
CUSTOM CLASSIFICATION HEAD (Trainable)
├── GlobalAveragePooling2D() → 1280-dim
├── Dropout(0.5)
├── Dense(512) + ReLU + Dropout(0.3)
├── Dense(256) + ReLU + Dropout(0.2)
└── Dense(4) + Softmax
    ↓
Output: 4 class probabilities
```

**Hyperparameters (Phase 1 - Backbone Frozen):**
| Parameter | Value |
|-----------|-------|
| Input Size | 224×224×3 |
| Learning Rate | 1e-3 |
| Optimizer | Adam |
| Max Epochs | 25 |
| Batch Size | 32 |
| Augmentation | Enabled (Rotation 10°, Zoom 15%, Flip, Contrast 15%) |
| Early Stopping | patience=5 |
| LR Scheduler | ReduceLROnPlateau (patience=2, factor=0.5) |

**Phase 2 - Fine-Tuning (optional):**

- Unfreeze top 30 layers of backbone
- Epochs: 8
- Learning Rate: 1e-5 (very conservative)
- Same loss & metrics

**Key Features:**

- ✅ Leverages 1.2M ImageNet-pretrained parameters
- ✅ Two-phase training: backbone frozen → fine-tune
- ✅ Custom head adapts features to medical domain
- ✅ Larger input (224×224) captures more detail

---

## 🎮 Usage & Execution

### Prerequisites

```bash
# Python version
python --version  # 3.8+

# Install dependencies
pip install tensorflow>=2.10 opencv-python scikit-learn numpy matplotlib
```

### Running Baseline Training

```bash
cd /Users/mac/Desktop/brain_tumor_project

# Train baseline CNN
python main.py  # Runs lancer_entrainement()

# Expected output:
# - Training logs (Epoch X/30...)
# - Validation metrics
# - Model saved: models/cnn_baseline.keras
# - Results: results/cnn_baseline_TIMESTAMP/
# - Visualizations: confusion matrix, ROC curves, confidence distribution
```

### Running Evaluation Only

```bash
# Evaluate without retraining
python -c "from main import lancer_evaluation_seule; lancer_evaluation_seule()"

# Or directly run evaluation module
python evaluation.py
```

### Running Transfer Learning

```bash
# Train transfer learning model with fine-tuning
python -c "from main import lancer_transfer_learning; lancer_transfer_learning()"

# Expected steps:
# 1. Phase 1: Train classification head (10-15 epochs)
# 2. Phase 2: Fine-tune top 30 backbone layers (8 epochs)
# 3. Save model: models/transfer_efficientnetb0.keras
# 4. Evaluate on test set
```

### Python API Usage

```python
# Load and use trained model
from tensorflow.keras.models import load_model
import numpy as np
from src.preprocessing import obtenir_donnees

# Load data
X_test, y_test, y_test_raw = obtenir_donnees(
    'data/Training', 'data/Testing', forcer_recalcul=False
)[4:7]

# Load model
model = load_model('models/cnn_baseline.keras')

# Predict
predictions = model.predict(X_test[:10])
# Shape: (10, 4) - probabilities for each class

# Get class prediction
class_pred = np.argmax(predictions, axis=1)

# Get confidence
confidence = np.max(predictions, axis=1)

# Flag for manual review
manual_review = confidence < 0.7
```

---

## 🔧 Key Implementation Details

### Smart Caching System

**Problem:** Loading 4,400 images repeatedly → slow development cycle

**Solution:** Preprocessing once, save as .npy files

**Flow:**

```
First run (cold cache):
  1. Load images from disk
  2. Resize, normalize
  3. Save to .npy → /data/cache/
  4. Time: ~2 minutes

Subsequent runs (warm cache):
  1. Check if .npy files exist
  2. Load directly from cache
  3. Time: ~2 seconds (100× faster!)

Force recalculation:
  obtenir_donnees(..., forcer_recalcul=True)
```

### Data Pipeline

```python
# Main data loading function
X_train, y_train, y_train_raw, \
X_val, y_val, \
X_test, y_test, y_test_raw = obtenir_donnees(
    'data/Training',
    'data/Testing',
    forcer_recalcul=False
)

# Returns:
# X_train: (3520, 128, 128, 3) - training images
# y_train: (3520, 4) - one-hot labels
# y_train_raw: (3520,) - raw class indices 0-3
# X_val: (880, 128, 128, 3) - validation images
# X_test: (1000, 128, 128, 3) - pristine test set
```

### Training Flow

**Baseline CNN:**

```
1. Load data (with cache)
2. Compute class weights (balanced + boost hard classes)
3. Build model architecture
4. Compile (optimizer + loss)
5. Train with callbacks (EarlyStopping, ReduceLROnPlateau)
6. Save best checkpoint
7. Evaluate on test set
```

**Transfer Learning:**

```
1. Load data (224×224 resolution)
2. Build model (EfficientNetV2B0 + custom head)
3. Train Phase 1 (backbone frozen, train head only)
4. Fine-tune Phase 2 (unfreeze top layers, lower LR)
5. Save model
6. Evaluate on test set
```

### Class Weighting

Medical domain: Some classes harder to detect

```python
# Balanced weights first
weights = compute_class_weight('balanced', classes=classes, y=y_train_raw)

# Then manually boost hard classes
class_weight[1] = 2.5   # meningioma (harder)
class_weight[3] = 1.8   # glioma (harder)

# Effect: Loss penalizes meningioma/glioma errors more heavily
# model.fit(..., class_weight=class_weight)
```

### Callbacks

**EarlyStopping:**

- Monitor: validation loss
- Patience: 4 epochs
- Action: Stop if no improvement for 4 epochs
- Restore: Best checkpoint

**ReduceLROnPlateau:**

- Monitor: validation loss
- Patience: 2 epochs
- Factor: 0.5 (halve learning rate)
- Minimum LR: 1e-6

---

## ⚙️ Configuration Reference

### src/model.py (Baseline)

```python
IMG_SIZE = 128
NUM_CLASSES = 4
MAX_EPOCHS = 30
BATCH_SIZE = 32
USE_AUGMENTATION = True  # Can disable for no augmentation
MODEL_VERSION = "EXP5_encodeur_decodeur_dense"

# Learning rate
Adam(learning_rate=0.0005)

# L2 regularization
kernel_regularizer=l2(1e-4)

# Dropout
Dropout(0.5)  # In bottleneck
Dropout(0.4)  # Before output
```

### src/TransferLearned.py

```python
IMG_SIZE = 224  # EfficientNetV2B0 standard
NUM_CLASSES = 4
MAX_EPOCHS = 25  # Phase 1 (backbone frozen)
BATCH_SIZE = 32
USE_AUGMENTATION = True

# Learning rates
Adam(learning_rate=1e-3)  # Phase 1
# fine_lr=1e-5 for Phase 2

# Fine-tuning parameters
num_unfreeze = 30       # Unfreeze top 30 layers
fine_tune_epochs = 8    # Phase 2 epochs
```

### src/preprocessing.py

```python
IMG_SIZE = 128
NUM_CLASSES = 4
CACHE_DIR = 'data/cache'

CLASS_MAPPING = {
    'notumor': 0, 'meningioma': 1,
    'pituitary': 2, 'glioma': 3
}

# Train/Val/Test split
train_size = 0.80  # 80% training
val_size = 0.20    # 20% validation
random_state = 42  # Reproducibility
```

### evaluation.py

```python
NOMS_CLASSES = ['notumor', 'meningioma', 'pituitary', 'glioma']
CHEMIN_MODELE = 'models/cnn_baseline.keras'
SEUIL_CONFIANCE = 0.7  # 70% confidence threshold

COULEURS = {
    'notumor':    '#639922',
    'meningioma': '#378ADD',
    'pituitary':  '#EF9F27',
    'glioma':     '#E24B4A'
}
```

---

## 📈 Expected Workflow

### Typical Session

```bash
# 1. First run - data preprocessing + training
python main.py
# → Loads images, caches them, trains model (15 min)
# → Saves model & results

# 2. Later sessions - reuse cache, just evaluate
python -c "from main import lancer_evaluation_seule; lancer_evaluation_seule()"
# → Uses cached data (2 sec)
# → Loads saved model
# → Generates metrics

# 3. Try transfer learning
python -c "from main import lancer_transfer_learning; lancer_transfer_learning()"
# → Loads higher-resolution cache
# → Trains TL model
# → Runs Phase 1 (frozen) + Phase 2 (fine-tune)
```

### File Organization After Training

```
results/
├── cnn_baseline_20260507_102222/
│   ├── matrice_confusion.png
│   ├── roc_auc.png
│   └── distribution_confiances.png
└── transfer_efficientnetv2b0_20260506_223305/
    ├── matrice_confusion.png
    ├── roc_auc.png
    └── distribution_confiances.png

models/
├── cnn_baseline.keras
├── transfer_efficientnetb0.keras
└── transfer_efficientnetv2b0.keras
```

---

## 📝 Notes

- **Data Leakage Prevention:** Testing folder completely separate from training
- **Reproducibility:** All random_state=42, deterministic splits
- **Memory:** ~2 GB RAM with full dataset loaded
- **GPU:** Recommended but CPU compatible
- **Medical Domain:** Class weighting + confidence thresholding respect clinical requirements

---

**End of Documentation**
