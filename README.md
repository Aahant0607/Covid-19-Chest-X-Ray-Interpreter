<div align="center">

# 🫁 Explainable Computer Vision — COVID-19 Chest X-Ray Classification

**AIMS DTU Research Intern 2026**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://kaggle.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Ensemble%20Accuracy-99.08%25-brightgreen?style=flat)]()

> End-to-end explainable medical image classification using **EfficientNet-B4** and **DeiT-Small**, with five standard saliency methods benchmarked against a novel proposed method — **UWSA (Uncertainty-Weighted Synergistic Aggregation)**.

**Author:** Aahant Kumar &nbsp;|&nbsp; **Roll No:** 24/EC/001

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Results](#-results)
- [Explainability Methods](#-explainability-methods)
- [Novel Method: UWSA & PACS-UWSA](#-novel-method-uwsa--pacs-uwsa)
- [Notebook Structure](#-notebook-structure)
- [Setup & Usage](#-setup--usage)
- [Dependencies](#-dependencies)
- [Project Structure](#-project-structure)

---

## 🔍 Overview

This project trains two deep learning models on a balanced 3-class chest X-ray dataset (COVID-19, Normal, Viral Pneumonia) and applies a comprehensive explainability analysis to interpret model predictions. The work covers:

- ✅ **Dual-model training** — CNN (EfficientNet-B4) + ViT (DeiT-Small) on dual Tesla T4 GPUs
- ✅ **Ensemble inference** with Test-Time Augmentation (TTA)
- ✅ **Five explainability methods** — Grad-CAM, Attention Rollout, Integrated Gradients, LIME, SHAP
- ✅ **Quantitative benchmarking** — Insertion/Deletion AUC, Saliency Entropy, AOPC
- ✅ **Novel method** — UWSA and PACS-UWSA, both outperforming standard methods on saliency entropy

---

## 🏗 Architecture

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT: Chest X-Ray Image                     │
│                           224 × 224 × 3                             │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │         DATA PREPROCESSING         │
              │  • Resize → 224×224                │
              │  • Normalize (ImageNet stats)      │
              │  • Train Augmentation:             │
              │    – RandomHorizontalFlip (p=0.5)  │
              │    – RandomRotation ±10°           │
              │    – ColorJitter (b/c ±0.2)        │
              └────────┬──────────────┬────────────┘
                       │              │
         ┌─────────────▼───┐      ┌────▼────────────────┐
         │  EfficientNet-B4│      │   DeiT-Small        │
         │  (CNN Backbone) │      │ (Vision Transformer)│
         │                 │      │                     │
         │  Compound Scale │      │  Patch Size: 16     │
         │  17.55M params  │      │  21.67M params      │
         │  ~36s / epoch   │      │  ~22s / epoch       │
         │                 │      │                     │
         │  Conv Layers ──►│      │  12 Attn Blocks ──► │
         │  Global AvgPool │      │  [CLS] Token        │
         │  FC Head (×3)   │      │  FC Head (×3)       │
         └────────┬────────┘      └────────┬────────────┘
                  │  Softmax probs         │  Softmax probs
                  └──────────┬─────────────┘
                             │
              ┌──────────────▼───────────────┐
              │     ENSEMBLE (Soft Vote)     │
              │   avg(p_CNN + p_ViT) / 2     │
              │   + Test-Time Augmentation   │
              │                              │
              │   ✅ 99.08% Test Accuracy    │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼──────────────────────────────┐
              │         EXPLAINABILITY MODULE               │
              │                                             │
              │  ┌─────────────┐   ┌────────────────────┐  │
              │  │  CNN-based  │   │    ViT-based        │  │
              │  │  Grad-CAM   │   │  Attention Rollout  │  │
              │  └─────────────┘   └────────────────────┘  │
              │                                             │
              │  ┌─────────────┐   ┌──────┐   ┌────────┐  │
              │  │  Integrated │   │ LIME │   │  SHAP  │  │
              │  │  Gradients  │   │      │   │        │  │
              │  └─────────────┘   └──────┘   └────────┘  │
              │                                             │
              │  ┌─────────────────────────────────────┐   │
              │  │  🆕 UWSA / PACS-UWSA (Proposed)     │   │
              │  │  Uncertainty-Weighted Synergistic    │   │
              │  │  Aggregation                        │   │
              │  └─────────────────────────────────────┘   │
              └──────────────┬──────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │    QUANTITATIVE EVALUATION  │
              │  • Insertion AUC  (↑)       │
              │  • Deletion AUC   (↓)       │
              │  • Saliency Entropy (↓)     │
              │  • AOPC           (↑)       │
              └─────────────────────────────┘
```

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Epochs | 20 |
| Learning Rate | 1×10⁻⁴ |
| Optimiser | AdamW (wd = 1×10⁻⁴) |
| Scheduler | Cosine Annealing (T_max=20) |
| Loss | CrossEntropyLoss + Label Smoothing (ε=0.1) |
| Hardware | 2× Tesla T4 (DataParallel) |

---

## 📊 Dataset

| Split | COVID | Normal | Viral Pneumonia | Total |
|---|---|---|---|---|
| Train | 1,000 | 1,000 | 1,000 | 3,000 |
| Validation | 200 | 200 | 200 | 600 |
| Test | 145 | 145 | 145 | 435 |

Dataset hosted on Kaggle: [`aahantkumar/covid19dataset`](https://www.kaggle.com/datasets/aahantkumar/covid19dataset)

---

## 🏆 Results

### Classification Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| EfficientNet-B4 | 94.02% | 94.07% | 94.02% | 94.02% |
| EfficientNet-B4 + TTA | 94.02% | 94.07% | 94.02% | 94.02% |
| DeiT-Small (ViT) | 98.85% | 98.86% | 98.85% | 98.85% |
| DeiT-Small + TTA | 98.85% | 98.86% | 98.85% | 98.85% |
| **Ensemble (B4 + ViT)** | **99.08%** | **98.62%** | **98.62%** | **98.62%** |

### Ensemble Per-Class Breakdown

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| COVID | 0.99 | 0.98 | 0.98 |
| Normal | 0.98 | 0.98 | 0.98 |
| Viral Pneumonia | 0.99 | **1.00** | **1.00** |

> Viral Pneumonia achieves perfect recall and F1 — the most clinically critical result, as missed pneumonia diagnoses carry the highest risk of treatment failure.

---

## 🔬 Explainability Methods

| Method | Type | Backbone | Description |
|---|---|---|---|
| **Grad-CAM** | Gradient | EfficientNet-B4 | Backpropagates class gradients to final conv layer to produce class-discriminative spatial heatmaps |
| **Attention Rollout** | Attention | DeiT-Small | Recursively multiplies normalised attention matrices across all ViT layers to trace patch influence on [CLS] token |
| **Integrated Gradients** | Attribution | EfficientNet-B4 | Integrates gradients along a 50-step linear path from black baseline to input; satisfies sensitivity & implementation invariance axioms |
| **LIME** | Surrogate | Ensemble | Fits local linear surrogate on perturbed superpixel segments (100 samples, quickshift); GPU-accelerated batch inference |
| **SHAP** | Shapley | Ensemble | GradientExplainer estimates pixel-level Shapley values; memory-optimised implementation |

### Quantitative Benchmark

| Method | Ins. AUC ↑ | Del. AUC ↓ | Entropy ↓ | AOPC ↑ |
|---|---|---|---|---|
| Grad-CAM | 0.5881 | 0.4598 | 9.9174 | 0.3838 |
| Attn Rollout | 0.5481 | 0.4637 | 10.6602 | 0.4110 |
| Integ. Grads | 0.5290 | 0.4143 | 10.3329 | 0.4082 |
| LIME | 0.7184 | 0.4329 | 9.9409 | 0.3987 |
| SHAP | 0.4042 | 0.3923 | 10.5327 | 0.3961 |
| **UWSA (Ours)** | 0.6752 | 0.4488 | **8.3859 ★** | 0.3828 |
| **PACS-UWSA (Ours)** | 0.6476 | 0.4451 | 9.5015 | 0.3989 |

> ↑ higher is better &nbsp;|&nbsp; ↓ lower is better &nbsp;|&nbsp; ★ best overall

---

## 🆕 Novel Method: UWSA & PACS-UWSA

### Motivation

Existing methods treat all predictions equally — a highly uncertain prediction and a near-certain one produce the same style of attribution map. UWSA corrects this: uncertainty from the model's own softmax output is used to **dynamically sharpen or relax** the saliency map.

### UWSA — Uncertainty-Weighted Synergistic Aggregation

```
1. Predictive Entropy:   H  = −Σ pᵢ·log(pᵢ) / log(C)       [C = 3 classes]
2. Confidence:           conf = 1 − H                         [0 = random, 1 = certain]
3. Synergy Fusion:       S  = 0.6 × Grad-CAM + 0.4 × IG      [semantic + edge blend]
4. Power Sharpening:     W  = S ^ (1 + conf × 3)             [exponent ∈ [1.0, 4.0]]
5. Normalise:            UWSA = (W − min) / (max − min)
```

The power exponent is the key innovation — it compresses saliency to a **sharp peak for confident predictions**, and relaxes it for uncertain ones. Result: **best saliency entropy of all methods (8.3859)** while maintaining competitive insertion faithfulness (0.6752).

### PACS-UWSA — Prior-Assisted Contrast-Suppressed UWSA

Extends UWSA with three clinically motivated enhancements:

```
1. Anatomical Spatial Prior
   └── 2D Gaussian mask centred on lung field (σ=0.6, slightly superior-shifted)
       Penalises peripheral scanner artifacts, text labels, and patient markers

2. Otsu Noise Suppression
   └── Automatic thresholding zeros activations below 60% of Otsu threshold
       Removes low-level background noise without manual tuning

3. Guided Edge-Preserving Filter
   └── OpenCV guided/bilateral filter using original image as guide
       Preserves sharp anatomical boundaries, smooths homogeneous regions
```

---

## 📓 Notebook Structure

The entire project runs as a **single sequential Kaggle notebook** (`explainable-cv_latest.ipynb`). Each cell block corresponds to a self-contained stage:

| Cell | Stage | Description |
|---|---|---|
| `0` | **Setup** | Install `grad-cam`, `timm`, `captum`, `lime`, `shap`; verify GPU environment (2× Tesla T4, PyTorch 2.10, CUDA) |
| `1` | **Imports** | All library imports — PyTorch, torchvision, timm, captum, sklearn, matplotlib |
| `2` | **Config** | Paths, hyperparameters (`IMG_SIZE=224`, `BATCH_SIZE=32`, `EPOCHS=20`, `LR=1e-4`), dataset verification |
| `3` | **Data Pipeline** | Train/val/test transforms, `ImageFolder` datasets, class-frequency inverse weights, `DataLoader` (4 workers) |
| `4` | **Model Builder** | `build_model()` — creates EfficientNet-B4 or DeiT-Small via `timm`, wraps with `DataParallel` |
| `5` | **Training Loop** | `train_model()` — AdamW + CosineAnnealing + CrossEntropyLoss with label smoothing, best-weight checkpointing |
| `6` | **Train EfficientNet-B4** | Full 20-epoch training run → best val accuracy 93.17% |
| `7` | **Train DeiT-Small** | Full 20-epoch training run → best val accuracy 98.83% |
| `8` | **Training Curves** | Loss/accuracy plots for both models side-by-side |
| `9` | **Evaluation** | Test-set classification reports + confusion matrices for all 5 variants (both models + TTA + ensemble) |
| `10` | **Probe Samples** | Load 9 test images (3 per class) for explainability probing; unwrap DataParallel models |
| `11` | **Grad-CAM** | `pytorch_grad_cam` on EfficientNet-B4 final conv layer; overlay heatmaps on original images |
| `12` | **Attention Rollout** | Custom implementation — patches DeiT attention to bypass Flash Attention; recursively multiplies attention matrices |
| `13` | **Integrated Gradients** | `captum.attr.IntegratedGradients`, 50 steps, black baseline, channel-averaged absolute attributions |
| `14` | **SHAP** | `shap.GradientExplainer` on ensemble; memory-optimised |
| `15` | **LIME** | `lime.lime_image`, quickshift segmentation, 100 perturbations, GPU-accelerated batch inference |
| `16` | **Visual Comparison** | 6-method comparison grid (Original + 5 methods) across all 9 probe images |
| `17` | **Metric Evaluation** | Batched Insertion/Deletion AUC, Saliency Entropy, AOPC for all 5 standard methods |
| `18` | *(marker)* | Novel method section marker |
| `19` | **UWSA** | Novel method implementation, saliency generation, and full metric evaluation vs all baselines |
| `20` | **PACS-UWSA** | Extended novel method with anatomical prior + Otsu suppression + guided filter; full evaluation |

---

## ⚙️ Setup & Usage

### Kaggle (Recommended)

```bash
# 1. Fork or import the notebook to Kaggle
# 2. Add the dataset:
#    Kaggle Datasets → aahantkumar/covid19dataset
# 3. Enable GPU: Settings → Accelerator → GPU T4 × 2
# 4. Run All Cells (Runtime → Run All)
```

### Local (GPU required)

```bash
# Clone
git clone https://github.com/aahantkumar/explainable-cv.git
cd explainable-cv

# Install dependencies
pip install torch torchvision timm captum lime shap grad-cam opencv-python-headless scikit-image scikit-learn matplotlib

# Update dataset paths in Cell 2:
# TRAIN_PATH = "path/to/COVID_19_dataset/train"
# VAL_PATH   = "path/to/COVID_19_dataset/val"
# TEST_PATH  = "path/to/COVID_19_dataset/test"

# Launch notebook
jupyter notebook explainable-cv_latest.ipynb
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` / `torchvision` | 2.10 | Model training & inference |
| `timm` | 1.0.26 | EfficientNet-B4 & DeiT-Small pretrained models |
| `captum` | 0.9.0 | Integrated Gradients |
| `grad-cam` | latest | Grad-CAM heatmaps |
| `shap` | latest | SHAP GradientExplainer |
| `lime` | latest | LIME image explainer |
| `opencv-python` | latest | Image processing & guided filter (PACS-UWSA) |
| `scikit-image` | latest | Superpixel segmentation (LIME) |
| `scikit-learn` | latest | Classification metrics & confusion matrix |
| `matplotlib` | latest | Visualisations |
| `numpy` | latest | Array operations |

---

## 📁 Project Structure

```
explainable-cv/
│
├── explainable-cv_latest.ipynb   # Main notebook (all 21 cells, sequential)
│
├── outputs/                      # Generated during notebook run
│   ├── best_efficientnet_b4.pth  # Best CNN checkpoint
│   ├── best_deit_small.pth       # Best ViT checkpoint
│   ├── gradcam_grid.png          # Grad-CAM visualisations
│   ├── attn_rollout_grid.png     # Attention Rollout visualisations
│   ├── ig_grid.png               # Integrated Gradients visualisations
│   ├── shap_grid.png             # SHAP visualisations
│   ├── lime_grid.png             # LIME visualisations
│   ├── method_comparison.png     # 6-method comparison grid
│   ├── uwsa_vs_gradcam.png       # UWSA vs Grad-CAM comparison
│   └── pacs_uwsa_grid.png        # PACS-UWSA visualisations
│
├── report/
│   └── ExCV_Report_AahantKumar.docx   # Final submission report
│
└── README.md
```

---

## 📈 Key Findings

- **DeiT-Small outperforms EfficientNet-B4** (98.85% vs 94.02%) — ViT's attention mechanism extracts more discriminative features from chest X-ray patterns
- **LIME is the most faithful** standard method (Insertion AUC 0.7184) but not the most focused
- **Grad-CAM is the most focused** standard method (lowest entropy 9.92) but spatially coarse
- **UWSA achieves the sharpest attributions of all methods** (entropy 8.3859) by coupling saliency sharpness to prediction confidence
- **PACS-UWSA improves clinical plausibility** by suppressing peripheral scanner artifacts via anatomical priors

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Made with ❤️ for AIMS DTU Research Intern 2026 &nbsp;|&nbsp; Aahant Kumar &nbsp;|&nbsp; 24/EC/001

</div>
