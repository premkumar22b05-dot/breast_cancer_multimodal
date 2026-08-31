# Multimodal Cross-Attentive Stacking Ensemble for Early Breast Cancer Detection

## ⚠️ DISCLAIMER

**This is a research prototype and does not constitute a clinically validated diagnostic system.**

This implementation is designed for academic research and experimentation purposes only. It should never be used for clinical decision-making, patient diagnosis, or clinical deployment without proper validation, regulatory approval, and clinical supervision.

---

## 📋 Overview

A comprehensive research pipeline implementing a multimodal deep learning system for breast cancer detection from mammographic and clinical data. The system combines:

- **Visual Feature Extraction**: Swin Transformer (global structure) + DenseNet-121 (local texture)
- **Cross-Attention Fusion**: Multi-head attention mechanism between imaging and clinical modalities
- **Gated Multimodal Fusion**: Learned weighting of visual and clinical features
- **Stacking Ensemble**: XGBoost, MLP, and Logistic Regression with meta-learner
- **Rigorous Evaluation**: Density subgroups, calibration analysis, bootstrap confidence intervals
- **Explainability**: Grad-CAM, attention visualization, SHAP feature importance

### Key Features

✅ Strict patient-level train/val/test splitting (no data leakage)  
✅ Automatic multimodal/image-only mode detection  
✅ Medical image preprocessing (DICOM loading, CLAHE, pectoral suppression)  
✅ Out-of-fold ensemble training with proper cross-validation  
✅ Publication-quality evaluation metrics and visualizations  
✅ CPU and GPU support with automatic device detection  
✅ Reproducible results with fixed random seeds  
✅ Comprehensive logging and checkpointing  

---

## 🏗️ Project Structure

```
breast_cancer_multimodal/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment (optional)
├── config.yaml                        # Main configuration file
│
├── run_experiment.py                  # Main entry point
├── train.py                           # Training pipeline
├── evaluate.py                        # Evaluation pipeline
├── inference.py                       # Single-sample inference
│
├── data/
│   ├── raw/                          # Raw DICOM files and annotations
│   │   ├── images/
│   │   ├── annotations/
│   │   └── clinical/
│   ├── processed/                    # Preprocessed images (cached)
│   └── splits/                       # Train/val/test patient lists
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/                         # Data handling modules
│   │   ├── __init__.py
│   │   ├── dataset.py               # Dataset class
│   │   ├── dicom_loader.py          # DICOM loading utilities
│   │   ├── preprocessing.py         # Image preprocessing pipeline
│   │   ├── splits.py                # Train/val/test splitting
│   │   └── leakage_check.py         # Data leakage validation
│   │
│   ├── models/                       # Model architecture components
│   │   ├── __init__.py
│   │   ├── swin_branch.py           # Swin Transformer feature extractor
│   │   ├── densenet_branch.py       # DenseNet-121 feature extractor
│   │   ├── tabular_transformer.py   # Clinical feature processor
│   │   ├── cross_attention.py       # Cross-attention mechanism
│   │   ├── gated_fusion.py          # Gated multimodal fusion
│   │   └── multimodal_model.py      # Complete multimodal model
│   │
│   ├── ensemble/                     # Ensemble learning
│   │   ├── __init__.py
│   │   └── stacking.py              # Stacking ensemble with meta-learner
│   │
│   ├── training/                     # Training utilities
│   │   ├── __init__.py
│   │   ├── losses.py                # Loss functions
│   │   ├── trainer.py               # Training loop
│   │   └── early_stopping.py        # Early stopping logic
│   │
│   ├── evaluation/                   # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── metrics.py               # Standard metrics calculation
│   │   ├── bootstrap_ci.py          # Bootstrap confidence intervals
│   │   ├── calibration.py           # Calibration analysis
│   │   └── subgroup_analysis.py     # Density/age subgroup evaluation
│   │
│   └── explainability/               # Interpretability methods
│       ├── __init__.py
│       ├── gradcam.py               # Grad-CAM visualization
│       ├── attention_rollout.py      # Attention map visualization
│       └── shap_analysis.py          # SHAP feature importance
│
├── experiments/                       # Experiment configurations
│   ├── baselines/
│   ├── ablations/
│   └── multimodal/
│
├── results/                          # Outputs (generated)
│   ├── checkpoints/                 # Model checkpoints
│   ├── predictions/                 # Prediction CSV files
│   ├── metrics/                     # Metric results
│   ├── logs/                        # Training logs
│   └── figures/                     # Generated visualizations
│
└── figures/                          # Output figures (generated)
    ├── architecture/                # Model architecture diagrams
    ├── performance/                 # Performance curves
    ├── explainability/              # Grad-CAM, SHAP, attention maps
    └── subgroup/                    # Density subgroup analysis
```

---

## 🚀 Quick Start

### 1. Installation

**Option A: Using pip**
```bash
git clone https://github.com/premkumar22b05-dot/breast_cancer_multimodal.git
cd breast_cancer_multimodal
pip install -r requirements.txt
```

**Option B: Using conda**
```bash
conda env create -f environment.yml
conda activate breast_cancer_multimodal
```

### 2. Prepare Data

Place your mammography dataset in `data/raw/`:

```
data/raw/
├── images/
│   └── [DICOM files organized by patient/exam]
├── annotations/
│   └── lesion_annotations.csv  (optional)
└── clinical/
    └── patient_clinical_data.csv  (optional)
```

### 3. Data Structure Examples

**Mammography Metadata (auto-detected from DICOM or from CSV):**
```csv
patient_id,exam_id,view,laterality,breast_density,birads,age,finding_present
P001,E001_CC_L,CC,L,B,2,52,0
P001,E001_CC_R,CC,R,B,2,52,0
P001,E001_MLO_L,MLO,L,B,2,52,0
P001,E001_MLO_R,MLO,R,B,2,52,0
```

**Clinical Data (paired by patient_id/exam_id):**
```csv
patient_id,exam_id,age,breast_density,family_history,menopausal_status,bmi,target
P001,E001,52,B,0,post,24.5,0
P002,E002,61,C,1,post,26.1,1
```

### 4. Run Smoke Test (Verify Setup)

```bash
python run_experiment.py --config config.yaml --mode smoke_test
```

This will:
- Load a small sample
- Run preprocessing
- Execute one forward pass
- Train for 1 epoch
- Generate one figure
- Complete in < 5 minutes

### 5. Run Full Training Pipeline

```bash
python run_experiment.py --config config.yaml --mode train
```

### 6. Evaluate Model

```bash
python evaluate.py --config config.yaml
```

### 7. Single-Sample Inference

```bash
python inference.py --exam_id EXAM_ID --config config.yaml
```

---

## 📊 Dataset Support

### Primary: VinDr-Mammo
- 4 mammographic views per exam (CC-L, CC-R, MLO-L, MLO-R)
- Breast-level BI-RADS (1-5)
- Breast density (A-D)
- Abnormality annotations (location, type)
- Patient age and other metadata

### Secondary: CBIS-DDSM
- Automatically switches to image-only mode without paired clinical data
- Supports lesion localization evaluation

### Clinical Data
- Optional paired clinical CSV
- Must contain matching patient/exam identifiers
- Automatic mode detection:
  - **With clinical data**: Multimodal fusion mode
  - **Without clinical data**: Image-only mode

---

## 🔧 Configuration Guide

Edit `config.yaml` to customize:

```yaml
# Data
data:
  dataset: vindr_mammo
  image_size: 512
  use_clahe: true

# Model components
model:
  d_model: 256
  swin:
    pretrained: true
  densenet:
    pretrained: true

# Training
training:
  epochs: 30
  batch_size: 4
  learning_rate: 1e-4
  early_stopping:
    patience: 5

# Ensemble
ensemble:
  base_models: [xgboost, mlp, logistic_regression]
  cv_folds: 5
```

See `config.yaml` for full documentation.

---

## 🧠 Model Architecture

### Visual Branch
```
DICOM Image
    ↓
[Swin Transformer] ──→ Global visual features
    ↓
[DenseNet-121]     ──→ Local visual features
```

### Clinical Branch
```
Clinical Features
    ↓
[Tabular Transformer] ──→ Clinical contextual tokens
```

### Fusion
```
Visual tokens + Clinical tokens
    ↓
[Cross-Attention]
    ↓
[Gated Fusion]
    ↓
Fused feature vector
```

### Ensemble
```
Fused features
    ↓
[Base Learners]
├── XGBoost
├── MLP
└── Logistic Regression
    ↓
[Out-of-Fold Predictions]
    ↓
[Meta-Learner]
    ↓
Final probability
```

---

## 📈 Expected Outputs

After running the pipeline, check:

### Metrics & Results
- `results/metrics.csv` - Main performance metrics
- `results/metrics_with_ci.csv` - Metrics with 95% bootstrap CI
- `results/test_predictions.csv` - Individual predictions
- `results/density_subgroup_results.csv` - Density-specific performance
- `results/baseline_comparison.csv` - Ablation study results
- `results/final_comparison.csv` - All models comparison

### Visualizations
- `figures/roc_curve.png` - ROC curves for all models
- `figures/pr_curve.png` - Precision-Recall curves
- `figures/confusion_matrix.png` - Confusion matrices
- `figures/calibration.png` - Calibration curves
- `figures/density_auc.png` - Density subgroup ROC curves
- `figures/gradcam/` - Grad-CAM heatmaps
- `figures/swin_attention/` - Attention visualizations
- `figures/shap_global.png` - SHAP feature importance

### Reports & Logs
- `results/final_report.md` - Comprehensive markdown report
- `results/run_metadata.json` - Reproducibility information
- `results/logs/training.log` - Training progress
- `results/dashboard.html` - Interactive summary

---

## 🔬 Experiments & Ablations

Run different configurations:

```bash
# Image-only baseline
python run_experiment.py --config config.yaml --experiment baseline_densenet
python run_experiment.py --config config.yaml --experiment baseline_swin

# Fusion variants
python run_experiment.py --config config.yaml --experiment early_concat
python run_experiment.py --config config.yaml --experiment late_fusion

# Cross-attention
python run_experiment.py --config config.yaml --experiment cross_attention

# Full pipeline
python run_experiment.py --config config.yaml --experiment proposed

# All ablations
python run_experiment.py --config config.yaml --experiment ablation_all
```

Results saved to: `results/ablation_results.csv`

---

## 📊 Evaluation Metrics

The system computes:

**Classification Metrics:**
- Accuracy, Precision, Recall (Sensitivity), Specificity
- F1-score, ROC-AUC, PR-AUC
- Matthews Correlation Coefficient (MCC)
- Balanced Accuracy, Cohen's Kappa
- Log Loss, Brier Score

**Additional:**
- Sensitivity at 90% specificity
- Sensitivity at 95% specificity
- 95% Confidence Intervals (bootstrap, n=2000)

**Subgroup Analysis:**
- Breast density groups (A, B, C, D)
- Age groups (optional)
- Lesion size groups (if annotations available)

**Calibration:**
- Reliability diagram
- Calibration curve
- Expected Calibration Error (ECE)
- Temperature scaling (optional)

---

## 🎯 Key Safety Features

### Data Leakage Prevention
✅ Patient-level splitting (no exam mixing)  
✅ Automatic leakage detection script  
✅ No test data in preprocessing  
✅ Train-only imputation and normalization  

### Ensemble Safety
✅ Enforced out-of-fold predictions  
✅ Cross-validation with stratification  
✅ Meta-learner never trained on in-sample predictions  

### Scientific Integrity
✅ No fabricated data or labels  
✅ Honest claim verification  
✅ Clear documentation of dataset limitations  
✅ Reproducible random seeds  

---

## 🖥️ System Requirements

### Minimum
- **CPU**: 4+ cores
- **RAM**: 16 GB
- **GPU**: Optional (4 GB+ VRAM recommended)
- **Disk**: 50 GB+ for dataset + outputs

### Recommended
- **CPU**: 8+ cores
- **RAM**: 32 GB
- **GPU**: NVIDIA GPU (12+ GB VRAM)
- **Disk**: 100 GB+

### Testing on CPU
Set in `config.yaml`:
```yaml
compute:
  device: cpu
training:
  batch_size: 1
model:
  swin:
    freeze_backbone: true
```

---

## 🔍 Data Leakage Check

Before training, verify no patient overlap:

```bash
python src/data/leakage_check.py --config config.yaml
```

Output example:
```
Dataset Leakage Check
====================
Total unique patients: 500
Total exams: 542
Train: 379 exams (372 patients)
Val: 81 exams (79 patients)
Test: 82 exams (80 patients)

Intersections:
  train ∩ val = 0 ✓
  train ∩ test = 0 ✓
  val ∩ test = 0 ✓

Status: PASS - No data leakage detected
```

---

## 📝 Important Notes

### About Claims
- The system does **NOT** automatically detect tumors smaller than 2 mm unless validated ground truth and explicit verification are provided
- Claims about lesion size require valid pixel spacing and size annotations
- Density subgroup analysis only reported if groups have sufficient samples (n≥10)

### About Clinical Data
- Clinical data must be user-provided and genuinely paired
- Do not use UCI Breast Cancer Coimbra as paired data with VinDr-Mammo
- Missing values in clinical data handled via train-only imputation
- MICE only applied if missing values actually exist

### About Results
- All metrics computed from actual experiments (never fabricated)
- Metrics with insufficient class representation report NaN
- Confidence intervals use bootstrap resampling (reproducible seed)

---

## 🚦 Troubleshooting

### Issue: CUDA out of memory
```yaml
# config.yaml
training:
  batch_size: 1
model:
  swin:
    freeze_backbone: true
```

### Issue: Data leakage detected
```
ERROR: Train/test patient overlap found
Action: Check data/splits/, ensure --seed=42
```

### Issue: Missing paired clinical data
```
INFO: No clinical data found - using image-only mode
(This is normal and expected if clinical data is unavailable)
```

### Issue: Smoke test fails
```bash
# Check imports
python -c "import torch; import timm; import xgboost; print('OK')"

# Run with verbose logging
python run_experiment.py --config config.yaml --mode smoke_test --loglevel DEBUG
```

---

## 📚 References & Documentation

See individual module docstrings:
- `src/data/preprocessing.py` - Image preprocessing details
- `src/models/cross_attention.py` - Cross-attention equations
- `src/ensemble/stacking.py` - Ensemble algorithm documentation
- `src/evaluation/metrics.py` - Metric definitions

---

## 📄 License

Apache License 2.0 - See LICENSE file

---

## ✍️ Citation

If you use this code in your research, please cite:

```bibtex
@misc{breast_cancer_multimodal_2024,
  title={Multimodal Cross-Attentive Stacking Ensemble for Early Breast Cancer Detection},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/premkumar22b05-dot/breast_cancer_multimodal}}
}
```

---

## 👤 Author & Support

For questions or issues:
- Open a GitHub Issue: [Issues](https://github.com/premkumar22b05-dot/breast_cancer_multimodal/issues)
- Check existing documentation in `src/` module docstrings
- Review `config.yaml` comments for configuration help

---

## 📋 Checklist Before Using

- [ ] Dataset placed in `data/raw/`
- [ ] Clinical CSV (if available) placed in `data/clinical/`
- [ ] `requirements.txt` installed
- [ ] Smoke test passes
- [ ] `config.yaml` reviewed and customized
- [ ] Data leakage check passes
- [ ] Random seed fixed (reproducibility)
- [ ] GPU/CPU device correctly detected
- [ ] Output directories writable

---

**Last Updated**: August 31, 2024  
**Status**: Phase 1 Complete (Project Foundation)
