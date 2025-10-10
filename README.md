# Bone Fracture Detection Practicum

## 🔍 Project Overview

This repository consolidates a seven-week applied research practicum focused on automated detection of upper-limb bone fractures from radiographic imagery. The work unifies exploratory data analysis, model development, transfer learning, hyperparameter search, threshold optimization, and interpretability into a single, reproducible workflow. All experiments are captured inside the `final_clean_pipeline.ipynb` notebook, enabling seamless review without rerunning expensive training loops.

### Objectives

- Deliver a clinically relevant binary classifier that distinguishes fractured from non-fractured x-rays.
- Investigate baseline convolutional networks versus transfer learning backbones (ResNet50, EfficientNet-B0).
- Quantify generalization with validation metrics (accuracy, sensitivity/recall, specificity, ROC-AUC, PR curves).
- Provide interpretable outputs (Grad-CAM, gradient attention) to support clinician trust.
- Document the full experimentation journey from scoping to evaluation, aligning with the weekly practicum plan.

### Methodology

1. **Data consolidation (Weeks 1–2)**
   - Combined YOLO-format labels with FracAtlas imagery into a harmonized directory structure (`data/combined_organized`).
   - Implemented resilient dataset loaders with oversampling and weighted sampling to handle class imbalance and occasional file corruption.
   - Applied ImageNet-style normalization and lightweight augmentations (horizontal flip, rotation, color jitter) to expand the effective training distribution.

2. **Model development (Weeks 3–4)**
   - Defined a baseline `SimpleCNN` to establish a performance floor.
   - Restored transfer learning architectures (ResNet50, EfficientNet-B0) with configurable feature freezing.
   - Introduced toggles for optional quick retraining and fine-tuning, preserving saved checkpoints to keep the notebook fully executable without long waits.

3. **Experiment tracking (Week 5)**
   - Parsed cached hyperparameter search logs (`week5_hyperparameter_results/`) to highlight the strongest configurations by sensitivity and F1-score.

4. **Evaluation and visualization (Week 6)**
   - Loaded the best checkpoints from `saved_models/` and evaluated them on the validation set.
   - Generated comparison tables, confusion matrices, ROC curves, and precision-recall curves to illustrate trade-offs.

5. **Optimization and interpretability (Week 7)**
   - Tuned probability thresholds to maximize F1-score for the top-performing model.
   - Produced Grad-CAM overlays and gradient attention maps on validation samples, surfacing decision-critical regions.

### Dataset

- Training and validation splits live under `data/combined_organized/train` and `data/combined_organized/val` respectively.
- Original YOLO-formatted sources reside in `data/bonefractureyolo`, and auxiliary assets from FracAtlas live in `data/FracAtlas`.
- The notebook includes an optional consolidation utility if the combined dataset needs rebuilding (disabled by default).

### Key Results

- ResNet50 transfer learning checkpoint delivered the strongest validation F1-score on the fractured class while maintaining high specificity.
- Threshold tuning improved the sensitivity-specificity balance, pushing F1 higher without retraining.
- ROC-AUC scores demonstrated consistent ranking across models, with transfer backbones outperforming the baseline CNN.
- Interpretability passes confirmed that heatmap attention aligns with anatomical regions of interest, reinforcing model trustworthiness.

### Challenges & Mitigation

| Challenge | Mitigation |
| --- | --- |
| **Imbalanced classes** made fractured cases rarer than non-fractured ones. | Applied oversampling, weighted sampling, and monitored class distributions per epoch. |
| **Heterogeneous image quality** produced occasional corrupted or truncated files. | Added a lenient loader that substitutes safe placeholders and logs problematic samples. |
| **Compute constraints** prevented repeated full training during evaluation. | Cached trained checkpoints and designed optional toggles to skip heavy loops on routine runs. |
| **Clinician interpretability expectations** required transparent predictions. | Integrated Grad-CAM and gradient attention visualizations, showcasing them on validation imagery. |

## 📁 Repository Structure

```
.
├── final_clean_pipeline.ipynb    # Unified seven-week notebook with code, outputs, and explanations
├── weekly_experiments.ipynb      # Historical notebook with incremental experiments (reference)
├── data_cleaning_backup.ipynb    # Auxiliary exploration and preprocessing notebook
├── data/                         # Consolidated dataset folders and source assets
│   ├── bonefractureyolo/         # YOLO-formatted dataset components
│   ├── combined/                 # Intermediate merged assets
│   ├── combined_organized/      # Train/val split used by the pipeline
│   └── FracAtlas/                # Supplementary images, annotations, and utilities
├── saved_models/                 # Serialized checkpoints and metadata for all evaluated models
├── results/                      # Aggregated plots, Grad-CAM outputs, and performance summaries
├── week5_hyperparameter_results/ # JSON logs from tuning sweeps
├── week6_results/                # Week 6 visualization exports
└── README.md                     # Landing page (this file)
```

## 🚀 Quick Start

1. **Environment setup**
   - Python 3.10 or later is recommended.
   - Install core dependencies (PyTorch, torchvision, pandas, scikit-learn, matplotlib, seaborn, efficientnet-pytorch) if you plan to rerun the notebook end-to-end.

2. **Data preparation**
   - Ensure the `data/combined_organized` directories are populated. They are included in this repository structure; the notebook verifies their presence.

3. **Run the consolidated notebook**
   - Open `final_clean_pipeline.ipynb` in VS Code or JupyterLab.
   - Execute cells sequentially. Heavy training loops are disabled by default; enable toggles if you need to rerun short fine-tuning routines.

4. **Review outputs**
   - Evaluation tables and plots appear inline.
   - Grad-CAM visualizations display interpretability overlays for random validation samples.

### Optional Commands

```bash
# Create a fresh virtual environment (macOS/Linux)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt  # (Create with the packages you use if not already present)
```

> **Note:** Since checkpoints are already provided, the notebook can be executed fully offline without additional downloads. Only the optional EfficientNet dependency is installed on demand if missing.

## 🧪 Validation Checklist

- Notebook confirms dataset availability before processing.
- Evaluation metrics are reproducible by re-running inference cells (no retraining required).
- Interpretability visuals regenerate deterministically thanks to fixed random seeds.
