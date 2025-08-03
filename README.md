# Cross-Gated Multi-Path Attention Fusion (CG-MAF) for Brain Tumor MRI Classification

![Network Architecture](Network.png)

This repository provides the complete source code, ablation experiments, and results for our research:

> **Cross-Gated Multi-Path Attention Fusion with Gate-Consistency Loss for Explainable Brain Tumor MRI Classification**  


---

## Overview

We propose a novel hybrid deep learning framework combining **ResNet-50** and **ViT-Base** backbones using a dynamic **Cross-Gated Multi-Path Attention Fusion (CG-MAF)** mechanism, reinforced by a **Gate-Consistency Loss (GCL)** for interpretable and robust brain tumor MRI classification.

- **Dataset**: Composite Brain MRI (Glioma, Meningioma, Pituitary, Healthy)
- **Accuracy**: 99.81%  |  **AUC**: 0.9999
- **Explainability**: Grad-CAM, Integrated Gradients
- **Multi-GPU Support**: Up to 7 GPUs

---

## Repository Structure

```
├── Ablation 1
│   ├── Code Files
│   └── JSON Results
├── Ablation 2
│   ├── Code Files
│   └── JSON Results
├── Ablation 3
│   ├── Code Files
│   └── JSON Results
├── Source Code
│   ├── Final Model Code (.py)
│   └── Supporting Scripts
├── requirements.txt
└── README.md
```

---

## Pre-trained Models & Results

All trained models are available here:

🔗 [Google Drive Link](https://drive.google.com/drive/folders/15f4Pq5fe9Gdwjfi9rGNAE_IG8NRuuuYF?usp=sharing)

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/YourRepo.git
cd YourRepo
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Multi-GPU Training Example
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python R-v-cgmaf-loss.py
```

---

## Ablation Studies

| Ablation | Description | Folder |
|----------|-------------|--------|
| **Ablation 1** | Naive feature fusion vs. CG-MAF Fusion | Ablation 1 |
| **Ablation 2** | Impact of Gate-Consistency Loss (GCL) | Ablation 2 |
| **Ablation 3** | Backbone comparison: ResNet-only vs. ViT-only vs. Hybrid | Ablation 3 |

Each folder contains code and JSON results with accuracy, AUC, F1-score, confusion matrices, and Grad-CAM visualizations.

---

## Expected Outputs

Running the code will produce:

1. **5-Fold Stratified Cross-Validation Training & Evaluation**
2. **Metrics JSON files** (Accuracy, Precision, Recall, F1-score, AUC, Confusion Matrix)
3. **Grad-CAM visualizations** per tumor class
4. **Best model checkpoints** per fold
5. **Training logs** per epoch
---

## Contact
- **Muhammad Ahmad**: [ahmajameel7171@gmail.com]
---
