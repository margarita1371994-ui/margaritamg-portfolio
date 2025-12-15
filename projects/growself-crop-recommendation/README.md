# GrowSelf — Crop Recommendation System

**End-to-end applied Data Science project** developed as part of my Master’s Thesis (TFM) at EOI.

The goal is to **predict and recommend agricultural land use / crop type**
using **soil, climate and geospatial information**.

---

## Problem statement

Agricultural decision-making depends heavily on:
- Soil properties
- Climate conditions
- Location and environment

This project builds a **multiclass classification model** to predict the most suitable crop category (`cultivo_grupo`) for a given soil profile.

---

## Data sources

- **CARBOSOL (PANGAEA)**  
  Soil profiles and horizons for Spain  
- **AEMET OpenData API**  
  Daily climate data (2017) from nearby meteorological stations

All data sources are **public**.

---

## Project structure

projects/growself-crop-recommendation/
│
├─ data/
│ ├─ CARBOSOL_profile.tab
│ ├─ CARBOSOL_horizons.tab
│ └─ CARBOSOL_info.tab
│
├─ notebooks/
│ ├─ 01_data_acquisition_eda.py
│ ├─ 02_split_modeling_smote.ipynb
│ ├─ requirements.txt
│ └─ README.md
│
├─ outputs/
│ └─ eda/
│ ├─ dataset_final_2017_full.csv
│ ├─ analysis/
│ ├─ figures/
│ └─ README.md
│
├─ .env # API keys (not tracked)
├─ .gitignore
└─ README.md

---

## Notebooks overview

### 01_data_acquisition_eda.py
- Raw data ingestion (CARBOSOL + AEMET)
- Robust parsing of PANGAEA files
- Climate data download with retry & caching
- Feature engineering:
  - Soil aggregation (horizons → profile)
  - Climate aggregation (daily → annual)
- Target creation from text descriptions
- Crop grouping (`cultivo_grupo`)
- Data quality checks and EDA outputs

 Produces the final modeling dataset.

---

### 02_split_modeling_smote.ipynb
- Train / test split
- Feature selection
- Handling class imbalance with **SMOTE**
- Model training:
  - Baseline models
  - XGBoost
- Model evaluation:
  - F1-macro (primary metric)
  - Confusion matrices
  - Per-class performance

---

## Modeling details

- **Target variable**: `cultivo_grupo`
- **Type**: Multiclass classification
- **Imbalance handling**: SMOTE
- **Main model**: XGBoost
- **Metric**: F1-macro ≈ **0.69** (baseline academic result)

---

## Reproducibility

1. Create a virtual environment
2. Install dependencies:
   ```bash
   pip install -r notebooks/requirements.txt
