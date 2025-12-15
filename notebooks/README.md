# Notebooks — Data Pipeline & Modeling

This folder contains the full analytical and modeling pipeline for the
**GrowSelf crop recommendation project**, developed as part of the EOI Master
Final Project (TFM).

The project follows an end-to-end, production-oriented workflow:
data ingestion → feature engineering → quality checks → modeling.

---

##  01_data_acquisition_eda.py

**Purpose**  
End-to-end data acquisition, preprocessing and exploratory analysis.

**Main responsibilities**
- Robust ingestion of **CARBOSOL** soil datasets (PANGAEA)
- Extraction of crop / land-use information from textual descriptions
- Advanced crop grouping logic (`cultivo_grupo`)
- Download and parsing of **AEMET climate data (2017)**
- Spatial assignment of nearest weather station (Haversine distance)
- Climate data imputation (station/month/global median strategy)
- Aggregation of daily climate into yearly indicators
- Soil horizon aggregation at profile level
- Generation of the final analytical dataset
- Advanced EDA & data quality checks:
  - Missing values
  - Outliers (IQR)
  - Class balance
  - PSI (Population Stability Index)
  - Correlation analysis
- Export of clean datasets to `outputs/eda/`

**Outputs**
- `dataset_final_2017_full.csv`
- `dataset_final_2017_model.csv`
- Multiple EDA artifacts (plots, summaries, reports)

This script is intentionally kept as a `.py` file to emphasize
**reproducibility, modularity and production-style pipelines**.

---

##  02_split_modeling_smote.ipynb

**Purpose**  
Supervised modeling notebook focused on classification and evaluation.

**Main steps**
- Load curated modeling dataset
- Train / test split
- Feature preprocessing
- Class imbalance handling using **SMOTE**
- Model training (tree-based models)
- Performance evaluation:
  - Confusion matrix
  - Precision / Recall / F1 (macro)
- Feature importance inspection

This notebook focuses **only on modeling**, keeping data leakage under control.

---

##  Environment & dependencies

See `requirements.txt` for the full list of required Python packages.

Sensitive credentials (AEMET API key) are handled via a `.env` file
and are **not stored in the repository**.

---

##  Reproducibility

To reproduce the full pipeline:

```bash
pip install -r requirements.txt
python notebooks/01_data_acquisition_eda.py
jupyter notebook notebooks/02_split_modeling_smote.ipynb
