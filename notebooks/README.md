# Notebooks — Data Pipeline & Modeling

This folder contains the full, reproducible data science workflow for the
**GrowSelf – Crop Recommendation** project (TFM, EOI).

The pipeline follows a clear separation of concerns:
- Data acquisition & feature engineering
- Exploratory analysis 
- Train / validation split and modeling with class imbalance handling

---

## 01_data_acquisition_eda.py

**Purpose**  
End-to-end data ingestion, feature engineering and exploratory analysis.

**Main responsibilities**
- Robust ingestion of CARBOSOL data (PANGAEA `.tab` files)
- Climate data extraction from AEMET API (daily data, year 2017)
- Spatial matching: soil profiles → nearest active AEMET station (Haversine)
- Climate aggregation per profile (mean, sum, rain days)
- Feature engineering:
  - Soil horizons aggregation (mean / median per profile)
  - Target variable extraction from textual descriptions
  - Advanced crop grouping (`cultivo_grupo`)
- Data quality checks:
  - Missing values
  - Outliers (IQR)
  - Distribution sanity checks
- Dataset generation:
  - `dataset_final_2017_full.csv`
  - `dataset_final_2017_model.csv` (filtered, ready for modeling)

**Outputs**
outputs/eda/
├── dataset_final_2017_full.csv
├── model/
│ └── dataset_final_2017_model.csv
├── analysis/
│ ├── missing_rates.csv
│ ├── outlier_rates_iqr.csv
│ ├── psi_by_provincia.csv
│ ├── corr_matrix.png
│ └── analysis_report.csv

---

## 02_split_modeling_smote.ipynb

**Purpose**  
Modeling notebook focused on classification under class imbalance.

**Main steps**
1. Load final modeling dataset
2. Feature / target separation
3. Train–test split (stratified)
4. Preprocessing:
   - Numeric scaling
   - Categorical encoding (if applicable)
5. Class imbalance handling:
   - SMOTE applied on training set only
6. Model training:
   - Baseline models
   - XGBoost classifier
7. Evaluation:
   - F1-macro
   - Confusion matrix
   - Class-wise performance

**Key focus**
- No data leakage
- SMOTE applied **after split**
- Metrics aligned with multi-class imbalance

---

## Reproducibility

To reproduce the full pipeline:
1. Create a `.env` file with your AEMET API key
2. Install dependencies from `requirements.txt`
3. Run `01_data_acquisition_eda.py`
4. Run `02_split_modeling_smote.ipynb`

---

## Notes

- Raw datasets are public but **not stored** in the repository
- All outputs are fully reproducible
- The project is designed for clarity, auditability and real-world constraints
