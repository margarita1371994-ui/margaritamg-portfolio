# EDA Outputs — GrowSelf (CARBOSOL + AEMET)

This folder contains all the **artifacts generated during the Exploratory Data Analysis (EDA) and feature engineering stage** of the GrowSelf project (TFM – EOI).

The outputs are produced mainly by:

- `notebooks/01_data_acquisition_eda.py`

All files here are **derived data** and **diagnostic reports**, not raw sources.

---

##  Folder Structure
outputs/eda/
│
├── analysis/
│ ├── analysis_report.csv
│ ├── missing_rates.csv
│ ├── outlier_rates_iqr.csv
│ ├── types_consistency.csv
│ ├── psi_by_provincia.csv
│ ├── target_balance.csv
│ ├── corr_matrix.png
│ ├── missing_heatmap.png
│ ├── hist_.png
│ └── box_.png
│
├── clima_2017_por_estacion.csv
├── clima_2017_por_estacion_imputado.csv
├── clima_2017_por_perfil.csv
├── clima_2017_por_perfil_imputado.csv
│
├── cultivo_categorias_unicas.csv
├── cultivo_frecuencias.csv
├── cultivo_grupo_frecuencias.csv
├── cultivo_otros_top150.csv
├── cultivo_missing.txt
│
├── perfiles_estaciones_check.csv
├── dataset_final_2017_full.csv
└── model/
└── dataset_final_2017_model.csv


---

##  Climate Data Outputs (AEMET – 2017)

### `clima_2017_por_estacion.csv`
Daily raw climate data (2017) downloaded from AEMET, grouped by meteorological station.

### `clima_2017_por_estacion_imputado.csv`
Same dataset after **robust imputation**:
1. Median per (station, month)
2. Median per station (annual)
3. Global median fallback

### `clima_2017_por_perfil.csv`
Daily climate data projected from stations to CARBOSOL soil profiles using **nearest-station assignment (Haversine distance)**.

### `clima_2017_por_perfil_imputado.csv`
Imputed daily climate data at **profile level**, used for aggregation.

---

##  Crop Variable Engineering

### `cultivo_categorias_unicas.csv`
List of all unique raw crop/land-use descriptions extracted from CARBOSOL metadata.

### `cultivo_frecuencias.csv`
Frequency table of original (raw) crop descriptions.

### `cultivo_grupo_frecuencias.csv`
Frequency table after **advanced rule-based grouping** into robust target classes
(e.g. Cereal de invierno, Olivar, Forest-Coníferas, Hortaliza, etc.).

### `cultivo_otros_top150.csv`
Top 150 most frequent descriptions classified as `"Otros"`, used to:
- Audit classification quality
- Iteratively improve grouping rules

### `cultivo_missing.txt`
Sanity check report indicating how many profiles lack valid crop information.

---

##  Geospatial & Station Assignment

### `perfiles_estaciones_check.csv`
Validation file for soil-profile → meteorological-station assignment, including:
- Assigned station ID
- Distance (km)
- Station activity in 2017
- Rank of station used (1st closest, 2nd, etc.)

Used to ensure **every profile receives a climate signal**, even when stations are inactive.

---

##  Final Datasets

### `dataset_final_2017_full.csv`
Main **feature-engineered dataset**, containing:
- Soil profile variables
- Aggregated horizon variables (mean & median)
- Aggregated climate indicators (2017)
- Target variable (`cultivo_grupo`)

Includes all profiles, even those classified as `"Otros"`.

### `model/dataset_final_2017_model.csv`
Filtered dataset used for modeling:
- Excludes `"Otros"` class
- Clean target distribution
- Input to `02_split_modeling_smote.ipynb`

---

##  EDA & Quality Analysis (`analysis/`)

This subfolder contains **automatically generated diagnostics**:

### Core reports
- `analysis_report.csv` → executive EDA summary
- `missing_rates.csv` → missing value percentages per feature
- `outlier_rates_iqr.csv` → IQR-based outlier detection
- `types_consistency.csv` → inferred vs. actual variable types

### Stability & drift
- `psi_by_provincia.csv` → Population Stability Index by province

### Visualizations
- `missing_heatmap.png`
- `corr_matrix.png`
- Histograms and boxplots for key numeric variables
- Target balance plots

---

##  Notes

- This folder is **fully reproducible** by running `01_data_acquisition_eda.py`
- Raw datasets (CARBOSOL, AEMET) are **not modified**
- Outputs are intentionally stored for:
  - Transparency
  - Model traceability
  - Portfolio review by recruiters / reviewers

---

##  Related Notebooks

- `notebooks/01_data_acquisition_eda.py` → Data ingestion, feature engineering & EDA
- `notebooks/02_split_modeling_smote.ipynb` → Train/test split, SMOTE balancing and modeling

---

**Author:** Margarita Martínez Gallardo  
**Project:** GrowSelf – Crop Recommendation System (TFM, EOI)
