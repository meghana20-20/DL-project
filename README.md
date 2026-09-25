# AI-Driven Unified Marine & Oceanographic Platform
### Multimodal Oceanographic Hydrography, Commercial Fisheries & Molecular Biodiversity
**VTU / CMRIT Deep Learning Mini Project**

---

## 🌊 Executive Project Overview
Marine ecosystems are undergoing severe thermal and biogeochemical shocks due to anthropogenic climate change, ocean acidification, and intensive commercial harvesting. Traditional marine analytics suffer from extreme domain fragmentation.

This project delivers an **AI-Driven Unified Marine & Oceanographic Data Platform** that ingests, harmonizes, cleans, normalizes, and structures multimodal marine observations into deep learning-ready pipelines to predict:
1. **Marine Catch Tonnage (MT)** via supervised regression using Classical ML (Random Forest) and Deep Learning (PyTorch MLP).

---

## 📊 Verified Dataset Sourcing Summary (Version 4 Final)

| Modality | Source Platform | Resolution / Domain Attributes |
| :--- | :--- | :--- |
| **Commercial Catch** | **IOTC (Indian Ocean Tuna Commission)** | 2018–2023, Catch Tonnage (MT), Gear Types, Species |
| **Commercial Effort** | **IOTC** | Hook/Fishing Day totals matched at record-level |
| **Ocean Hydrography** | **INCOIS (ERDDAP)** | Expanded Argo Grid (Lat: -30 to 30, Lon: 30 to 120), Depth: 5m, Temp & Salinity |
| **Marine Biodiversity** | **IndOBIS / OBIS** | Occurrences within ±12 months and 500km radius |
| **Grid Mapping** | **IOTC Official Reference** | Exact geographic centroids for official IOTC 1°x1° and 5°x5° grids |

The final unified V4 dataset contains exactly **20,000 real records**, successfully maintaining the constraint that **1 row = 1 real IOTC catch record**, matched chronologically and spatially with hydrography data (achieving a >98% match rate within 50km for Argo data).

---

## 📁 Repository Directory Structure

```
DL PROJECT/
├── .venv/                         # Isolated Python virtual environment
├── data/
│   ├── raw/                       # Sourced multi-source raw records (CSV)
│   │   ├── argo/
│   │   ├── indobis/
│   │   └── iotc/
│   ├── processed/                 # Partitioned and normalized data
│   │   ├── marine_catch_modeling_dataset_v4.csv # The core V4 dataset
│   │   ├── v4_train.csv / val.csv / test.csv    # Chronological strict splits
│   │   ├── v4_X_train.npz / val / test          # Encoded sparse matrices
│   │   ├── v4_preprocessor.joblib               # Fitted sklearn Pipeline
│   │   ├── v4_random_forest_baseline.joblib     # Baseline RF Model
│   │   ├── v4_mlp_optimized.pt                  # Optimized PyTorch MLP
│   │   └── *.json                               # Comprehensive audit & metrics reports
│   └── figures/                   
│       └── v4_mlp_training_validation_loss.png
├── scratch/                       # Python scripts for data building, auditing, and modeling
│   ├── build_modeling_v4.py       # Constructs the V4 dataset
│   ├── preprocess_v4.py           # Feature engineering & strict scaling/encoding
│   ├── train_v4_rf.py             # Random Forest training script
│   └── train_v4_mlp.py            # PyTorch MLP tuning & training script
├── requirements.txt               # Pinned project dependencies
└── README.md                      # Complete project documentation
```

---

## 🚀 Data Processing & Feature Engineering Pipeline
* **Spatial Matching:** Uses `sklearn.neighbors.BallTree` (haversine distance) for fast, highly accurate geospatial nearest-neighbor matching across domains.
* **Missingness Indicators:** Explicit boolean flags (e.g., `has_effort_hooks`, `has_argo_temp`) are injected before median imputation to preserve informational signal regarding missing records.
* **Target Normalization:** The target, `target_catch_mt`, exhibits extreme positive skewness (Skewness > 41). The MLP dynamically applies a `log1p` transformation during training and an `expm1` inversion for strict MT evaluation.
* **Zero Leakage:** The dataset undergoes a strict chronological split (Train: ~80%, Val: ~10%, Test: ~10% by complete Year-Month periods). All scalers (`StandardScaler`) and encoders (`OneHotEncoder`) are fit strictly on the Training set.

---

## 🧠 Modeling Performance

We evaluated a classical machine learning baseline against a PyTorch Deep Learning model using a hyperparameter search grid.

### 1. Random Forest (Classical Baseline)
*   **Test RMSE:** 206.69 MT
*   **Test R²:** 0.194
*   **Test Pearson:** 0.464

### 2. PyTorch MLP (Optimized via Early Stopping Search)
*   **Best Architecture:** `[256, 128]`, Dropout: 0.0, LR: 0.001, Batch: 256
*   **Test RMSE:** 229.48 MT
*   **Test R²:** 0.006
*   **Test Pearson:** 0.187

**Key Finding:** The MLP minimizes MAE better (due to the proportional nature of the log1p loss function), making it more accurate for typical median catches. However, the Random Forest drastically outperforms the MLP on RMSE, R², and Pearson correlation, demonstrating a superior capability to handle extreme variance and rare massive commercial hauls.
