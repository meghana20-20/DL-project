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
├── figures/                       # Loss curves, residual plots, tuning plots (tracked)
├── models/                        # Tuned MLP: .pt weights, .onnx export, config (tracked)
├── scripts/                       # Python scripts for data building, auditing, and modeling
│   ├── build_modeling_v4.py       # Constructs the V4 dataset
│   ├── preprocess_v4.py           # Feature engineering & strict scaling/encoding
│   ├── train_v4_rf.py             # Random Forest training script
│   ├── train_v4_mlp.py            # PyTorch MLP (importable: FishMLP, get_dataloaders, train_mlp, ...) + original sweep
│   └── tune_v4_mlp.py             # Tuning & evaluation (random search, seeds, ablation, export)
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

---

## 🎛️ Tuning & Evaluation

All scripts resolve paths relative to the repo root (`data/`, `figures/`, `models/`). To use a different checkout or data location, set `DL_PROJECT_DIR`:

```bash
export DL_PROJECT_DIR=/path/to/DL-project   # optional; defaults to the repo root
```

Task: regression on catch tonnage (MT). The network is trained on `log1p(MT)`; metrics are reported in MT after `expm1` (predictions clipped in log space to `[0, max train log target + 1]`, i.e. at most about 11,974 MT).

### Run

```bash
python scripts/build_modeling_v4.py && python scripts/preprocess_v4.py   # build data/processed/
python scripts/train_v4_rf.py                                            # Random Forest baseline
python scripts/train_v4_mlp.py                                           # original 10-config MLP sweep
python scripts/tune_v4_mlp.py                                            # tuning & evaluation (~5 min on CPU)
python scripts/tune_v4_mlp.py --quick --out-dir /tmp/q --fig-dir /tmp/qf # tiny smoke run, writes elsewhere
```

`tune_v4_mlp.py` runs, in order: a seeded (seed 0) 40-config random search over architecture, dropout, learning rate (log-uniform 1e-4 to 3e-3), weight decay, batch size and optimizer (adam / adamw / SGD with momentum 0.9); the top 3 configs x 5 seeds; a regularization ablation on the chosen config (3 seeds: as chosen, dropout 0, weight decay 0, no early stopping at a fixed 60 epochs); then the final evaluation. Configs are ranked by **validation log-space RMSE** (the training objective). The test set is used once, for the final chosen model, and never for selection. It never overwrites the original `v4_mlp_optimized.*` files.

### Outputs

| Location | Files |
| :--- | :--- |
| `data/processed/` (gitignored) | `v4_mlp_tuning_results.csv`, `v4_mlp_tuning_seeds.csv`, `v4_mlp_ablation.csv`, `v4_mlp_tuned_results.json`, `v4_mlp_tuned_comparison.csv`, `v4_mlp_tuned_test_predictions.npy`, `v4_mlp_tuning.log` |
| `models/` (tracked) | `v4_mlp_tuned.pt` (state_dict), `v4_mlp_tuned.onnx` (dynamic batch axis), `v4_mlp_tuned_config.json` |
| `figures/` (tracked) | `v4_mlp_tuned_loss_curves.png`, `v4_mlp_tuned_predicted_vs_actual_log1p.png`, `v4_mlp_tuned_residuals_symlog.png`, `v4_mlp_tuning_top10_val_rmse.png`, `v4_mlp_tuning_val_rmse_vs_lr.png` |

The tuned model's input is the 1,573-dim preprocessed feature vector (`v4_preprocessor.joblib`) and its output is `log1p(MT)`. To load it:

```python
from scripts.train_v4_mlp import load_mlp_checkpoint, predict_mt
model = load_mlp_checkpoint("models/v4_mlp_tuned_config.json", "models/v4_mlp_tuned.pt")
```

### Results (last run, test set, MT)

Chosen config: SGD (momentum 0.9), `[128, 64]`, dropout 0.2, lr 3.29e-4, weight decay 1e-5, batch size 64 (seed 2). Over 5 seeds its validation log-RMSE is 0.962 +/- 0.012 and validation MT RMSE 296.1 +/- 9.8.

| Model | MAE | RMSE | R² | Pearson | Median AE | Top-5 rows' share of squared error |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tuned MLP | **21.8** | 222.3 | 0.067 | 0.321 | **2.34** | 93.3% |
| Original MLP (first sweep, config `[256,128,64]`, lr 1e-4) | 22.9 | 220.6 | 0.081 | 0.297 | 2.71 | 92.5% |
| Random Forest | 25.6 | **206.8** | **0.193** | **0.460** | 3.37 | 93.3% |

Tuned MLP across splits: train RMSE 77.2 / R² 0.383, validation RMSE 287.5 / R² 0.069, test RMSE 222.3 / R² 0.067.

Takeaways:
* Tuning improved typical-haul error (MAE, median AE) but not RMSE or R², which are dominated by a handful of very large hauls: the 5 largest test errors account for about 93% of the total squared error. The Random Forest remains best on RMSE, R² and Pearson.
* Ranking by log-space RMSE and by MT-space RMSE disagree (Spearman 0.27; 4 of the top 10 configs in common), so the choice of selection metric matters.
* In the ablation, early stopping clearly narrows the train/validation gap (0.14 vs 0.19 in log-MSE); the effect of dropout and weight decay is within seed noise.

---

## 🧩 Using the model

For anyone wiring the tuned MLP into an app. Files in `models/`: `v4_mlp_tuned.pt` / `.onnx` (the network), `v4_mlp_tuned_config.json` (architecture + the prediction clip), `v4_preprocessor.joblib` (fitted sklearn `ColumnTransformer`), `v4_feature_names.json` (the 1,573 encoded feature names).

**Read this first:** the model outputs **`log1p(catch in MT)`**, not tonnage, and its quality is modest: on the test set R² is about **0.07** (Pearson 0.32), and the Random Forest baseline is better (R² 0.19). It is reasonable for a typical haul (test median absolute error about 2.3 MT) but misses very large hauls badly. The UI should present predictions as **rough estimates**, not precise values.

### Raw input columns

One row = one IOTC catch record. The preprocessor needs these columns (everything else the dataset carries is ignored):

| Column | Type | Notes |
| :--- | :--- | :--- |
| `FLEET`, `FISHERY`, `FISHERY_GROUP`, `GEAR`, `SPECIES`, `SPECIES_CATEGORY` | categorical (str) | IOTC names, e.g. `FLEET="Japan"`, `GEAR="Longline (deep-freezing)"` |
| `FISHING_GROUND_CODE` | categorical (**must be `str`**) | IOTC grid code, e.g. `"5106060"` (1,484 known codes) |
| `MONTH_START` | int 1-12 | |
| `fishing_ground_lat`, `fishing_ground_lon`, `grid_resolution`, `ocean_area_iotc_km2` | float | grid-cell centroid, 1 or 5 degree cell size, ocean area; can be NaN |
| `argo_temp`, `argo_sal`, `nearest_argo_distance_km` | float, NaN if no Argo match | |
| `total_effort_hooks`, `total_effort_fdays` | float, NaN if no effort record | |
| `effort_record_count` | int | |

Five indicator flags (`has_effort_hooks`, `has_effort_fdays`, `has_argo_temp`, `has_argo_sal`, `has_argo_match`) are derived from the NaN pattern by the same step used in `scripts/preprocess_v4.py`; the example below reproduces it. `YEAR` is not a model feature.

Missing values: numeric NaNs are median-imputed and categorical NaNs get the most frequent training category. **Unseen categories** (a fleet, gear or grid code not in the training data) do **not** raise an error: the one-hot block for that column is all zeros, so the model just has no information from it. The list of known values is `preprocessor.named_transformers_["cat"].named_steps["ohe"].categories_`. Known categories are few for some columns (`SPECIES`: 3, `SPECIES_CATEGORY`: 1, so they carry almost no signal) and many for others (`FISHING_GROUND_CODE`: 1,484).

**Gotcha:** passing `FISHING_GROUND_CODE` as an integer silently matches nothing (all zeros, no warning). Always cast it to `str`.

### Example

```python
import json
import torch  # import torch BEFORE anything that loads sklearn (see scripts/train_v4_mlp.py)
import joblib
import numpy as np
import pandas as pd
from scripts.train_v4_mlp import load_mlp_checkpoint

CONFIG, WEIGHTS = "models/v4_mlp_tuned_config.json", "models/v4_mlp_tuned.pt"
preprocessor = joblib.load("models/v4_preprocessor.joblib")  # needs the sklearn version it was saved with
model = load_mlp_checkpoint(CONFIG, WEIGHTS)                 # eval mode
clip_max = json.load(open(CONFIG))["clip_max_log"]           # same upper bound used in training (~11,974 MT)

CAT = ["FLEET", "FISHERY", "FISHERY_GROUP", "GEAR", "SPECIES", "SPECIES_CATEGORY", "FISHING_GROUND_CODE"]
NUM = ["MONTH_START", "fishing_ground_lat", "fishing_ground_lon", "grid_resolution", "ocean_area_iotc_km2",
       "argo_temp", "argo_sal", "nearest_argo_distance_km", "total_effort_hooks", "total_effort_fdays",
       "effort_record_count"]


def predict_catch_mt(row: dict) -> float:
    """Rough catch estimate in metric tons for one record (omit or set None any value that is unknown)."""
    df = pd.DataFrame([row]).reindex(columns=CAT + NUM)
    df[NUM] = df[NUM].astype(float)
    df["FISHING_GROUND_CODE"] = df["FISHING_GROUND_CODE"].astype(str)
    # same feature engineering as scripts/preprocess_v4.py
    df["has_effort_hooks"] = df["total_effort_hooks"].notna().astype(int)
    df["has_effort_fdays"] = df["total_effort_fdays"].notna().astype(int)
    df["has_argo_temp"] = df["argo_temp"].notna().astype(int)
    df["has_argo_sal"] = df["argo_sal"].notna().astype(int)
    df["has_argo_match"] = df["nearest_argo_distance_km"].notna().astype(int)
    df[["total_effort_hooks", "total_effort_fdays"]] = df[["total_effort_hooks", "total_effort_fdays"]].fillna(0)

    x = torch.tensor(preprocessor.transform(df).toarray(), dtype=torch.float32)
    with torch.no_grad():
        log_pred = model(x.to(next(model.parameters()).device)).item()   # = log1p(catch MT)
    return float(np.expm1(np.clip(log_pred, 0.0, clip_max)))


# e.g. a record from data/processed/v4_test.csv:
row = pd.read_csv("data/processed/v4_test.csv", nrows=1).iloc[0].to_dict()
print(predict_catch_mt(row))
```

The ONNX file (`models/v4_mlp_tuned.onnx`, input `features` of shape `[batch, 1573]`, output `log_catch_pred`) gives the same network for non-Python runtimes; apply the same preprocessing, then `expm1` and the clip.
