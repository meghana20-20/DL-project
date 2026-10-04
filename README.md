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
