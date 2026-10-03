from pathlib import Path
import numpy as np
import os
import json
import time
import joblib
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore")

# Repo root by default; override with DL_PROJECT_DIR to point at a different checkout/data location.
base_dir = os.environ.get("DL_PROJECT_DIR") or str(Path(__file__).resolve().parents[1])
out_dir = os.path.join(base_dir, "data", "processed")

print("1. Loading V4 sparse matrices and targets...")
X_train = sparse.load_npz(os.path.join(out_dir, "v4_X_train.npz"))
X_val = sparse.load_npz(os.path.join(out_dir, "v4_X_val.npz"))
X_test = sparse.load_npz(os.path.join(out_dir, "v4_X_test.npz"))

y_train = np.load(os.path.join(out_dir, "v4_y_train.npy"))
y_val = np.load(os.path.join(out_dir, "v4_y_val.npy"))
y_test = np.load(os.path.join(out_dir, "v4_y_test.npy"))

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

print("2. Initializing and training Random Forest Baseline...")
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

start_time = time.time()
rf.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"Training completed in {train_time:.2f} seconds.")

print("3. Generating Predictions...")
p_train = rf.predict(X_train)
p_val = rf.predict(X_val)
p_test = rf.predict(X_test)

print("4. Calculating Metrics...")
def get_metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # Handle pearson correlation for constant predictions or true arrays
    try:
        pearson_corr, _ = pearsonr(y_true, y_pred)
    except Exception:
        pearson_corr = np.nan
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "pearson_corr": float(pearson_corr),
        "pred_mean": float(np.mean(y_pred)),
        "pred_std": float(np.std(y_pred))
    }

results = {
    "training_time_seconds": float(train_time),
    "train": get_metrics(y_train, p_train, "Train"),
    "val": get_metrics(y_val, p_val, "Validation"),
    "test": get_metrics(y_test, p_test, "Test")
}

print(json.dumps(results, indent=4))

print("5. Saving model, predictions, and results...")
joblib.dump(rf, os.path.join(out_dir, "v4_random_forest_baseline.joblib"))

np.save(os.path.join(out_dir, "v4_rf_val_predictions.npy"), p_val)
np.save(os.path.join(out_dir, "v4_rf_test_predictions.npy"), p_test)

with open(os.path.join(out_dir, "v4_random_forest_results.json"), "w") as f:
    json.dump(results, f, indent=4)

print("Done. V4 Random Forest Baseline complete.")
