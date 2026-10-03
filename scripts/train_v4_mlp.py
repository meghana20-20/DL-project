"""V4 marine-catch MLP (regression on catch tonnage, trained on log1p(MT)).

Importable building blocks (nothing runs on import):
    set_seed, FishMLP, get_metrics, get_dataloaders, train_mlp, predict_mt,
    load_mlp_checkpoint, export_model

`python scripts/train_v4_mlp.py` reproduces the original baseline + 10-config
sweep + final evaluation.
"""
import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
# Keep torch imported BEFORE sklearn/scipy.stats: on macOS (conda) the reverse order segfaults
# silently in Adam's optimizer step (native OpenMP load-order clash). Don't let an import sorter reorder these.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Repo root by default; override with DL_PROJECT_DIR to point at a different checkout/data location.
BASE_DIR = Path(os.environ.get("DL_PROJECT_DIR") or Path(__file__).resolve().parents[1])
DEFAULT_DATA_DIR = BASE_DIR / "data" / "processed"
DEFAULT_FIG_DIR = BASE_DIR / "figures"


# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# MLP Model definition
class FishMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super(FishMLP, self).__init__()
        layers = []
        current_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(current_dim, h))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = h
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    try:
        pearson_corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
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


def load_split_tensors(data_dir=DEFAULT_DATA_DIR):
    """Load the preprocessed V4 arrays from data_dir into dense tensors (log-space targets)
    plus the original-scale (MT) targets for evaluation."""
    data_dir = Path(data_dir)
    out = {"tensors": {}, "y_mt": {}}
    for split in ("train", "val", "test"):
        X = sparse.load_npz(data_dir / f"v4_X_{split}.npz")
        y_log = np.load(data_dir / f"v4_y_log_{split}.npy")
        out["tensors"][split] = (
            torch.FloatTensor(X.toarray()),
            torch.FloatTensor(y_log).unsqueeze(1),
        )
        out["y_mt"][split] = np.load(data_dir / f"v4_y_{split}.npy")
    out["input_dim"] = out["tensors"]["train"][0].shape[1]
    return out


def get_dataloaders(data_dir=DEFAULT_DATA_DIR, batch_size=128, seed=42, data=None):
    """Build train/val/test DataLoaders from the preprocessed V4 arrays.

    Returns (train_loader, val_loader, test_loader, meta). meta has input_dim and the
    original-scale targets (y_train_mt / y_val_mt / y_test_mt). Pass `data` (the result
    of load_split_tensors) to avoid re-reading from disk when building loaders repeatedly.
    The train loader shuffles with its own seeded generator.
    """
    if data is None:
        data = load_split_tensors(data_dir)
    t = data["tensors"]
    train_loader = DataLoader(TensorDataset(*t["train"]), batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(TensorDataset(*t["val"]), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(*t["test"]), batch_size=batch_size, shuffle=False)
    meta = {
        "input_dim": data["input_dim"],
        "y_train_mt": data["y_mt"]["train"],
        "y_val_mt": data["y_mt"]["val"],
        "y_test_mt": data["y_mt"]["test"],
    }
    return train_loader, val_loader, test_loader, meta


def predict_mt(model, loader, device=None):
    """Predict catch tonnage (MT) for every row in loader (undoes the log1p target transform)."""
    device = device or next(model.parameters()).device
    model.eval()
    preds_log = []
    with torch.no_grad():
        for X_batch, _ in loader:
            preds_log.append(model(X_batch.to(device)).cpu().numpy())
    return np.expm1(np.vstack(preds_log).flatten())


def train_mlp(config, train_loader, val_loader, y_val_mt=None, is_baseline=False, device=None):
    """Train FishMLP with early stopping (patience 10) and restore the best-val-loss weights.

    config keys: arch, dropout, lr, weight_decay (batch_size is baked into the loaders);
    optional: seed (default 42). Validation metrics are in MT; if y_val_mt is not given it is
    reconstructed from the log-space validation targets in val_loader.
    """
    device = device or get_device()
    # Re-seed at the start of every run so each config is independently reproducible.
    seed = config.get('seed', 42)
    set_seed(seed)
    if train_loader.generator is not None:
        train_loader.generator.manual_seed(seed)

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    input_dim = train_dataset.tensors[0].shape[1]
    if y_val_mt is None:
        y_val_mt = np.expm1(val_dataset.tensors[1].numpy().flatten())

    model = FishMLP(input_dim, config['arch'], config['dropout']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = nn.MSELoss()

    epochs = 50 if is_baseline else 100
    patience = 10

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    train_losses = []
    val_losses = []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_dataset)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if not is_baseline and patience_counter >= patience:
            break

    train_time = time.time() - start_time

    # Evaluate best model on val (in MT)
    if not is_baseline:
        model.load_state_dict(best_model_state)

    val_preds_mt = predict_mt(model, val_loader, device)
    val_metrics = get_metrics(y_val_mt, val_preds_mt)

    return {
        'model': model,
        'state_dict': best_model_state if not is_baseline else model.state_dict(),
        'val_metrics': val_metrics,
        'train_time': train_time,
        'best_epoch': best_epoch,
        'train_losses': train_losses,
        'val_losses': val_losses
    }


def load_mlp_checkpoint(config_path=None, weights_path=None, input_dim=None, device=None):
    """Rebuild FishMLP from the saved best config and load its weights; returns the model in eval mode.

    Defaults to <repo>/data/processed/v4_mlp_best_config.json and v4_mlp_optimized.pt. The config
    does not store input_dim, so it is inferred from the first Linear layer's weights unless given.
    """
    config_path = Path(config_path) if config_path else DEFAULT_DATA_DIR / "v4_mlp_best_config.json"
    weights_path = Path(weights_path) if weights_path else DEFAULT_DATA_DIR / "v4_mlp_optimized.pt"
    device = device or get_device()

    with open(config_path) as f:
        config = json.load(f)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if input_dim is None:
        input_dim = state_dict["network.0.weight"].shape[1]

    model = FishMLP(input_dim, config['arch'], config['dropout']).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_model(model, input_dim, out_dir, name="v4_mlp_optimized"):
    """Save <name>.pt (state_dict) and <name>.onnx (dummy input (1, input_dim), dynamic batch axis).

    The network outputs log1p(MT); apply expm1 to the output to get tonnage.
    Returns {"pt": path, "onnx": path}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / f"{name}.pt"
    onnx_path = out_dir / f"{name}.onnx"

    original_device = next(model.parameters()).device
    was_training = model.training
    model = model.cpu().eval()
    try:
        torch.save(model.state_dict(), pt_path)
        dummy = torch.zeros(1, input_dim, dtype=torch.float32)
        export_kwargs = dict(
            input_names=["features"],
            output_names=["log_catch_pred"],
            dynamic_axes={"features": {0: "batch"}, "log_catch_pred": {0: "batch"}},
            opset_version=17,
        )
        try:
            # Legacy TorchScript exporter; newer torch defaults to dynamo, which needs onnxscript.
            torch.onnx.export(model, dummy, str(onnx_path), dynamo=False, **export_kwargs)
        except TypeError:  # torch < 2.5 has no `dynamo` argument and always uses the legacy exporter
            torch.onnx.export(model, dummy, str(onnx_path), **export_kwargs)
    finally:
        model.to(original_device).train(was_training)
    return {"pt": pt_path, "onnx": onnx_path}


# Original sweep: 10 hand-picked configs
BASELINE_CONFIG = {
    'arch': [128, 64],
    'dropout': 0.0,
    'lr': 0.001,
    'batch_size': 128,
    'weight_decay': 0
}

SWEEP_CONFIGS = [
    {'arch': [128, 64], 'dropout': 0.0, 'lr': 0.001, 'batch_size': 128, 'weight_decay': 0},
    {'arch': [256, 128, 64], 'dropout': 0.1, 'lr': 0.001, 'batch_size': 128, 'weight_decay': 1e-5},
    {'arch': [256, 128], 'dropout': 0.2, 'lr': 0.0005, 'batch_size': 64, 'weight_decay': 1e-4},
    {'arch': [512, 256, 128], 'dropout': 0.3, 'lr': 0.0001, 'batch_size': 256, 'weight_decay': 0},
    {'arch': [128, 64], 'dropout': 0.1, 'lr': 0.0005, 'batch_size': 64, 'weight_decay': 0},
    {'arch': [256, 128, 64], 'dropout': 0.2, 'lr': 0.0001, 'batch_size': 128, 'weight_decay': 1e-4},
    {'arch': [256, 128], 'dropout': 0.0, 'lr': 0.001, 'batch_size': 256, 'weight_decay': 1e-5},
    {'arch': [512, 256, 128], 'dropout': 0.1, 'lr': 0.0005, 'batch_size': 128, 'weight_decay': 1e-5},
    {'arch': [128, 64], 'dropout': 0.3, 'lr': 0.001, 'batch_size': 256, 'weight_decay': 1e-4},
    {'arch': [256, 128, 64], 'dropout': 0.0, 'lr': 0.0001, 'batch_size': 64, 'weight_decay': 0},
]


def main(data_dir=DEFAULT_DATA_DIR, fig_dir=DEFAULT_FIG_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_dir, fig_dir = Path(data_dir), Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    print("Loading V4 data...")
    data = load_split_tensors(data_dir)
    y_val_mt = data["y_mt"]["val"]

    def loaders_for(cfg):
        return get_dataloaders(data_dir, cfg['batch_size'], data=data)

    # --- 1. BASELINE ---
    print("Training Baseline MLP...")
    train_loader, val_loader, _, _ = loaders_for(BASELINE_CONFIG)
    baseline_res = train_mlp(BASELINE_CONFIG, train_loader, val_loader, y_val_mt, is_baseline=True, device=device)
    print(f"Baseline Val RMSE: {baseline_res['val_metrics']['rmse']:.4f}")

    # --- 2. HYPERPARAMETER SEARCH ---
    results = []
    best_rmse = float('inf')
    best_res = None
    best_config = None

    print("Starting Hyperparameter Search...")
    for i, cfg in enumerate(SWEEP_CONFIGS):
        print(f"Config {i+1}/{len(SWEEP_CONFIGS)}: {cfg}")
        train_loader, val_loader, _, _ = loaders_for(cfg)
        res = train_mlp(cfg, train_loader, val_loader, y_val_mt, device=device)
        metrics = res['val_metrics']

        results.append({
            'architecture': str(cfg['arch']),
            'dropout': cfg['dropout'],
            'learning_rate': cfg['lr'],
            'batch_size': cfg['batch_size'],
            'weight_decay': cfg['weight_decay'],
            'best_epoch': res['best_epoch'],
            'training_time': res['train_time'],
            'validation_MAE': metrics['mae'],
            'validation_RMSE': metrics['rmse'],
            'validation_R2': metrics['r2']
        })

        if metrics['rmse'] < best_rmse:
            best_rmse = metrics['rmse']
            best_res = res
            best_config = cfg

    pd.DataFrame(results).to_csv(data_dir / "v4_mlp_hyperparameter_results.csv", index=False)

    print(f"Best RMSE: {best_rmse:.4f} with config {best_config}")

    # --- FINAL OPTIMIZED MLP ---
    best_model = best_res['model']
    best_model.load_state_dict(best_res['state_dict'])
    torch.save(best_model.state_dict(), data_dir / "v4_mlp_optimized.pt")

    with open(data_dir / "v4_mlp_best_config.json", "w") as f:
        json.dump(best_config, f, indent=4)

    # Evaluate on Train, Val, Test
    train_loader, val_loader, test_loader, meta = get_dataloaders(data_dir, 256, data=data)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256, shuffle=False)

    p_train = predict_mt(best_model, train_eval_loader, device)
    p_val = predict_mt(best_model, val_loader, device)
    p_test = predict_mt(best_model, test_loader, device)

    opt_results = {
        "train": get_metrics(meta["y_train_mt"], p_train),
        "val": get_metrics(meta["y_val_mt"], p_val),
        "test": get_metrics(meta["y_test_mt"], p_test)
    }

    with open(data_dir / "v4_mlp_optimized_results.json", "w") as f:
        json.dump(opt_results, f, indent=4)

    np.save(data_dir / "v4_mlp_val_predictions.npy", p_val)
    np.save(data_dir / "v4_mlp_test_predictions.npy", p_test)

    # --- PLOT LOSS CURVE ---
    plt.figure(figsize=(10, 6))
    plt.plot(best_res['train_losses'], label='Training Loss')
    plt.plot(best_res['val_losses'], label='Validation Loss')
    plt.title('V4 Optimized MLP Training & Validation Loss (Log Space)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(fig_dir / "v4_mlp_training_validation_loss.png")
    plt.close()

    # --- COMPARE WITH RF ---
    with open(data_dir / "v4_random_forest_results.json", "r") as f:
        rf_results = json.load(f)

    def compare_row(split):
        return {
            "rf_mae": rf_results[split]['mae'],
            "mlp_mae": opt_results[split]['mae'],
            "rf_rmse": rf_results[split]['rmse'],
            "mlp_rmse": opt_results[split]['rmse'],
            "rf_r2": rf_results[split]['r2'],
            "mlp_r2": opt_results[split]['r2'],
            "rf_pearson": rf_results[split]['pearson_corr'],
            "mlp_pearson": opt_results[split]['pearson_corr']
        }

    compare = {"validation": compare_row("val"), "test": compare_row("test")}

    with open(data_dir / "v4_rf_vs_mlp_results.json", "w") as f:
        json.dump(compare, f, indent=4)

    print("Done. V4 MLP complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V4 MLP baseline + hyperparameter sweep")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="Directory with the preprocessed v4_* arrays (default: <repo>/data/processed)")
    parser.add_argument("--fig-dir", default=str(DEFAULT_FIG_DIR), help="Where to save the loss-curve figure")
    args = parser.parse_args()
    main(args.data_dir, args.fig_dir)
