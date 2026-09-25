import os
import json
import time
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import copy
import random

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

base_dir = r"C:\Users\hemas\OneDrive\Desktop\dl project\DL-project"
out_dir = os.path.join(base_dir, "data", "processed")
fig_dir = os.path.join(base_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

# 1. Load Data
print("Loading V4 data...")
X_train_sp = sparse.load_npz(os.path.join(out_dir, "v4_X_train.npz"))
X_val_sp = sparse.load_npz(os.path.join(out_dir, "v4_X_val.npz"))
X_test_sp = sparse.load_npz(os.path.join(out_dir, "v4_X_test.npz"))

# We use log targets for training
y_train_log = np.load(os.path.join(out_dir, "v4_y_log_train.npy"))
y_val_log = np.load(os.path.join(out_dir, "v4_y_log_val.npy"))
y_test_log = np.load(os.path.join(out_dir, "v4_y_log_test.npy"))

# And original targets for evaluation
y_train_mt = np.load(os.path.join(out_dir, "v4_y_train.npy"))
y_val_mt = np.load(os.path.join(out_dir, "v4_y_val.npy"))
y_test_mt = np.load(os.path.join(out_dir, "v4_y_test.npy"))

input_dim = X_train_sp.shape[1]

# Convert to dense tensors
X_train_t = torch.FloatTensor(X_train_sp.toarray())
X_val_t = torch.FloatTensor(X_val_sp.toarray())
X_test_t = torch.FloatTensor(X_test_sp.toarray())

y_train_t = torch.FloatTensor(y_train_log).unsqueeze(1)
y_val_t = torch.FloatTensor(y_val_log).unsqueeze(1)
y_test_t = torch.FloatTensor(y_test_log).unsqueeze(1)

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

def train_mlp(config, is_baseline=False):
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    val_dataset = TensorDataset(X_val_t, y_val_t)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

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
    
    model.eval()
    val_preds_log = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            val_preds_log.append(outputs.cpu().numpy())
            
    val_preds_log = np.vstack(val_preds_log).flatten()
    val_preds_mt = np.expm1(val_preds_log)
    
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

# --- 1. BASELINE ---
print("Training Baseline MLP...")
baseline_config = {
    'arch': [128, 64],
    'dropout': 0.0,
    'lr': 0.001,
    'batch_size': 128,
    'weight_decay': 0
}
baseline_res = train_mlp(baseline_config, is_baseline=True)
print(f"Baseline Val RMSE: {baseline_res['val_metrics']['rmse']:.4f}")

# --- 2. HYPERPARAMETER SEARCH ---
configs = [
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

results = []
best_rmse = float('inf')
best_res = None
best_config = None

print("Starting Hyperparameter Search...")
for i, cfg in enumerate(configs):
    print(f"Config {i+1}/{len(configs)}: {cfg}")
    res = train_mlp(cfg)
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

pd.DataFrame(results).to_csv(os.path.join(out_dir, "v4_mlp_hyperparameter_results.csv"), index=False)

print(f"Best RMSE: {best_rmse:.4f} with config {best_config}")

# --- FINAL OPTIMIZED MLP ---
# Save the model
best_model = best_res['model']
best_model.load_state_dict(best_res['state_dict'])
torch.save(best_model.state_dict(), os.path.join(out_dir, "v4_mlp_optimized.pt"))

with open(os.path.join(out_dir, "v4_mlp_best_config.json"), "w") as f:
    json.dump(best_config, f, indent=4)

# Evaluate on Train, Val, Test
best_model.eval()

def predict_mt(loader):
    preds_log = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            preds_log.append(best_model(X_batch).cpu().numpy())
    preds_log = np.vstack(preds_log).flatten()
    return np.expm1(preds_log)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=False)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=256, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=256, shuffle=False)

p_train = predict_mt(train_loader)
p_val = predict_mt(val_loader)
p_test = predict_mt(test_loader)

opt_results = {
    "train": get_metrics(y_train_mt, p_train),
    "val": get_metrics(y_val_mt, p_val),
    "test": get_metrics(y_test_mt, p_test)
}

with open(os.path.join(out_dir, "v4_mlp_optimized_results.json"), "w") as f:
    json.dump(opt_results, f, indent=4)

np.save(os.path.join(out_dir, "v4_mlp_val_predictions.npy"), p_val)
np.save(os.path.join(out_dir, "v4_mlp_test_predictions.npy"), p_test)

# --- PLOT LOSS CURVE ---
plt.figure(figsize=(10, 6))
plt.plot(best_res['train_losses'], label='Training Loss')
plt.plot(best_res['val_losses'], label='Validation Loss')
plt.title('V4 Optimized MLP Training & Validation Loss (Log Space)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(fig_dir, "v4_mlp_training_validation_loss.png"))
plt.close()

# --- COMPARE WITH RF ---
with open(os.path.join(out_dir, "v4_random_forest_results.json"), "r") as f:
    rf_results = json.load(f)

compare = {
    "validation": {
        "rf_mae": rf_results['val']['mae'],
        "mlp_mae": opt_results['val']['mae'],
        "rf_rmse": rf_results['val']['rmse'],
        "mlp_rmse": opt_results['val']['rmse'],
        "rf_r2": rf_results['val']['r2'],
        "mlp_r2": opt_results['val']['r2'],
        "rf_pearson": rf_results['val']['pearson_corr'],
        "mlp_pearson": opt_results['val']['pearson_corr']
    },
    "test": {
        "rf_mae": rf_results['test']['mae'],
        "mlp_mae": opt_results['test']['mae'],
        "rf_rmse": rf_results['test']['rmse'],
        "mlp_rmse": opt_results['test']['rmse'],
        "rf_r2": rf_results['test']['r2'],
        "mlp_r2": opt_results['test']['r2'],
        "rf_pearson": rf_results['test']['pearson_corr'],
        "mlp_pearson": opt_results['test']['pearson_corr']
    }
}

with open(os.path.join(out_dir, "v4_rf_vs_mlp_results.json"), "w") as f:
    json.dump(compare, f, indent=4)

print("Done. V4 MLP complete.")
