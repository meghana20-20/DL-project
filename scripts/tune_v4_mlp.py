"""Tuning & evaluation for the V4 catch-tonnage MLP (regression on log1p(MT)).

Stages (all selection uses the validation set only; the test set is read once, for the final model):
  1. seeded random search over architecture/dropout/lr/weight-decay/batch-size/optimizer
  2. top-3 configs x 5 seeds
  3. regularization ablation on the chosen config (3 seeds each)
  4. final train/val/test evaluation, checkpoint + ONNX export, figures

`python scripts/tune_v4_mlp.py` runs everything; `--quick` is a tiny smoke run.
Never writes v4_mlp_optimized.*; outputs are v4_mlp_tuning_*.csv, v4_mlp_ablation.csv, v4_mlp_tuned.*.
"""
# torch must be imported before sklearn/scipy.stats (see the comment in train_v4_mlp.py).
import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_v4_mlp import (  # noqa: E402
    DEFAULT_DATA_DIR, DEFAULT_FIG_DIR, FishMLP, export_model, get_dataloaders, get_device,
    get_metrics, load_split_tensors, predict_mt, set_seed,
)

ARCHS = [[128, 64], [256, 128], [256, 128, 64], [512, 256, 128], [64, 32]]
DROPOUTS = [0.0, 0.1, 0.2, 0.3, 0.5]
WEIGHT_DECAYS = [0, 1e-5, 1e-4, 1e-3]
BATCH_SIZES = [64, 128, 256]
OPTIMIZERS = ['adam', 'adamw', 'sgd']
MAX_EPOCHS = 100
PATIENCE = 10
FIXED_EPOCHS = 60  # ablation (d): no early stopping


# ----------------------------------------------------------------------------- search space
def sample_configs(n, seed=0):
    rng = random.Random(seed)
    configs, seen = [], set()
    while len(configs) < n:
        cfg = {
            'arch': list(rng.choice(ARCHS)),
            'dropout': rng.choice(DROPOUTS),
            'lr': float(round(math.exp(rng.uniform(math.log(1e-4), math.log(3e-3))), 6)),
            'weight_decay': rng.choice(WEIGHT_DECAYS),
            'batch_size': rng.choice(BATCH_SIZES),
            'optimizer': rng.choice(OPTIMIZERS),
        }
        key = json.dumps(cfg, sort_keys=True)
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
    return configs


def build_optimizer(model, cfg):
    name = cfg.get('optimizer', 'adam')
    if name == 'adam':
        return optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    if name == 'adamw':
        return optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    if name == 'sgd':
        return optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=cfg['weight_decay'])
    raise ValueError(f"unknown optimizer {name!r}")


# ----------------------------------------------------------------------------- metrics
def mt_metrics(y_true, y_pred):
    m = get_metrics(y_true, y_pred)
    m['medae'] = float(np.median(np.abs(y_true - y_pred)))
    return m


def top_k_error_share(y_true, y_pred, k=5):
    sq = (y_true - y_pred) ** 2
    return float(np.sort(sq)[::-1][:k].sum() / sq.sum())


def _mean_loss(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            total += criterion(model(X), y).item() * X.size(0)
            n += X.size(0)
    return total / n


# ----------------------------------------------------------------------------- training
def train_run(cfg, data, seed, clip_max, device, early_stop=True, epochs=None, keep_state=False):
    """Train one config/seed. With early_stop: patience 10 on val log-MSE, restore the best epoch.
    Without: train `epochs` (default FIXED_EPOCHS) epochs and report the last one.

    Reported losses are log-space MSE computed in eval mode on the full train/val sets for the
    reported weights (so train loss is not inflated by dropout).
    """
    set_seed(seed)
    train_loader, val_loader, _, meta = get_dataloaders(batch_size=cfg['batch_size'], seed=seed, data=data)
    train_loader.generator.manual_seed(seed)
    n_epochs = (MAX_EPOCHS if early_stop else (epochs or FIXED_EPOCHS))

    model = FishMLP(data['input_dim'], cfg['arch'], cfg['dropout']).to(device)
    optimizer = build_optimizer(model, cfg)
    criterion = nn.MSELoss()

    best_val, best_state, best_epoch, bad = float('inf'), None, 0, 0
    train_curve, val_curve = [], []
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        running, n = 0.0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * X.size(0)
            n += X.size(0)
        train_curve.append(running / n)
        val_loss = _mean_loss(model, val_loader, criterion, device)
        val_curve.append(val_loss)
        if not math.isfinite(train_curve[-1]):
            break
        if early_stop:
            if val_loss < best_val:
                best_val, best_state, best_epoch, bad = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    train_time = time.time() - t0

    if early_stop:
        if best_state is None:  # never produced a finite val loss
            return {'diverged': True, 'train_time': train_time, 'epochs_run': len(val_curve),
                    'train_curve': train_curve, 'val_curve': val_curve}
        model.load_state_dict(best_state)
        reported_epoch = best_epoch
    else:
        if not math.isfinite(val_curve[-1]):
            return {'diverged': True, 'train_time': train_time, 'epochs_run': len(val_curve),
                    'train_curve': train_curve, 'val_curve': val_curve}
        reported_epoch = len(val_curve) - 1

    eval_train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=512, shuffle=False)
    train_log_mse = _mean_loss(model, eval_train_loader, criterion, device)
    val_log_mse = _mean_loss(model, val_loader, criterion, device)
    p_val = predict_mt(model, val_loader, device, clip_max=clip_max)
    out = {
        'diverged': False, 'seed': seed, 'best_epoch': reported_epoch, 'epochs_run': len(val_curve),
        'train_time': train_time, 'train_log_rmse': math.sqrt(train_log_mse),
        'val_log_rmse': math.sqrt(val_log_mse), 'train_log_mse': train_log_mse, 'val_log_mse': val_log_mse,
        'val_mt': mt_metrics(meta['y_val_mt'], p_val),
        'train_curve': train_curve, 'val_curve': val_curve,
    }
    if keep_state:
        out['state_dict'] = copy.deepcopy(model.state_dict())
    return out


def result_row(cfg, res, **extra):
    row = {'arch': str(cfg['arch']), 'dropout': cfg['dropout'], 'lr': cfg['lr'],
           'weight_decay': cfg['weight_decay'], 'batch_size': cfg['batch_size'],
           'optimizer': cfg['optimizer'], **extra}
    if res['diverged']:
        row.update(best_epoch=np.nan, epochs_run=res['epochs_run'], train_time=res['train_time'],
                   train_log_rmse=np.nan, val_log_rmse=np.inf, val_mae_mt=np.nan, val_rmse_mt=np.inf,
                   val_r2_mt=np.nan, val_pearson_mt=np.nan, val_medae_mt=np.nan)
    else:
        m = res['val_mt']
        row.update(best_epoch=res['best_epoch'], epochs_run=res['epochs_run'], train_time=res['train_time'],
                   train_log_rmse=res['train_log_rmse'], val_log_rmse=res['val_log_rmse'],
                   val_mae_mt=m['mae'], val_rmse_mt=m['rmse'], val_r2_mt=m['r2'],
                   val_pearson_mt=m['pearson_corr'], val_medae_mt=m['medae'])
    return row


# ----------------------------------------------------------------------------- stages
def run_search(configs, data, clip_max, device, out_dir, log):
    rows, results = [], []
    for i, cfg in enumerate(configs):
        res = train_run(cfg, data, seed=0, clip_max=clip_max, device=device)
        results.append(res)
        rows.append(result_row(cfg, res, config_id=i, seed=0))
        r = rows[-1]
        log(f"[search {i+1}/{len(configs)}] {cfg['optimizer']:5s} {str(cfg['arch']):15s} do={cfg['dropout']} "
            f"lr={cfg['lr']:.5f} wd={cfg['weight_decay']} bs={cfg['batch_size']} -> "
            f"val log-RMSE={r['val_log_rmse']:.4f} MT-RMSE={r['val_rmse_mt']:.1f} ep={r['best_epoch']} {r['train_time']:.1f}s")
    df = pd.DataFrame(rows)
    df['rank_log_rmse'] = df['val_log_rmse'].rank(method='min').astype(int)
    df['rank_mt_rmse'] = df['val_rmse_mt'].rank(method='min').astype(int)
    df = df.sort_values('rank_log_rmse').reset_index(drop=True)
    df.to_csv(out_dir / 'v4_mlp_tuning_results.csv', index=False)
    return df, results


def run_seeds(top_ids, configs, data, clip_max, device, out_dir, log, seeds):
    rows, runs = [], {}
    for cid in top_ids:
        cfg = configs[cid]
        for s in seeds:
            res = train_run(cfg, data, seed=s, clip_max=clip_max, device=device, keep_state=True)
            runs[(cid, s)] = res
            rows.append(result_row(cfg, res, config_id=cid, seed=s))
            r = rows[-1]
            log(f"[seeds] config {cid} seed {s}: val log-RMSE={r['val_log_rmse']:.4f} MT-RMSE={r['val_rmse_mt']:.1f} ep={r['best_epoch']}")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'v4_mlp_tuning_seeds.csv', index=False)
    return df, runs


def run_ablation(cfg, data, clip_max, device, out_dir, log, seeds):
    variants = {
        'a_as_chosen': (cfg, True),
        'b_dropout_0': ({**cfg, 'dropout': 0.0}, True),
        'c_weight_decay_0': ({**cfg, 'weight_decay': 0}, True),
        'd_no_early_stopping_60ep': (cfg, False),
    }
    rows = []
    for name, (vcfg, early) in variants.items():
        for s in seeds:
            res = train_run(vcfg, data, seed=s, clip_max=clip_max, device=device, early_stop=early)
            if res['diverged']:
                log(f"[ablation] {name} seed {s}: DIVERGED")
                continue
            rows.append({'variant': name, 'seed': s, 'dropout': vcfg['dropout'], 'weight_decay': vcfg['weight_decay'],
                         'early_stopping': early, 'reported_epoch': res['best_epoch'], 'epochs_run': res['epochs_run'],
                         'train_log_mse': res['train_log_mse'], 'val_log_mse': res['val_log_mse'],
                         'gap_log_mse': res['val_log_mse'] - res['train_log_mse'],
                         'val_log_rmse': res['val_log_rmse'], 'val_rmse_mt': res['val_mt']['rmse'],
                         'val_mae_mt': res['val_mt']['mae']})
            log(f"[ablation] {name} seed {s}: train={res['train_log_mse']:.3f} val={res['val_log_mse']:.3f} "
                f"MT-RMSE={res['val_mt']['rmse']:.1f} ep={res['best_epoch']}")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'v4_mlp_ablation.csv', index=False)
    return df


# ----------------------------------------------------------------------------- figures
SERIES = {'adam': ('#2a78d6', 'o'), 'adamw': ('#eb6834', 's'), 'sgd': ('#1baf7a', '^')}


def make_figures(fig_dir, search_df, chosen_res, y_test, p_test, plt):
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. loss curves of the chosen model
    plt.figure(figsize=(10, 6))
    plt.plot(chosen_res['train_curve'], label='Training loss')
    plt.plot(chosen_res['val_curve'], label='Validation loss')
    plt.axvline(chosen_res['best_epoch'], color='gray', linestyle=':', label=f"Best epoch ({chosen_res['best_epoch']})")
    plt.title('Tuned MLP Training & Validation Loss (Log Space)')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss (log1p MT)')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(fig_dir / 'v4_mlp_tuned_loss_curves.png'); plt.close()

    # 2. predicted vs actual on log1p axes
    plt.figure(figsize=(10, 6))
    lp, la = np.log1p(p_test), np.log1p(y_test)
    plt.scatter(lp, la, alpha=0.4, s=12)
    lims = [0, max(lp.max(), la.max()) * 1.02]
    plt.plot(lims, lims, color='red', linestyle='--', label='y = x')
    plt.title('Tuned MLP Predicted vs Actual Catch (Test Set, log1p axes)')
    plt.xlabel('log1p(Predicted Catch, MT)'); plt.ylabel('log1p(Actual Catch, MT)')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(fig_dir / 'v4_mlp_tuned_predicted_vs_actual_log1p.png'); plt.close()

    # 3. residuals with a symmetric-log y axis (x symlog too: predictions are heavily right-skewed)
    plt.figure(figsize=(10, 6))
    plt.scatter(p_test, y_test - p_test, alpha=0.4, s=12)
    plt.axhline(0, color='red', linestyle='--', label='Zero residual')
    plt.yscale('symlog', linthresh=10); plt.xscale('symlog', linthresh=1)
    plt.title('Tuned MLP Residuals vs Predicted (Test Set, symlog axes)')
    plt.xlabel('Predicted Catch (MT)'); plt.ylabel('Residual (Actual - Predicted, MT)')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(fig_dir / 'v4_mlp_tuned_residuals_symlog.png'); plt.close()

    # 4. top-10 configs by validation RMSE (log space = selection metric, with MT RMSE beside it)
    top = search_df.head(10).iloc[::-1]
    labels = [f"{r.optimizer} {r.arch} do={r.dropout} lr={r.lr:.4f} wd={r.weight_decay:g} bs={r.batch_size}" for r in top.itertuples()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    axes[0].barh(range(len(top)), top['val_log_rmse'], color='#2a78d6')
    axes[0].set_title('Val RMSE, log space (selection metric)'); axes[0].set_xlabel('RMSE (log1p MT)')
    axes[1].barh(range(len(top)), top['val_rmse_mt'], color='#eb6834')
    axes[1].set_title('Val RMSE, tonnage'); axes[1].set_xlabel('RMSE (MT)')
    axes[0].set_yticks(range(len(top))); axes[0].set_yticklabels(labels, fontsize=7)
    for ax in axes:
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    fig.suptitle('Top 10 random-search configs (best at top)')
    fig.tight_layout(); fig.savefig(fig_dir / 'v4_mlp_tuning_top10_val_rmse.png'); plt.close(fig)

    # 5. val RMSE vs learning rate, by optimizer
    plt.figure(figsize=(10, 6))
    finite = search_df[np.isfinite(search_df['val_log_rmse'])]
    for opt, (color, marker) in SERIES.items():
        sub = finite[finite['optimizer'] == opt]
        plt.scatter(sub['lr'], sub['val_log_rmse'], color=color, marker=marker, s=40, alpha=0.85,
                    edgecolors='white', linewidths=0.8, label=opt)
    plt.xscale('log')
    plt.title('Random Search: Validation RMSE (log space) vs Learning Rate')
    plt.xlabel('Learning rate (log scale)'); plt.ylabel('Val RMSE (log1p MT)')
    plt.legend(title='Optimizer'); plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(fig_dir / 'v4_mlp_tuning_val_rmse_vs_lr.png'); plt.close()


# ----------------------------------------------------------------------------- main
def main(data_dir=DEFAULT_DATA_DIR, out_dir=None, fig_dir=DEFAULT_FIG_DIR, quick=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data_dir = Path(data_dir)
    out_dir = Path(out_dir) if out_dir else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(fig_dir)
    t_start = time.time()
    log_file = open(out_dir / 'v4_mlp_tuning.log', 'a')

    def log(msg):
        line = f"[{time.time()-t_start:7.1f}s] {msg}"
        print(line, flush=True)
        log_file.write(line + '\n'); log_file.flush()

    device = get_device()
    data = load_split_tensors(data_dir)
    clip_max = float(data['tensors']['train'][1].max()) + 1.0
    n_search, n_seeds, n_abl = (4, 2, 2) if quick else (40, 5, 3)
    log(f"device={device} input_dim={data['input_dim']} clip_max(log)={clip_max:.3f} (= {math.expm1(clip_max):.0f} MT) quick={quick}")

    # --- 1. random search (seed 0)
    configs = sample_configs(n_search, seed=0)
    search_df, _ = run_search(configs, data, clip_max, device, out_dir, log)
    rho = search_df[['rank_log_rmse', 'rank_mt_rmse']].corr(method='spearman').iloc[0, 1]
    top10_log = set(search_df.nsmallest(10, 'rank_log_rmse')['config_id'])
    top10_mt = set(search_df.nsmallest(10, 'rank_mt_rmse')['config_id'])
    log(f"ranking agreement log-RMSE vs MT-RMSE: spearman={rho:.3f}, top-10 overlap={len(top10_log & top10_mt)}/10")

    # --- 2. seed variation on top 3
    top_ids = [int(c) for c in search_df.head(3)['config_id']]
    seed_df, runs = run_seeds(top_ids, configs, data, clip_max, device, out_dir, log, seeds=list(range(n_seeds)))
    agg = seed_df.groupby('config_id').agg(
        val_log_rmse_mean=('val_log_rmse', 'mean'), val_log_rmse_std=('val_log_rmse', 'std'),
        val_rmse_mt_mean=('val_rmse_mt', 'mean'), val_rmse_mt_std=('val_rmse_mt', 'std')).loc[top_ids]
    log("seed summary (mean/std over seeds):\n" + agg.round(4).to_string())
    chosen_id = int(agg['val_log_rmse_mean'].idxmin())
    chosen_cfg = configs[chosen_id]
    best_seed = int(seed_df[seed_df.config_id == chosen_id].sort_values('val_log_rmse').iloc[0]['seed'])
    chosen_res = runs[(chosen_id, best_seed)]
    log(f"chosen config_id={chosen_id}: {chosen_cfg}; saving seed {best_seed} (lowest val log-RMSE of its seeds)")

    # --- 3. ablation
    abl_df = run_ablation(chosen_cfg, data, clip_max, device, out_dir, log, seeds=list(range(n_abl)))
    log("ablation summary (mean over seeds):\n" + abl_df.groupby('variant')[
        ['train_log_mse', 'val_log_mse', 'gap_log_mse', 'val_rmse_mt']].agg(['mean', 'std']).round(3).to_string())

    # --- 4. final evaluation: the only place the test set is touched
    model = FishMLP(data['input_dim'], chosen_cfg['arch'], chosen_cfg['dropout']).to(device)
    model.load_state_dict(chosen_res['state_dict'])
    _, val_loader, test_loader, meta = get_dataloaders(batch_size=256, data=data)
    train_eval = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*data['tensors']['train']), batch_size=256, shuffle=False)
    preds = {'train': predict_mt(model, train_eval, device, clip_max=clip_max),
             'val': predict_mt(model, val_loader, device, clip_max=clip_max),
             'test': predict_mt(model, test_loader, device, clip_max=clip_max)}
    ys = {'train': meta['y_train_mt'], 'val': meta['y_val_mt'], 'test': meta['y_test_mt']}
    final = {s: mt_metrics(ys[s], preds[s]) for s in ('train', 'val', 'test')}
    final['test']['top5_sq_error_share'] = top_k_error_share(ys['test'], preds['test'])
    with open(out_dir / 'v4_mlp_tuned_results.json', 'w') as f:
        json.dump(final, f, indent=4)
    np.save(out_dir / 'v4_mlp_tuned_test_predictions.npy', preds['test'])

    export_model(model, data['input_dim'], out_dir, name='v4_mlp_tuned')  # writes v4_mlp_tuned.pt and .onnx
    with open(out_dir / 'v4_mlp_tuned_config.json', 'w') as f:
        json.dump({**chosen_cfg, 'seed': best_seed, 'config_id': chosen_id, 'clip_max_log': clip_max,
                   'val_log_rmse': chosen_res['val_log_rmse'], 'best_epoch': chosen_res['best_epoch']}, f, indent=4)

    # comparison with RF and the original MLP (their saved test predictions; used for reporting only)
    rows = {'tuned MLP': preds['test']}
    for name, fn in (('Random Forest', 'v4_rf_test_predictions.npy'), ('original MLP (hemas config)', 'v4_mlp_test_predictions.npy')):
        if (data_dir / fn).exists():
            rows[name] = np.load(data_dir / fn)
    comp = []
    for name, p in rows.items():
        m = mt_metrics(ys['test'], p)
        comp.append({'model': name, 'test_mae': m['mae'], 'test_rmse': m['rmse'], 'test_r2': m['r2'],
                     'test_pearson': m['pearson_corr'], 'test_medae': m['medae'],
                     'top5_sq_error_share': top_k_error_share(ys['test'], p)})
    comp_df = pd.DataFrame(comp)
    comp_df.to_csv(out_dir / 'v4_mlp_tuned_comparison.csv', index=False)
    log("final tuned model:\n" + pd.DataFrame(final).T.round(3).to_string())
    log("test comparison:\n" + comp_df.round(3).to_string(index=False))

    make_figures(fig_dir, search_df, chosen_res, ys['test'], preds['test'], plt)
    log(f"done in {time.time()-t_start:.0f}s")
    log_file.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='V4 MLP tuning & evaluation')
    ap.add_argument('--data-dir', default=str(DEFAULT_DATA_DIR), help='preprocessed v4_* arrays (input)')
    ap.add_argument('--out-dir', default=None, help='where CSV/model outputs go (default: --data-dir)')
    ap.add_argument('--fig-dir', default=str(DEFAULT_FIG_DIR))
    ap.add_argument('--quick', action='store_true', help='tiny smoke run (4 configs, 2 seeds, 2 ablation seeds)')
    a = ap.parse_args()
    main(a.data_dir, a.out_dir, a.fig_dir, a.quick)
