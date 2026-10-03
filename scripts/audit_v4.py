from pathlib import Path
import pandas as pd
import numpy as np
import os
import json

# Repo root by default; override with DL_PROJECT_DIR to point at a different checkout/data location.
base_dir = os.environ.get("DL_PROJECT_DIR") or str(Path(__file__).resolve().parents[1])
out_dir = os.path.join(base_dir, "data", "processed")
file_path = os.path.join(out_dir, "marine_catch_modeling_dataset_v4.csv")

print("Loading dataset...")
df = pd.read_csv(file_path, low_memory=False)
n_rows = len(df)
print(f"Total rows: {n_rows}")

report = {}

# 1. DUPLICATES
print("Checking duplicates...")
exact_dupes = df.duplicated().sum()
key_cols = ['YEAR', 'MONTH_START', 'FISHING_GROUND_CODE', 'FLEET', 'GEAR', 'SPECIES', 'target_catch_mt']
# Need to be careful with missing cols if FLEET_CODE is missing. 
# We'll use what we have in V4.
avail_key_cols = [c for c in key_cols if c in df.columns]
logical_dupes = df.duplicated(subset=avail_key_cols).sum()
report['duplicates'] = {
    'exact': int(exact_dupes),
    'logical': int(logical_dupes)
}

# 2. TARGET LEAKAGE (Correlation proxy + sanity check)
print("Checking target leakage...")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
leakage = {}
for c in num_cols:
    if c != 'target_catch_mt':
        corr = df[['target_catch_mt', c]].corr().iloc[0, 1]
        leakage[c] = float(corr) if not pd.isna(corr) else None
report['target_leakage_correlations'] = leakage

# 3. EFFORT MATCHING
print("Checking effort matching...")
effort_cols = ['total_effort_hooks', 'total_effort_fdays']
effort_groups = df.groupby(['YEAR', 'MONTH_START', 'FISHING_GROUND_CODE', 'FLEET', 'GEAR']).size().reset_index(name='catch_records_sharing_effort')
shared_stats = effort_groups['catch_records_sharing_effort'].describe().to_dict()
top_shared = effort_groups.sort_values('catch_records_sharing_effort', ascending=False).head(5).to_dict('records')
report['effort_matching'] = {
    'stats_of_shared_records': shared_stats,
    'top_shared_groups': top_shared
}

# 4. ARGO QUALITY
print("Checking Argo quality...")
def pct_stat(s, p):
    return float(np.nanpercentile(s, p)) if not s.dropna().empty else None

argo_quality = {}
for c in ['nearest_argo_distance_km', 'argo_temp', 'argo_sal']:
    s = df[c]
    argo_quality[c] = {
        'missing_pct': float(s.isna().mean() * 100),
        'min': float(s.min()) if not s.dropna().empty else None,
        'median': pct_stat(s, 50),
        'mean': float(s.mean()) if not s.dropna().empty else None,
        'p75': pct_stat(s, 75),
        'p90': pct_stat(s, 90),
        'p95': pct_stat(s, 95),
        'max': float(s.max()) if not s.dropna().empty else None
    }

dist = df['nearest_argo_distance_km'].dropna()
total_argo_matches = len(dist)
if total_argo_matches > 0:
    argo_quality['distance_buckets_pct'] = {
        'within_50km': float((dist <= 50).sum() / total_argo_matches * 100),
        'within_100km': float((dist <= 100).sum() / total_argo_matches * 100),
        'within_200km': float((dist <= 200).sum() / total_argo_matches * 100),
        'within_300km': float((dist <= 300).sum() / total_argo_matches * 100)
    }
report['argo_quality'] = argo_quality

# 5. FEATURE VARIABILITY
print("Checking feature variability...")
var_report = {}
for c in num_cols:
    s = df[c]
    var_report[c] = {
        'unique_values': int(s.nunique()),
        'missing_pct': float(s.isna().mean() * 100),
        'min': float(s.min()) if not s.dropna().empty else None,
        'max': float(s.max()) if not s.dropna().empty else None,
        'mean': float(s.mean()) if not s.dropna().empty else None,
        'std': float(s.std()) if not s.dropna().empty else None
    }
report['feature_variability'] = var_report

# 6. CATEGORICAL COVERAGE
print("Checking categorical coverage...")
cat_cols = ['FLEET', 'FISHERY', 'GEAR', 'SPECIES', 'SPECIES_CATEGORY', 'FISHING_GROUND_CODE']
cat_report = {}
for c in cat_cols:
    if c in df.columns:
        s = df[c].astype(str)
        counts = s.value_counts()
        cat_report[c] = {
            'unique_values': int(s.nunique()),
            'top_10': counts.head(10).to_dict(),
            'has_rare_under_5': bool((counts < 5).any())
        }
report['categorical_coverage'] = cat_report

# 7. TARGET DISTRIBUTION
print("Checking target distribution...")
tgt = df['target_catch_mt'].dropna()
report['target_distribution'] = {
    'zero_count': int((tgt == 0).sum()),
    'min': float(tgt.min()),
    'median': pct_stat(tgt, 50),
    'mean': float(tgt.mean()),
    'std': float(tgt.std()),
    'p90': pct_stat(tgt, 90),
    'p95': pct_stat(tgt, 95),
    'p99': pct_stat(tgt, 99),
    'max': float(tgt.max()),
    'skewness': float(tgt.skew())
}

# 8. INDOBIS
print("Checking IndOBIS...")
obs = df['nearby_biodiversity_observation_count']
matched_obs = df[obs.notna()]
report['indobis'] = {
    'missing_pct': float(obs.isna().mean() * 100),
    'matched_records': len(matched_obs)
}
if not matched_obs.empty:
    report['indobis']['top_months_matched'] = matched_obs['MONTH_START'].value_counts().head(3).to_dict()
    report['indobis']['top_grids_matched'] = matched_obs['FISHING_GROUND_CODE'].value_counts().head(3).to_dict()

out_json = os.path.join(out_dir, "audit_v4_report.json")
with open(out_json, "w") as f:
    json.dump(report, f, indent=4)

print(f"Audit complete. Results saved to {out_json}")
