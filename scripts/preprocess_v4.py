import pandas as pd
import numpy as np
import os
import json
import joblib
from scipy import sparse
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

base_dir = r"C:\Users\hemas\OneDrive\Desktop\dl project\DL-project"
out_dir = os.path.join(base_dir, "data", "processed")
file_path = os.path.join(out_dir, "marine_catch_modeling_dataset_v4.csv")

print("1. Loading V4 dataset...")
df = pd.read_csv(file_path, low_memory=False)

# Sort chronologically
df = df.sort_values(by=['YEAR', 'MONTH_START']).reset_index(drop=True)

# 1. Create Chronological Splits
print("2. Creating chronological splits...")
# Unique year-months
df['year_month'] = df['YEAR'].astype(str) + "_" + df['MONTH_START'].astype(str)
unique_ym = df[['YEAR', 'MONTH_START', 'year_month']].drop_duplicates().sort_values(['YEAR', 'MONTH_START'])
ym_list = unique_ym['year_month'].tolist()

n_ym = len(ym_list)
train_end = int(n_ym * 0.8)
val_end = int(n_ym * 0.9)

train_ym = set(ym_list[:train_end])
val_ym = set(ym_list[train_end:val_end])
test_ym = set(ym_list[val_end:])

df_train = df[df['year_month'].isin(train_ym)].copy()
df_val = df[df['year_month'].isin(val_ym)].copy()
df_test = df[df['year_month'].isin(test_ym)].copy()

print(f"Train rows: {len(df_train)}, Val rows: {len(df_val)}, Test rows: {len(df_test)}")

df_train.drop(columns=['year_month'], inplace=True)
df_val.drop(columns=['year_month'], inplace=True)
df_test.drop(columns=['year_month'], inplace=True)

df_train.to_csv(os.path.join(out_dir, "v4_train.csv"), index=False)
df_val.to_csv(os.path.join(out_dir, "v4_val.csv"), index=False)
df_test.to_csv(os.path.join(out_dir, "v4_test.csv"), index=False)

# Targets
print("3. Handling Targets...")
y_train = df_train['target_catch_mt'].values
y_val = df_val['target_catch_mt'].values
y_test = df_test['target_catch_mt'].values

y_log_train = np.log1p(y_train)
y_log_val = np.log1p(y_val)
y_log_test = np.log1p(y_test)

# Custom Preprocessing before Scikit-Learn
print("4. Feature Engineering...")

def feature_engineer(data):
    d = data.copy()
    
    # Missingness indicators for effort
    d['has_effort_hooks'] = d['total_effort_hooks'].notna().astype(int)
    d['has_effort_fdays'] = d['total_effort_fdays'].notna().astype(int)
    
    # Fill effort missing with 0
    d['total_effort_hooks'] = d['total_effort_hooks'].fillna(0)
    d['total_effort_fdays'] = d['total_effort_fdays'].fillna(0)
    
    # Missingness indicators for Argo
    d['has_argo_temp'] = d['argo_temp'].notna().astype(int)
    d['has_argo_sal'] = d['argo_sal'].notna().astype(int)
    d['has_argo_match'] = d['nearest_argo_distance_km'].notna().astype(int)
    
    return d

X_train = feature_engineer(df_train)
X_val = feature_engineer(df_val)
X_test = feature_engineer(df_test)

# Report variables before Pipeline
missing_before = df_train.isna().sum().to_dict()

# Features for Pipeline
cat_cols = ['FLEET', 'FISHERY', 'FISHERY_GROUP', 'GEAR', 'SPECIES', 'SPECIES_CATEGORY', 'FISHING_GROUND_CODE']
# Convert FGC to string to treat as categorical
X_train['FISHING_GROUND_CODE'] = X_train['FISHING_GROUND_CODE'].astype(str)
X_val['FISHING_GROUND_CODE'] = X_val['FISHING_GROUND_CODE'].astype(str)
X_test['FISHING_GROUND_CODE'] = X_test['FISHING_GROUND_CODE'].astype(str)

num_cols = [
    'MONTH_START', 
    'fishing_ground_lat', 'fishing_ground_lon', 'grid_resolution', 'ocean_area_iotc_km2',
    'argo_temp', 'argo_sal', 'nearest_argo_distance_km',
    'total_effort_hooks', 'total_effort_fdays', 'effort_record_count',
    'has_effort_hooks', 'has_effort_fdays', 'has_argo_temp', 'has_argo_sal', 'has_argo_match'
]

print("5. Fitting Scikit-Learn Pipeline...")

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('cat', cat_pipeline, cat_cols),
    ('num', num_pipeline, num_cols)
])

# FIT ONLY ON TRAIN
X_train_transformed = preprocessor.fit_transform(X_train)
X_val_transformed = preprocessor.transform(X_val)
X_test_transformed = preprocessor.transform(X_test)

# Feature Names
cat_features = preprocessor.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(cat_cols)
all_feature_names = list(cat_features) + num_cols

print("6. Saving Artifacts...")
joblib.dump(preprocessor, os.path.join(out_dir, "v4_preprocessor.joblib"))
with open(os.path.join(out_dir, "v4_feature_names.json"), "w") as f:
    json.dump(all_feature_names, f, indent=4)

sparse.save_npz(os.path.join(out_dir, "v4_X_train.npz"), X_train_transformed)
sparse.save_npz(os.path.join(out_dir, "v4_X_val.npz"), X_val_transformed)
sparse.save_npz(os.path.join(out_dir, "v4_X_test.npz"), X_test_transformed)

np.save(os.path.join(out_dir, "v4_y_train.npy"), y_train)
np.save(os.path.join(out_dir, "v4_y_val.npy"), y_val)
np.save(os.path.join(out_dir, "v4_y_test.npy"), y_test)

np.save(os.path.join(out_dir, "v4_y_log_train.npy"), y_log_train)
np.save(os.path.join(out_dir, "v4_y_log_val.npy"), y_log_val)
np.save(os.path.join(out_dir, "v4_y_log_test.npy"), y_log_test)

def get_stats(arr):
    if len(arr) == 0: return {}
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr))
    }

print("7. Generating Report...")
report = {
    "split_sizes": {
        "train": len(X_train),
        "val": len(X_val),
        "test": len(X_test)
    },
    "year_month_ranges": {
        "train_min": str(min(train_ym)), "train_max": str(max(train_ym)),
        "val_min": str(min(val_ym)), "val_max": str(max(val_ym)),
        "test_min": str(min(test_ym)), "test_max": str(max(test_ym))
    },
    "feature_counts": {
        "categorical_features_count": len(cat_cols),
        "numerical_features_count": len(num_cols),
        "final_encoded_feature_count": len(all_feature_names)
    },
    "missing_values_before_handling": {k: int(v) for k, v in missing_before.items() if v > 0},
    "missingness_indicators_created": [
        'has_effort_hooks', 'has_effort_fdays', 
        'has_argo_temp', 'has_argo_sal', 'has_argo_match'
    ],
    "confirmation": "All imputers (median/most_frequent), scalers (StandardScaler), and encoders (OHE) were fitted STRICTLY on the training set only.",
    "target_statistics": {
        "before_log1p": get_stats(y_train),
        "after_log1p": get_stats(y_log_train)
    }
}

with open(os.path.join(out_dir, "v4_preprocessing_report.json"), "w") as f:
    json.dump(report, f, indent=4)

print("Done. V4 preprocessed successfully.")
