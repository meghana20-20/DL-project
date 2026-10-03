from pathlib import Path
import pandas as pd
import numpy as np
import os
import json
import re
from sklearn.neighbors import BallTree

# Repo root by default; override with DL_PROJECT_DIR to point at a different checkout/data location.
base_dir = os.environ.get("DL_PROJECT_DIR") or str(Path(__file__).resolve().parents[1])
out_dir = os.path.join(base_dir, "data", "processed")
raw_iotc = os.path.join(base_dir, "data", "raw", "iotc")
raw_argo = os.path.join(base_dir, "data", "raw", "argo", "indian_ocean_argo_full.csv")
raw_indobis = os.path.join(base_dir, "data", "raw", "indobis", "occurrence.txt")
grid_file = os.path.join(raw_iotc, "IOTC_GRIDS_CE_SF.csv")

EARTH_RADIUS = 6371.0 # km

def extract_resolution(name):
    if not isinstance(name, str): return 5.0
    match = re.search(r'(\d+)[\D]+(\d+)[\D]+latitude', name, re.IGNORECASE)
    if match:
        return float(abs(int(match.group(2)) - int(match.group(1))))
    return 5.0

print("1. Loading IOTC CATCH...")
ca_df = pd.read_csv(os.path.join(raw_iotc, "IOTC-2026-WPTT28-DATA03_CA.csv"), low_memory=False)
ca_df = ca_df[(ca_df['YEAR'] >= 2018) & (ca_df['YEAR'] <= 2023) & (ca_df['CATCH_UNIT_CODE'] == 'MT')].copy()
ca_df['target_catch_mt'] = ca_df['CATCH']

keep_cols = [
    'YEAR', 'MONTH_START', 'MONTH_END', 'FISHING_GROUND_CODE', 'FLEET_CODE', 'FLEET', 
    'FISHERY_TYPE', 'FISHERY_GROUP', 'FISHERY', 'GEAR_CODE', 'GEAR', 'SPECIES', 
    'SPECIES_CATEGORY', 'SPECIES_CODE', 'target_catch_mt'
]
existing_cols = [c for c in keep_cols if c in ca_df.columns]
ca_df = ca_df[existing_cols].copy()

if len(ca_df) > 20000:
    ca_df = ca_df.sample(20000, random_state=42)

print(f"Sampled {len(ca_df)} catch records.")

print("2. Attaching Official Geography...")
grid_df = pd.read_csv(grid_file)
grid_df['code'] = grid_df['code'].astype(str)
grid_df['grid_resolution'] = grid_df['name_en'].apply(extract_resolution)

ca_df['FISHING_GROUND_CODE'] = ca_df['FISHING_GROUND_CODE'].astype(str)
ca_df = ca_df.merge(
    grid_df[['code', 'center_lat', 'center_lon', 'grid_resolution', 'ocean_area_iotc_km2']],
    left_on='FISHING_GROUND_CODE', right_on='code', how='left'
)
ca_df.rename(columns={'center_lat': 'fishing_ground_lat', 'center_lon': 'fishing_ground_lon'}, inplace=True)
ca_df.drop(columns=['code'], inplace=True, errors='ignore')

print("3. Loading IOTC EFFORT...")
ef_df = pd.read_csv(os.path.join(raw_iotc, "IOTC-2026-WPTT28-DATA03_EF.csv"), low_memory=False)
ef_df = ef_df[(ef_df['YEAR'] >= 2018) & (ef_df['YEAR'] <= 2023)].copy()
ef_df['FISHING_GROUND_CODE'] = ef_df['FISHING_GROUND_CODE'].astype(str)

hooks = ef_df[ef_df['EFFORT_UNIT_CODE'].isin(['HOOKS', 'HK'])].groupby(['YEAR', 'MONTH_START', 'FISHING_GROUND_CODE', 'FLEET_CODE', 'GEAR_CODE'])['EFFORT'].sum().reset_index(name='total_effort_hooks')
fdays = ef_df[ef_df['EFFORT_UNIT_CODE'].isin(['FDAYS', 'FD'])].groupby(['YEAR', 'MONTH_START', 'FISHING_GROUND_CODE', 'FLEET_CODE', 'GEAR_CODE'])['EFFORT'].sum().reset_index(name='total_effort_fdays')
record_count = ef_df.groupby(['YEAR', 'MONTH_START', 'FISHING_GROUND_CODE', 'FLEET_CODE', 'GEAR_CODE']).size().reset_index(name='effort_record_count')

ca_df = ca_df.merge(hooks, on=['YEAR', 'MONTH_START', 'FISHING_GROUND_CODE', 'FLEET_CODE', 'GEAR_CODE'], how='left')
ca_df = ca_df.merge(fdays, on=['YEAR', 'MONTH_START', 'FISHING_GROUND_CODE', 'FLEET_CODE', 'GEAR_CODE'], how='left')
ca_df = ca_df.merge(record_count, on=['YEAR', 'MONTH_START', 'FISHING_GROUND_CODE', 'FLEET_CODE', 'GEAR_CODE'], how='left')

print("4. Loading Argo (Expanded Dataset)...")
argo_df = pd.read_csv(raw_argo, skiprows=[1], low_memory=False)
argo_df['time'] = pd.to_datetime(argo_df['time'], errors='coerce', utc=True)
argo_df = argo_df.dropna(subset=['time', 'latitude', 'longitude'])
argo_df['year'] = argo_df['time'].dt.year
argo_df['month'] = argo_df['time'].dt.month
# Drop any rows where both TEMP and SAL are missing to ensure we only get valid observations
argo_df = argo_df.dropna(subset=['TEMP', 'SAL'], how='all')

print("5. Loading IndOBIS...")
obis_df = pd.read_csv(raw_indobis, sep='\t', usecols=['eventDate', 'decimalLatitude', 'decimalLongitude', 'scientificName'], low_memory=False)
obis_df['date'] = pd.to_datetime(obis_df['eventDate'], errors='coerce', utc=True)
obis_df = obis_df.dropna(subset=['date', 'decimalLatitude', 'decimalLongitude'])
obis_df['species'] = obis_df['scientificName']
obis_df['genus'] = obis_df['scientificName'].astype(str).str.split().str[0]
obis_df = obis_df.dropna(subset=['date']).copy()
obis_df['lat_rad'] = np.radians(obis_df['decimalLatitude'])
obis_df['lon_rad'] = np.radians(obis_df['decimalLongitude'])

if not obis_df.empty:
    obis_tree = BallTree(obis_df[['lat_rad', 'lon_rad']].values, metric='haversine')
else:
    obis_tree = None

print("6. Matching Argo & IndOBIS...")

argo_cols = ['argo_temp', 'argo_sal', 'nearest_argo_lat', 'nearest_argo_lon', 'nearest_argo_distance_km']
indobis_cols = ['nearby_biodiversity_observation_count', 'nearby_biodiversity_species_count', 'nearby_biodiversity_genus_count', 'nearest_biodiversity_distance_km']

for col in argo_cols + indobis_cols:
    ca_df[col] = np.nan

unique_st = ca_df[['YEAR', 'MONTH_START', 'fishing_ground_lat', 'fishing_ground_lon']].drop_duplicates().dropna()

results_argo = {}
results_obis = {}

# Build Argo trees per year/month
argo_trees = {}
for (y, m), group in argo_df.groupby(['year', 'month']):
    group = group.copy()
    group['lat_rad'] = np.radians(group['latitude'])
    group['lon_rad'] = np.radians(group['longitude'])
    tree = BallTree(group[['lat_rad', 'lon_rad']].values, metric='haversine')
    argo_trees[(y, m)] = (tree, group)

for idx, row in unique_st.iterrows():
    y, m, lat, lon = int(row['YEAR']), int(row['MONTH_START']), row['fishing_ground_lat'], row['fishing_ground_lon']
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # --- ARGO (300 km) ---
    a_res = [np.nan] * 5
    if (y, m) in argo_trees:
        tree, group = argo_trees[(y, m)]
        dist, ind = tree.query([[lat_rad, lon_rad]], k=1)
        dist_km = dist[0][0] * EARTH_RADIUS
        if dist_km <= 300.0:
            nearest_row = group.iloc[ind[0][0]]
            a_res = [
                nearest_row['TEMP'],
                nearest_row['SAL'],
                nearest_row['latitude'],
                nearest_row['longitude'],
                dist_km
            ]
    results_argo[(y, m, lat, lon)] = a_res
    
    # --- INDOBIS (500 km, +- 12 months) ---
    o_res = [np.nan] * 4
    if obis_tree is not None:
        target_date = pd.to_datetime(f"{y}-{m:02d}-15", utc=True)
        ind, dist = obis_tree.query_radius([[lat_rad, lon_rad]], r=500.0/EARTH_RADIUS, return_distance=True)
        indices = ind[0]
        distances = dist[0] * EARTH_RADIUS
        
        if len(indices) > 0:
            nearby_obis = obis_df.iloc[indices].copy()
            nearby_obis['dist_km'] = distances
            nearby_obis['days_diff'] = (nearby_obis['date'] - target_date).dt.days.abs()
            time_filtered = nearby_obis[nearby_obis['days_diff'] <= 365]
            
            if not time_filtered.empty:
                o_res = [
                    len(time_filtered),
                    time_filtered['species'].nunique(),
                    time_filtered['genus'].nunique(),
                    time_filtered['dist_km'].min()
                ]
    results_obis[(y, m, lat, lon)] = o_res

def get_argo(row):
    return results_argo.get((row['YEAR'], row['MONTH_START'], row['fishing_ground_lat'], row['fishing_ground_lon']), [np.nan]*5)

def get_obis(row):
    return results_obis.get((row['YEAR'], row['MONTH_START'], row['fishing_ground_lat'], row['fishing_ground_lon']), [np.nan]*4)

mask = ca_df['fishing_ground_lat'].notna()
ca_df.loc[mask, argo_cols] = ca_df[mask].apply(get_argo, axis=1, result_type='expand').values
ca_df.loc[mask, indobis_cols] = ca_df[mask].apply(get_obis, axis=1, result_type='expand').values

print("7. Quality Checks and Report...")

report = {
    "final_row_count": len(ca_df),
    "argo_match_rate": float(ca_df['argo_temp'].notna().mean() * 100),
    "indobis_match_rate": float(ca_df['nearby_biodiversity_observation_count'].notna().mean() * 100),
    "effort_match_rate": float(ca_df['effort_record_count'].notna().mean() * 100),
    "missing_percentage": {},
    "unique_values": {},
    "sample_rows_proof": []
}

env_bio_cols = ['argo_temp', 'argo_sal', 'nearby_biodiversity_species_count', 'nearest_argo_distance_km', 'nearest_biodiversity_distance_km']
for c in env_bio_cols + ['total_effort_hooks', 'total_effort_fdays']:
    report['missing_percentage'][c] = float(ca_df[c].isna().mean() * 100)

for c in ['argo_temp', 'argo_sal', 'total_effort_hooks', 'total_effort_fdays', 'nearby_biodiversity_species_count']:
    report['unique_values'][c] = int(ca_df[c].nunique())

sample = ca_df.dropna(subset=['argo_temp']).head(10)
report['sample_rows_proof'] = sample[['YEAR', 'MONTH_START', 'FISHING_GROUND_CODE', 'fishing_ground_lat', 'fishing_ground_lon', 'argo_temp', 'argo_sal', 'nearest_argo_distance_km']].to_dict('records')

print("Saving V4 dataset...")
ca_df.to_csv(os.path.join(out_dir, "marine_catch_modeling_dataset_v4.csv"), index=False)
with open(os.path.join(out_dir, "marine_catch_modeling_dataset_v4_report.json"), "w") as f:
    json.dump(report, f, indent=4)

print("Done. V4 built.")
