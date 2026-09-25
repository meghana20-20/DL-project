import os
import requests
import pandas as pd
import numpy as np
from io import BytesIO

base_dir = r"C:\Users\hemas\OneDrive\Desktop\dl project\DL-project"
out_dir = os.path.join(base_dir, "data", "raw", "argo")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "indian_ocean_argo_full.csv")

url = "https://erddap.incois.gov.in/erddap/griddap/incois_argo_mnt_VAM.csv"
url += "?TEMP[(2018-01-01T00:00:00Z):1:(2023-12-31T00:00:00Z)][(5.0):1:(5.0)][(-30.0):1:(30.0)][(30.0):1:(120.0)],"
url += "SAL[(2018-01-01T00:00:00Z):1:(2023-12-31T00:00:00Z)][(5.0):1:(5.0)][(-30.0):1:(30.0)][(30.0):1:(120.0)]"

import urllib3
urllib3.disable_warnings()

print(f"Downloading from ERDDAP: {url}")
# We can stream the download in case it's large
try:
    with requests.get(url, stream=True, verify=False) as r:
        r.raise_for_status()
        with open(out_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Download complete: {out_file}")
except Exception as e:
    print("Download error:", e)
    
print("Analyzing the downloaded data...")

# ERDDAP CSVs have a second row with units. We skip it for data processing
# but we can just load the whole thing as strings, or skip the second row.
file_size = os.path.getsize(out_file)

df = pd.read_csv(out_file, skiprows=[1])

# Basic stats
num_rows = len(df)
cols = list(df.columns)
lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
time_min, time_max = df['time'].min(), df['time'].max()
non_null_temp = df['TEMP'].notna().sum()
non_null_sal = df['SAL'].notna().sum()
unique_locs = df[['latitude', 'longitude']].drop_duplicates().shape[0]

print("==================================================")
print("REPORT")
print("==================================================")
print(f"1. File size: {file_size / (1024*1024):.2f} MB")
print(f"2. Number of rows: {num_rows}")
print(f"3. Columns: {cols}")
print(f"4. Actual latitude range: {lat_min} to {lat_max}")
print(f"5. Actual longitude range: {lon_min} to {lon_max}")
print(f"6. Actual time range: {time_min} to {time_max}")
print(f"7. Number of non-null TEMP values: {non_null_temp}")
print(f"8. Number of non-null SAL values: {non_null_sal}")
print(f"9. Number of unique latitude/longitude locations: {unique_locs}")
print("==================================================")
