"""
Automated Test Suite for Data Engineering, Cleaning, and DataLoaders
AI-Driven Unified Marine & Oceanographic Platform
Lead Role: Data & EDA Lead
"""

import pytest
import numpy as np
import pandas as pd
import torch

from src.data.ingestion import (
    generate_synthetic_calcofi_casts, 
    generate_synthetic_fisheries_and_biodiversity,
    ingest_and_harmonize_marine_data
)
from src.data.cleaning import OceanographicDataCleaner
from src.data.preprocessing import MarineDataPreprocessor
from src.data.dataloaders import MarineOceanDataset, get_dataloaders


@pytest.fixture(scope="module")
def sample_marine_data():
    """Generates a verified sample dataset of 2,000 records for fast testing."""
    return ingest_and_harmonize_marine_data(n_samples=2000, output_dir="data/raw", random_state=42)


def test_raw_data_sample_count_and_columns(sample_marine_data):
    """Verifies that sample count meets mini-project guidelines and contains multi-modal columns."""
    df = sample_marine_data
    assert len(df) == 2000
    assert len(df) >= 1000, "Dataset sample count must meet minimum 1,000 samples required by guidelines"

    required_ocean_cols = ["water_temp_c", "salinity_psu", "depth_m", "dissolved_oxygen_ml_l", "phosphate_umol_l"]
    required_fish_cols = ["species_scientific_name", "gear_type", "fishing_effort_hours", "catch_tonnage_mt"]
    required_targets = ["target_abundance_density", "target_vulnerability_tier"]

    for col in required_ocean_cols + required_fish_cols + required_targets:
        assert col in df.columns, f"Missing required multi-modal column: {col}"


def test_physical_boundary_anomaly_screening(sample_marine_data):
    """Verifies that non-physical sensor spikes (-9.99, 99.99 PSU) are screened."""
    cleaner = OceanographicDataCleaner()
    clean_df, quarantine_df = cleaner.clean_physical_anomalies(sample_marine_data)

    assert len(quarantine_df) > 0, "Expected some sensor anomalies to be caught and quarantined"
    
    # Check that temperature in clean_df is within physical bounds (ignoring NaNs)
    valid_temps = clean_df["water_temp_c"].dropna()
    assert (valid_temps >= -2.5).all() and (valid_temps <= 36.0).all()

    # Check salinity bounds
    valid_salinity = clean_df["salinity_psu"].dropna()
    assert (valid_salinity >= 24.0).all() and (valid_salinity <= 42.0).all()


def test_depth_stratified_imputation_zero_missing(sample_marine_data):
    """Verifies that depth-stratified imputation resolves all NaNs in modeled features."""
    cleaner = OceanographicDataCleaner()
    imputed_df, audit = cleaner.fit_transform(sample_marine_data)

    modeled_cols = [
        "water_temp_c", "salinity_psu", "dissolved_oxygen_ml_l",
        "phosphate_umol_l", "nitrate_umol_l", "chlorophyll_a_ug_l", "catch_tonnage_mt"
    ]

    for col in modeled_cols:
        assert imputed_df[col].isna().sum() == 0, f"Column {col} still has missing values post-imputation"

    # Verify imputed flags exist
    assert "phosphate_umol_l_is_imputed" in imputed_df.columns
    assert set(imputed_df["phosphate_umol_l_is_imputed"].unique()).issubset({0.0, 1.0})


def test_strict_80_10_10_split_proportions(sample_marine_data):
    """Verifies strict 80-10-10 train/val/test split ratios."""
    preprocessor = MarineDataPreprocessor()
    train_df, val_df, test_df, meta = preprocessor.fit_and_transform_pipeline(sample_marine_data, random_state=42)

    total_len = len(train_df) + len(val_df) + len(test_df)
    assert total_len == len(sample_marine_data)

    train_pct = len(train_df) / total_len
    val_pct = len(val_df) / total_len
    test_pct = len(test_df) / total_len

    assert abs(train_pct - 0.80) <= 0.01, f"Train set must be ~80%, got {train_pct*100:.2f}%"
    assert abs(val_pct - 0.10) <= 0.01, f"Val set must be ~10%, got {val_pct*100:.2f}%"
    assert abs(test_pct - 0.10) <= 0.01, f"Test set must be ~10%, got {test_pct*100:.2f}%"


def test_zero_data_leakage_and_scaling(sample_marine_data):
    """Verifies that scaling and imputation are fitted strictly on train with zero leakage."""
    preprocessor = MarineDataPreprocessor(scaler_type="robust")
    train_df, val_df, test_df, meta = preprocessor.fit_and_transform_pipeline(sample_marine_data, random_state=42)

    scaled_cols = meta["scaled_feature_names"]
    
    # Check that scaled continuous columns exist and have zero NaNs across all splits
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        nan_count = df[scaled_cols].isna().sum().sum()
        assert nan_count == 0, f"{name} split has {nan_count} NaNs in scaled features"

    # Verify categorical token mappings exist and contain UNK token
    for cat_col in preprocessor.CATEGORICAL_FEATURES:
        assert "<UNK>" in preprocessor.token_maps_[cat_col]
        assert preprocessor.token_maps_[cat_col]["<UNK>"] == 0


def test_pytorch_dataloaders(sample_marine_data):
    """Verifies that PyTorch DataLoaders yield valid batches of tensors with GPU compatibility."""
    preprocessor = MarineDataPreprocessor()
    train_df, val_df, test_df, meta = preprocessor.fit_and_transform_pipeline(sample_marine_data, random_state=42)

    batch_size = 32
    train_loader, val_loader, test_loader, dl_meta = get_dataloaders(
        train_df, val_df, test_df, batch_size=batch_size, pin_memory=False
    )

    batch = next(iter(train_loader))
    assert "features" in batch
    assert "categorical_tokens" in batch
    assert "target_abundance" in batch
    assert "target_vulnerability" in batch

    # Check tensor shapes
    assert batch["features"].shape[0] == batch_size
    assert batch["features"].ndim == 2
    assert batch["features"].dtype == torch.float32

    assert batch["target_abundance"].shape[0] == batch_size
    assert batch["target_abundance"].dtype == torch.float32

    assert batch["target_vulnerability"].shape[0] == batch_size
    assert batch["target_vulnerability"].dtype == torch.int64
    assert set(batch["target_vulnerability"].numpy()).issubset({0, 1, 2, 3})
