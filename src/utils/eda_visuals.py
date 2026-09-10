"""
Exploratory Data Analysis (EDA) Visualization Suite
AI-Driven Unified Marine & Oceanographic Platform

Generates publication-standard visual artifacts for data distribution analysis,
missingness audits, cross-domain correlations, and geospatial mapping.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict


def setup_visualization_theme():
    """Sets a clean, modern aesthetic for publication-grade figures."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.titlesize"] = 16
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]


def plot_feature_distributions(
    df: pd.DataFrame, 
    output_path: str = "data/figures/01_feature_distributions.png"
):
    """
    Plots 4-panel histograms + KDE distributions of core oceanographic variables
    with skewness and mean indicators.
    """
    setup_visualization_theme()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    features = [
        ("water_temp_c", "Seawater Temperature (°C)", "#1f77b4"),
        ("salinity_psu", "Practical Salinity (PSU)", "#2ca02c"),
        ("dissolved_oxygen_ml_l", "Dissolved Oxygen (ml/L)", "#ff7f0e"),
        ("depth_m", "Sampling Depth (m)", "#9467bd")
    ]

    for ax, (col, title, color) in zip(axes.flatten(), features):
        if col in df.columns:
            valid_vals = df[col].dropna()
            skew_val = valid_vals.skew()
            mean_val = valid_vals.mean()
            median_val = valid_vals.median()

            sns.histplot(valid_vals, kde=True, ax=ax, color=color, alpha=0.5, edgecolor="black", bins=40)
            ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.8, label=f"Mean: {mean_val:.2f}")
            ax.axvline(median_val, color="black", linestyle=":", linewidth=1.8, label=f"Median: {median_val:.2f}")
            
            ax.set_title(f"{title} (Skew: {skew_val:+.2f})", fontweight="bold")
            ax.set_xlabel(title)
            ax.set_ylabel("Count")
            ax.legend(loc="upper right", frameon=True)

    plt.suptitle("Core Oceanographic Hydrographic Parameter Distributions", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved feature distributions plot to {output_path}")


def plot_correlation_matrix(
    df: pd.DataFrame, 
    output_path: str = "data/figures/02_correlation_matrix.png"
):
    """
    Plots a triangular Pearson correlation heatmap demonstrating cross-domain coupling
    between ocean physics, geochemistry, fisheries, and biological abundance.
    """
    setup_visualization_theme()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cols = [
        "water_temp_c", "salinity_psu", "dissolved_oxygen_ml_l", 
        "phosphate_umol_l", "nitrate_umol_l", "chlorophyll_a_ug_l", 
        "depth_m", "fishing_effort_hours", "catch_tonnage_mt", 
        "shannon_diversity_index", "target_abundance_density"
    ]
    existing_cols = [c for c in cols if c in df.columns]
    corr = df[existing_cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient (r)"},
        ax=ax
    )

    ax.set_title("Cross-Domain Oceanographic & Ecological Correlation Matrix", fontsize=15, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved correlation matrix plot to {output_path}")


def plot_geospatial_mapping(
    df: pd.DataFrame, 
    output_path: str = "data/figures/03_geospatial_temperature_salinity.png"
):
    """
    Geospatial scatter visualization: Latitude vs Longitude coordinates colored
    by seawater temperature with point sizes proportional to species abundance density.
    """
    setup_visualization_theme()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    
    sc = ax.scatter(
        df["longitude"], 
        df["latitude"], 
        c=df["water_temp_c"], 
        cmap="plasma", 
        alpha=0.65, 
        s=df["target_abundance_density"] / 12.0 + 15,
        edgecolors="none"
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Seawater Temperature (°C)", fontsize=11)

    ax.set_title("Geospatial Distribution: California Current Hydrographic Stations\nPoint Size = Species Abundance Density | Color = Water Temperature", fontsize=14, fontweight="bold")
    ax.set_xlabel("Station Longitude (°W)", fontweight="bold")
    ax.set_ylabel("Station Latitude (°N)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved geospatial mapping plot to {output_path}")


def plot_missingness_imputation_audit(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    output_path: str = "data/figures/04_missingness_imputation_audit.png"
):
    """
    Audits missing value distribution before vs after depth-stratified imputation.
    """
    setup_visualization_theme()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    target_cols = ["phosphate_umol_l", "nitrate_umol_l", "chlorophyll_a_ug_l", "catch_tonnage_mt"]
    existing_cols = [c for c in target_cols if c in raw_df.columns]

    fig, axes = plt.subplots(len(existing_cols), 2, figsize=(14, 3.5 * len(existing_cols)))

    for i, col in enumerate(existing_cols):
        # Raw with missing values
        ax_raw = axes[i, 0] if len(existing_cols) > 1 else axes[0]
        missing_count = raw_df[col].isna().sum()
        missing_pct = (missing_count / len(raw_df)) * 100.0

        sns.histplot(raw_df[col].dropna(), kde=True, ax=ax_raw, color="#d62728", alpha=0.5, bins=35)
        ax_raw.set_title(f"Raw {col}\n(Missing: {missing_count} rows | {missing_pct:.1f}%)", fontweight="bold")
        ax_raw.set_ylabel("Count")

        # Cleaned / Imputed
        ax_clean = axes[i, 1] if len(existing_cols) > 1 else axes[1]
        missing_after = clean_df[col].isna().sum()
        sns.histplot(clean_df[col], kde=True, ax=ax_clean, color="#2ca02c", alpha=0.5, bins=35)
        ax_clean.set_title(f"Depth-Stratified Imputed {col}\n(Post-Imputation Missing: {missing_after})", fontweight="bold")
        ax_clean.set_ylabel("Count")

    plt.suptitle("Missing Data & Depth-Stratified Imputation Integrity Verification", fontsize=15, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved missingness imputation audit plot to {output_path}")


def plot_target_distributions(
    df: pd.DataFrame, 
    output_path: str = "data/figures/05_target_biodiversity_distribution.png"
):
    """
    Dual-target distribution analysis:
    Panel A: Continuous Species Abundance Density (log-normal biological distribution)
    Panel B: Multi-class Marine Biodiversity Vulnerability Tiers (Low, Moderate, High, Critical)
    """
    setup_visualization_theme()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: Abundance density
    sns.histplot(df["target_abundance_density"], kde=True, ax=ax1, color="#17becf", bins=40, edgecolor="black")
    mean_abund = df["target_abundance_density"].mean()
    median_abund = df["target_abundance_density"].median()
    ax1.axvline(mean_abund, color="red", linestyle="--", label=f"Mean: {mean_abund:.1f}")
    ax1.axvline(median_abund, color="black", linestyle=":", label=f"Median: {median_abund:.1f}")
    ax1.set_title("Target A: Species Abundance Density (Regression)", fontweight="bold")
    ax1.set_xlabel("Abundance Density (individuals / 1000 m³)")
    ax1.set_ylabel("Sample Count")
    ax1.legend()

    # Panel B: Vulnerability tier balance
    tier_labels = ["0: Low Risk", "1: Moderate Risk", "2: High Risk", "3: Critical Vulnerability"]
    palette = ["#2ca02c", "#ffbb78", "#ff7f0e", "#d62728"]
    tier_counts = df["target_vulnerability_tier"].value_counts().sort_index()

    bars = ax2.bar(range(len(tier_counts)), tier_counts.values, color=palette, edgecolor="black", width=0.6)
    ax2.set_xticks(range(len(tier_counts)))
    ax2.set_xticklabels(tier_labels, rotation=15, ha="right", fontweight="bold")
    ax2.set_title("Target B: Ecological Biodiversity Vulnerability Tiers (Classification)", fontweight="bold")
    ax2.set_ylabel("Sample Count")

    # Add percentages above bars
    total = len(df)
    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100.0
        ax2.annotate(f"{height:,}\n({pct:.1f}%)",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha="center", va="bottom", fontweight="bold")

    plt.suptitle("Dual-Task Modeling Target Distribution Analysis", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved target distribution plot to {output_path}")


def generate_all_eda_plots(
    raw_df: pd.DataFrame, 
    clean_df: pd.DataFrame,
    figures_dir: str = "data/figures"
):
    """Executes the full suite of 5 publication-standard EDA visual generators."""
    print(f"\n[*] Generating comprehensive EDA visualizations in '{figures_dir}'...")
    plot_feature_distributions(clean_df, os.path.join(figures_dir, "01_feature_distributions.png"))
    plot_correlation_matrix(clean_df, os.path.join(figures_dir, "02_correlation_matrix.png"))
    plot_geospatial_mapping(clean_df, os.path.join(figures_dir, "03_geospatial_temperature_salinity.png"))
    plot_missingness_imputation_audit(raw_df, clean_df, os.path.join(figures_dir, "04_missingness_imputation_audit.png"))
    plot_target_distributions(clean_df, os.path.join(figures_dir, "05_target_biodiversity_distribution.png"))
    print("[✓] All EDA figures successfully generated!")
