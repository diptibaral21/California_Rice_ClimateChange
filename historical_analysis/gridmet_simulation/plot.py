"""
plot_gridmet_results.py

Author: Dipti Baral
Project: California Rice Climate Change

=========================================================
OVERVIEW
=========================================================
This script generates publication-quality figures for:

    (a) Model performance → R² distribution
    (b) Historical yield simulation → statewide area-weighted yield

The script ensures that both simulated and observed yields are
aggregated consistently at the STATEWIDE level using area-weighting.

=========================================================
WHY THIS SCRIPT IS IMPORTANT
=========================================================
Model outputs are generated at the county level, but:

    - Rice production is not evenly distributed across counties
    - Larger rice-growing counties contribute more to total production

Therefore, simple averaging is incorrect.

Instead, we compute:

    Statewide Yield = Σ (Yield_county × Area_county) / Σ Area_county

This ensures:
    - realistic representation of statewide yield
    - comparability between simulated and observed yields
    - scientifically valid interpretation

=========================================================
INPUT FILES
=========================================================

1. final_metrics.csv
    → output from Lasso model training
    → contains R² and RMSE for 1000 iterations

2. gridmet_simulation_statewide.csv
    → output from simulation script
    → contains:
        year, mean, p67_low, p67_high, p95_low, p95_high

3. county-level observed yield
    → must contain:
        county, year, yield

4. county rice area file
    → must contain:
        county, rice_area_ha

=========================================================
OUTPUT FILES
=========================================================

1. R2_distribution.png
2. Statewide_Yield.png
3. Combined_figure.png

=========================================================
PLOT STRUCTURE
=========================================================

Panel (a): R² distribution
    - Shows model stability across 1000 runs
    - Compares training vs testing performance

Panel (b): Time series
    - Simulated statewide yield (mean)
    - 67% confidence interval (dark band)
    - 95% confidence interval (light band)
    - Observed statewide yield (dashed line)

=========================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# PATHS (MATCHES YOUR WORKFLOW)
# =========================================================
CLIMATE_DIR = "/group/moniergrp/dbaral"

output_path = os.path.join(CLIMATE_DIR, "run_project/output_data/historical_model")
input_path = os.path.join(CLIMATE_DIR, "run_project/input_data")

metrics_path = os.path.join(output_path, "final_cleaned_metrics.csv")
simulation_path = os.path.join(output_path, "gridmet_simulation_statewide.csv")

observed_county_path = os.path.join(input_path, "yield/rice_yield_1979_2023.csv")
area_path = os.path.join(input_path, "rice_area/county_rice_area_static.csv")


# =========================================================
# STYLE SETTINGS
# =========================================================
sns.set_style("whitegrid")

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 15
})


# =========================================================
# FUNCTION: COMPUTE STATEWIDE OBSERVED YIELD
# =========================================================
def compute_statewide_observed():

    """
    Convert county-level observed yield into statewide area-weighted yield.

    Steps:
    ------
    1. Load county-level observed yield
    2. Load county rice area
    3. Standardize county names
    4. Merge datasets
    5. Compute area-weighted yield per year

    Returns:
    --------
    DataFrame with:
        year, yield_kg_ha
    """

    df_obs = pd.read_csv(observed_county_path)
    df_area = pd.read_csv(area_path)

    # -----------------------------------------------------
    # Standardize county names (IMPORTANT)
    # -----------------------------------------------------
    df_obs["county"] = df_obs["county"].str.strip().str.lower()
    df_area["county"] = df_area["county"].str.strip().str.lower()

    # -----------------------------------------------------
    # Merge
    # -----------------------------------------------------
    df = df_obs.merge(df_area, on="county", how="left")

    # -----------------------------------------------------
    # Compute area-weighted yield
    # -----------------------------------------------------
    statewide = (
        df.groupby("year")
        .apply(lambda x: np.sum(x["yield_kg_ha"] * x["rice_area_ha"]) / np.sum(x["rice_area_ha"]))
        .reset_index(name="yield_kg_ha")
    )

    return statewide


# =========================================================
# PANEL (a): R² DISTRIBUTION
# =========================================================
def plot_r2():

    df = pd.read_csv(metrics_path)

    df_melt = df.melt(
        value_vars=["R2_train", "R2_test"],
        var_name="Dataset",
        value_name="R2"
    )

    df_melt["Dataset"] = df_melt["Dataset"].map({
        "R2_train": "Training set",
        "R2_test": "Testing set"
    })

    # -----------------------------------------------------
    # Compute summary stats (mean or median)
    # -----------------------------------------------------
    r2_train_mean = df["R2_train"].mean()
    r2_test_mean = df["R2_test"].mean()

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------
    plt.figure(figsize=(5,5))

    sns.boxplot(
        data=df_melt,
        x="Dataset",
        y="R2",
        color="#2C6C8C",
        showfliers=False   #removes outliers
    )

    # -----------------------------------------------------
    # Add text (top-right corner)
    # -----------------------------------------------------
    text_str = (
        f"Train R² = {r2_train_mean:.2f}\n"
        f"Test R² = {r2_test_mean:.2f}"
    )

    plt.text(
        0.98, 1.00, text_str,
        transform=plt.gca().transAxes,  # relative axis coords
        ha="right", va="top",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
    )

    # -----------------------------------------------------
    # Labels
    # -----------------------------------------------------
    plt.title("a", loc="left", fontweight="bold")
    plt.ylabel(r"$R^2$")
    plt.xlabel("")

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "R2_distribution.png"), dpi=300)
    plt.show()

# =========================================================
# PANEL (b): STATEWIDE TIME SERIES
# =========================================================
def plot_statewide_timeseries():

    df_pred = pd.read_csv(simulation_path)

    df_obs = compute_statewide_observed()

    df = df_pred.merge(df_obs, on="year")

    plt.figure(figsize=(10,5))

    # 95% CI
    plt.fill_between(
        df["year"], df["p95_low"], df["p95_high"],
        color="#4C84A6", alpha=0.2, label="95% CI"
    )

    # 67% CI
    plt.fill_between(
        df["year"], df["p67_low"], df["p67_high"],
        color="#4C84A6", alpha=0.4, label="67% CI"
    )

    # predicted
    plt.plot(
        df["year"], df["mean"],
        color="#2C6C8C", linewidth=2, label="Simulated (GridMET)"
    )

    # observed
    plt.plot(
        df["year"], df["yield_kg_ha"],
        color="black", linestyle="--", linewidth=2, label="Observed"
    )

    plt.title("b", loc="left", fontweight="bold")
    plt.ylabel("Statewide Yield (kg/ha)")
    plt.xlabel("Year")

    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "Statewide_Yield.png"), dpi=300)
    plt.show()


# =========================================================
# COMBINED FIGURE
# =========================================================
def plot_combined():

    df_metrics = pd.read_csv(metrics_path)
    df_pred = pd.read_csv(simulation_path)
    df_obs = compute_statewide_observed()

    df = df_pred.merge(df_obs, on="year")

    fig, axes = plt.subplots(1, 2, figsize=(12,5))

    # =====================================================
    # Panel (a): R² distribution
    # =====================================================
    df_melt = df_metrics.melt(
        value_vars=["R2_train", "R2_test"],
        var_name="Dataset",
        value_name="R2"
    )

    df_melt["Dataset"] = df_melt["Dataset"].map({
        "R2_train": "Training set",
        "R2_test": "Testing set"
    })

    # compute summary 
    r2_train = df_metrics["R2_train"].median()
    r2_test = df_metrics["R2_test"].median()

    sns.boxplot(
        data=df_melt,
        x="Dataset",
        y="R2",
        ax=axes[0],
        color="#2C6C8C",
        showfliers=False   # remove outliers
    )

    axes[0].set_title("a", loc="left", fontweight="bold")
    axes[0].set_ylabel(r"$R^2$")
    axes[0].set_xlabel("")

    # add annotation (top-right inside panel)
    text_str = (
        f"Train R² = {r2_train:.2f}\n"
        f"Test R² = {r2_test:.2f}"
    )

    axes[0].text(
        0.98, 1.00, text_str,
        transform=axes[0].transAxes,
        ha="right", va="top",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
    )

    # =====================================================
    # Panel (b): Time series
    # =====================================================
    ax = axes[1]

    # 95% CI
    ax.fill_between(
        df["year"], df["p95_low"], df["p95_high"],
        color="#4C84A6", alpha=0.2, label="95% CI"
    )

    # 67% CI
    ax.fill_between(
        df["year"], df["p67_low"], df["p67_high"],
        color="#4C84A6", alpha=0.4, label="67% CI"
    )

    # predicted
    ax.plot(
        df["year"], df["mean"],
        color="#2C6C8C", linewidth=2, label="Simulated"
    )

    # observed
    ax.plot(
        df["year"], df["yield_kg_ha"],
        color="black", linestyle="--", linewidth=2, label="Observed"
    )

    ax.set_title("b", loc="left", fontweight="bold")
    ax.set_ylabel("Statewide Yield (kg/ha)")
    ax.set_xlabel("Year")

    ax.legend(frameon=False)

    # =====================================================
    # Final layout
    # =====================================================
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "Combined_figure.png"), dpi=300)
    plt.show()


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    plot_r2()
    plot_statewide_timeseries()
    plot_combined()