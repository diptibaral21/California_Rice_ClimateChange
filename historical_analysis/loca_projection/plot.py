"""

=========================================================
OVERVIEW
=========================================================
This script generates a publication-quality plot for the
LOCA multi-model ensemble of historical rice yield.

The plot compares:

    - LOCA ensemble predictions (13 climate models × 1000 runs)
    - Observed statewide yield (area-weighted)

=========================================================
WHAT THIS PLOT SHOWS
=========================================================

1. Ensemble median yield (central estimate)
2. 67% confidence interval (model spread)
3. 95% confidence interval (full uncertainty)
4. Observed yield (ground truth)

=========================================================
WHY THIS IS IMPORTANT
=========================================================

This figure is used to:

    - validate model performance
    - assess uncertainty from:
        • climate models
        • statistical models
    - demonstrate how well simulations reproduce
      historical yield variability

=========================================================
INPUT FILES
=========================================================

1. Ensemble summary (from ensemble script):
    loca_13model_historical_ensemble_summary.csv

    Contains:
        year, mean, median, p2_5, p16_5, p83_5, p97_5

2. Observed yield (county-level):
    rice_yield_1979_2023.csv

3. County rice area:
    county_rice_area_static.csv

=========================================================
OUTPUT
=========================================================

Plot:
    loca_ensemble_historical.png

=========================================================
KEY METHOD DETAILS
=========================================================

Observed yield is converted to statewide yield using:

    Statewide Yield =
        Σ (Yield_county × Area_county) / Σ Area_county

This ensures consistency with simulated outputs,
which are also area-weighted.

=========================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# PATHS
# =========================================================
CLIMATE_DIR = "/group/moniergrp/dbaral"
PROJECT_DIR = os.path.join(CLIMATE_DIR, "run_project")

ensemble_dir = os.path.join(
    PROJECT_DIR,
    "output_data",
    "projection",
    "loca_hist_ensemble"
)

input_dir = os.path.join(PROJECT_DIR, "input_data")

observed_file = os.path.join(
    input_dir,
    "yield",
    "rice_yield_1979_2023.csv"
)

area_file = os.path.join(
    input_dir,
    "rice_area",
    "county_rice_area_static.csv"
)


# =========================================================
# SETTINGS
# =========================================================
scenario = "historical"

ensemble_file = os.path.join(
    ensemble_dir,
    f"loca_13model_{scenario}_ensemble_summary.csv"
)


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
# LOAD OBSERVED (AREA-WEIGHTED)
# =========================================================
def compute_observed_statewide():
    """
    Convert county-level observed yield into statewide
    area-weighted yield.

    Returns:
        DataFrame:
            year, observed_yield
    """

    obs = pd.read_csv(observed_file)
    area = pd.read_csv(area_file)

    # Clean county names for safe merge
    obs["county"] = obs["county"].str.lower().str.strip()
    area["county"] = area["county"].str.lower().str.strip()

    df = obs.merge(area, on="county")

    # Compute weighted yield
    statewide = (
        df.groupby("year")
        .apply(lambda x: np.sum(x["yield_kg_ha"] * x["rice_area_ha"]) / np.sum(x["rice_area_ha"]))
        .reset_index(name="observed_yield")
    )

    return statewide


# =========================================================
# MAIN PLOT FUNCTION
# =========================================================
def plot_ensemble():
    """
    Generate LOCA ensemble plot.

    Includes:
        - median prediction
        - 67% CI
        - 95% CI
        - observed yield
    """

    # -----------------------------------------------------
    # Debug check (prevents silent failure)
    # -----------------------------------------------------
    print("Looking for ensemble file:")
    print(ensemble_file)

    if not os.path.exists(ensemble_file):
        print("\nERROR: File not found!")
        print("Available files:")
        print(os.listdir(ensemble_dir))
        raise FileNotFoundError("Fix filename or rerun ensemble step.")

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------
    df = pd.read_csv(ensemble_file)
    obs = compute_observed_statewide()

    df = df.merge(obs, on="year")

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------
    plt.figure(figsize=(7, 4.5))

    # 95% CI (light band)
    plt.fill_between(
        df["year"],
        df["p2_5"],
        df["p97_5"],
        color="steelblue",
        alpha=0.2,
        label="95% CI"
    )

    # 67% CI (darker band)
    plt.fill_between(
        df["year"],
        df["p16_5"],
        df["p83_5"],
        color="steelblue",
        alpha=0.4,
        label="67% CI"
    )

    # Ensemble median
    plt.plot(
        df["year"],
        df["median"],
        color="navy",
        linewidth=2,
        label="LOCA Ensemble"
    )

    # Observed yield
    plt.plot(
        df["year"],
        df["observed_yield"],
        color="black",
        linestyle="--",
        linewidth=2,
        label="Observed"
    )

    # -----------------------------------------------------
    # Labels and formatting
    # -----------------------------------------------------
    plt.ylabel("Statewide Yield (kg/ha)")
    plt.xlabel("Year")

    plt.legend(frameon=False)

    plt.tight_layout()

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------
    out_fp = os.path.join(
        ensemble_dir,
        f"loca_ensemble_{scenario}.png"
    )

    plt.savefig(out_fp, dpi=300)
    plt.show()

    print("\nSaved plot:", out_fp)


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    plot_ensemble()