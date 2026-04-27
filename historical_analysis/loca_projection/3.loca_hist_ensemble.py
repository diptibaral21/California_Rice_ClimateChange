"""

=========================================================
OVERVIEW
=========================================================
This script builds a multi-model ensemble of historical
rice yield simulations using LOCA climate datasets.

Each LOCA climate model produces:
    - 995 statistical yield simulations (from Lasso model)
    - statewide area-weighted yield

This script combines outputs from all LOCA models into
a single ensemble dataset.

=========================================================
ENSEMBLE STRUCTURE
=========================================================
Total ensemble members:

    13 LOCA climate models × 995 statistical models

Each ensemble member represents:
    - one climate realization
    - one statistical model realization

=========================================================
WHY THIS STEP IS IMPORTANT
=========================================================
This step captures two key sources of uncertainty:

1. Climate model uncertainty (LOCA models)
2. Statistical model uncertainty (Lasso ensemble)

The resulting ensemble provides:
    - robust central estimates
    - uncertainty bounds (67% and 95%)

=========================================================
INPUT FILES
=========================================================
Each LOCA model output file:

    {model}_{ssp}_statewide_all_1000.csv

Contains:
    - year
    - pred_iter_0 ... pred_iter_999

=========================================================
OUTPUT FILES
=========================================================
1. Full ensemble:
    loca_13model_{ssp}_all_iterations.csv

2. Ensemble summary:
    loca_13model_{ssp}_ensemble_summary.csv

=========================================================
SUMMARY METRICS
=========================================================
For each year:
    - mean
    - median
    - 2.5th percentile (lower 95% CI)
    - 16.5th percentile (lower 67% CI)
    - 83.5th percentile (upper 67% CI)
    - 97.5th percentile (upper 95% CI)

=========================================================
"""

import os
import numpy as np
import pandas as pd

# =========================================================
# PATHS
# =========================================================
CLIMATE_DIR = "/group/moniergrp/dbaral"
PROJECT_DIR = os.path.join(CLIMATE_DIR, "run_project")

loca_output_dir = os.path.join(
    PROJECT_DIR,
    "output_data",
    "projection",
    "loca_hist"
)

ensemble_output_dir = os.path.join(
    PROJECT_DIR,
    "output_data",
    "projection",
    "loca_hist_ensemble"
)
os.makedirs(ensemble_output_dir, exist_ok=True)

# =========================================================
# SETTINGS
# =========================================================
ssp = "historical"  

model_list = [
    "ACCESS-CM2","CNRM-ESM2-1","EC-Earth3","EC-Earth3-Veg",
    "FGOALS-g3","GFDL-ESM4","HadGEM3-GC31-LL","INM-CM5-0",
    "IPSL-CM6A-LR","KACE-1-0-G","MIROC6","MPI-ESM1-2-HR","MRI-ESM2-0"
]

# =========================================================
# BUILD ENSEMBLE
# =========================================================
def build_ensemble():
    """
    Load all LOCA model outputs and merge into a single ensemble.

    Steps:
    ------
    1. Read each LOCA model file
    2. Rename prediction columns to avoid overlap
    3. Merge all models by year
    4. Return combined dataset

    Returns:
    --------
    DataFrame:
        year + all ensemble prediction columns
    """

    merged_dfs = []

    for model in model_list:

        fp = os.path.join(
            loca_output_dir,
            f"{model}_{ssp}_statewide_all_1000.csv"
        )

        if not os.path.exists(fp):
            print(f"Skipping missing file: {fp}")
            continue

        df = pd.read_csv(fp)

        pred_cols = [c for c in df.columns if c.startswith("pred_iter_")]

        # Rename to keep unique across models
        rename_dict = {col: f"{model}_{col}" for col in pred_cols}
        df = df.rename(columns=rename_dict)

        df = df[["year"] + list(rename_dict.values())]

        merged_dfs.append(df)

        print(f"Loaded {model} → {len(pred_cols)} members")

    if len(merged_dfs) == 0:
        raise ValueError("No LOCA files found.")

    ensemble_df = merged_dfs[0]

    for df_next in merged_dfs[1:]:
        ensemble_df = ensemble_df.merge(df_next, on="year")

    ensemble_df = ensemble_df.sort_values("year").reset_index(drop=True)

    return ensemble_df


# =========================================================
# SUMMARY
# =========================================================
def compute_summary(ensemble_df):
    """
    Compute ensemble statistics across all models.

    Returns:
    --------
    DataFrame:
        year + summary statistics
    """

    pred_cols = [c for c in ensemble_df.columns if c != "year"]

    summary = pd.DataFrame({
        "year": ensemble_df["year"],
        "mean": ensemble_df[pred_cols].mean(axis=1),
        "median": ensemble_df[pred_cols].median(axis=1),
        "p2_5": ensemble_df[pred_cols].quantile(0.025, axis=1),
        "p16_5": ensemble_df[pred_cols].quantile(0.165, axis=1),
        "p83_5": ensemble_df[pred_cols].quantile(0.835, axis=1),
        "p97_5": ensemble_df[pred_cols].quantile(0.975, axis=1),
    })

    return summary


# =========================================================
# SAVE
# =========================================================
def save_outputs(ensemble_df, summary_df):

    all_fp = os.path.join(
        ensemble_output_dir,
        f"loca_13model_{ssp}_all_iterations.csv"
    )

    summary_fp = os.path.join(
        ensemble_output_dir,
        f"loca_13model_{ssp}_ensemble_summary.csv"
    )

    ensemble_df.to_csv(all_fp, index=False)
    summary_df.to_csv(summary_fp, index=False)

    print("\nSaved:")
    print(all_fp)
    print(summary_fp)


# =========================================================
# MAIN
# =========================================================
def main():

    print("Building LOCA ensemble...")

    ensemble_df = build_ensemble()

    summary_df = compute_summary(ensemble_df)

    save_outputs(ensemble_df, summary_df)

    print("Done.")


if __name__ == "__main__":
    main()