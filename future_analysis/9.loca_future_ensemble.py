"""
=========================================================
LOCA FUTURE MULTI-MODEL ENSEMBLE (STATEWIDE YIELD)
=========================================================

This script constructs a multi-model ensemble of projected
rice yield using LOCA climate projections and Lasso-based
yield simulations.

=========================================================
OBJECTIVE
=========================================================

To quantify uncertainty in future rice yield projections by
combining:

    1. Model uncertainty (13 LOCA GCMs)
    2. Statistical uncertainty (1000 Lasso iterations)

This results in a large ensemble:

    TOTAL MEMBERS = 13 models × 1000 iterations

=========================================================
WHY THIS STEP IS CRITICAL
=========================================================

Individual model projections are insufficient because:

- Climate models differ in their projections
- Statistical models contain parameter uncertainty

This ensemble framework captures both sources of uncertainty
simultaneously, enabling robust inference.

=========================================================
INPUT REQUIREMENTS
=========================================================

For each:
    LOCA model × SSP × trend_mode

The script expects:

    {model}_{ssp}_{trend}_statewide_all_1000.csv

Each file must contain:
    - year column
    - pred_iter_0 ... pred_iter_999

These represent:
    → statewide yield per iteration

=========================================================
ENSEMBLE CONSTRUCTION
=========================================================

Step 1: Load all model outputs
Step 2: Rename iteration columns to avoid collision
Step 3: Merge across models on year
Step 4: Stack all ensemble members

Each column in final dataset represents:

    one realization = (model × iteration)

=========================================================
SUMMARY STATISTICS
=========================================================

For each year, compute:

    - Mean
    - Median
    - 67% CI  (16.5–83.5 percentiles)
    - 95% CI  (2.5–97.5 percentiles)

=========================================================
IMPORTANT DESIGN CHOICES
=========================================================

1. Ensemble is built using ALL iterations
   → avoids averaging too early

2. Statewide aggregation is done BEFORE ensemble
   → ensures proper weighting

3. Years retained: 2020–2100
   → overlap with historical retained internally
   → plotting can start from 2024 if desired

=========================================================
OUTPUTS
=========================================================

For each SSP × trend:

1. Full ensemble:
    loca_13model_{ssp}_{trend}_all_iterations.csv

2. Summary:
    loca_13model_{ssp}_{trend}_ensemble_summary.csv

3. Comparison plot:
    loca_ensemble_trend_comparison.png

=========================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PATHS
# =========================================================
CLIMATE_DIR = "/group/moniergrp/dbaral"
PROJECT_DIR = os.path.join(CLIMATE_DIR, "run_project")

loca_output_dir = os.path.join(
    PROJECT_DIR,
    "output_data",
    "projection",
    "loca_future"
)

ensemble_output_dir = os.path.join(
    PROJECT_DIR,
    "output_data",
    "projection",
    "loca_future_ensemble"
)

os.makedirs(ensemble_output_dir, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================
ssp_list = ["ssp245", "ssp585"]

model_list = [
    "ACCESS-CM2", "CNRM-ESM2-1", "EC-Earth3", "EC-Earth3-Veg",
    "FGOALS-g3", "GFDL-ESM4", "HadGEM3-GC31-LL",
    "INM-CM5-0", "IPSL-CM6A-LR", "KACE-1-0-G",
    "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0"
]

trend_list = ["sustained", "fixed"]


# =========================================================
# BUILD ENSEMBLE
# =========================================================


def build_loca_future_ensemble(model_list, ssp, trend_mode):
    """
    Build ensemble for a given SSP and trend using STATEWIDE data.

    Returns:
        ensemble_all_df   → full ensemble (all members)
        ensemble_summary  → summary statistics
    """

    merged_dfs = []

    for model in model_list:

        # use STATEWIDE iteration files
        filename = f"{model}_{ssp}_{trend_mode}_statewide_all_1000.csv"
        fp = os.path.join(loca_output_dir, filename)

        if not os.path.exists(fp):
            print(f"[WARNING] Missing file: {fp}")
            continue

        df = pd.read_csv(fp)

        # Identify iteration columns
        pred_cols = [c for c in df.columns if c.startswith("pred_iter_")]

        if len(pred_cols) == 0:
            raise ValueError(f"No pred_iter columns in {fp}")

        # Rename to keep models separate
        rename_dict = {col: f"{model}_{col}" for col in pred_cols}
        df = df.rename(columns=rename_dict)

        df = df[["year"] + list(rename_dict.values())]

        merged_dfs.append(df)

        print(f"[INFO] Loaded {model} ({ssp}, {trend_mode})")

    if len(merged_dfs) == 0:
        raise ValueError(f"No valid files found for {ssp}-{trend_mode}")

    ensemble_all_df = merged_dfs[0]
    for df_next in merged_dfs[1:]:
        ensemble_all_df = ensemble_all_df.merge(df_next, on="year", how="inner")

    ensemble_all_df = ensemble_all_df.sort_values("year").reset_index(drop=True)

    pred_cols = [c for c in ensemble_all_df.columns if c != "year"]

    # Summary statistics
    ensemble_summary_df = pd.DataFrame({
        "year": ensemble_all_df["year"],
        "mean": ensemble_all_df[pred_cols].mean(axis=1),
        "median": ensemble_all_df[pred_cols].median(axis=1),
        "p2_5": ensemble_all_df[pred_cols].quantile(0.025, axis=1),
        "p16_5": ensemble_all_df[pred_cols].quantile(0.165, axis=1),
        "p83_5": ensemble_all_df[pred_cols].quantile(0.835, axis=1),
        "p97_5": ensemble_all_df[pred_cols].quantile(0.975, axis=1),
    })

    return ensemble_all_df, ensemble_summary_df
# =========================================================
# SAVE OUTPUTS
# =========================================================
def save_outputs(all_df, summary_df, ssp, trend):

    all_fp = os.path.join(
        ensemble_output_dir,
        f"loca_13model_{ssp}_{trend}_all_iterations.csv"
    )

    summary_fp = os.path.join(
        ensemble_output_dir,
        f"loca_13model_{ssp}_{trend}_ensemble_summary.csv"
    )

    all_df.to_csv(all_fp, index=False)
    summary_df.to_csv(summary_fp, index=False)

    print(f"[INFO] Saved {ssp} - {trend}")


# =========================================================
# MAIN
# =========================================================
def main():

    print("\n=== BUILDING LOCA ENSEMBLE ===")

    all_summaries = {}

    for ssp in ssp_list:

        all_summaries[ssp] = {}

        for trend in trend_list:

            print(f"\nProcessing {ssp} - {trend}")

            all_df, summary_df = build_loca_future_ensemble(
                model_list,
                ssp,
                trend
            )

            save_outputs(all_df, summary_df, ssp, trend)

            all_summaries[ssp][trend] = summary_df

    print("\n=== DONE ===")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    main()