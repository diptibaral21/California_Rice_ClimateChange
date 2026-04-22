"""

=========================================================
OVERVIEW
=========================================================
This script simulates historical rice yield using trained 
Lasso models and computes both:

    1. County-level predicted yields
    2. Statewide area-weighted yields

The simulation uses:
    - Model coefficients from 1000 Lasso iterations
    - Training normalization parameters (metadata)

=========================================================
WHY THIS STEP IS IMPORTANT
=========================================================
This step bridges model training and real-world interpretation.

It allows us to:
    - Validate model performance (predicted vs observed yield)
    - Understand spatial variability in yield
    - Aggregate results to meaningful regional scales
    - Quantify uncertainty using ensemble predictions

=========================================================
KEY CONCEPT: AREA-WEIGHTED YIELD
=========================================================
Rice production is not evenly distributed across counties.

Simple averaging would incorrectly assume:
    - each county contributes equally

Instead, we compute:

    Statewide Yield = Σ (Yield_county × Area_county) / Σ Area_county

This ensures:
    - counties with larger rice area have greater influence
    - results reflect actual production patterns

=========================================================
METHOD
=========================================================
1. Build prediction design matrix using training normalization
2. Load 1000 sets of model coefficients
3. Compute predictions using matrix multiplication
4. Generate:
    - county-level predicted yields
    - statewide area-weighted yield per iteration
5. Compute ensemble statistics:
    - mean prediction
    - 67% confidence interval
    - 95% confidence interval

=========================================================
OUTPUTS
=========================================================
1. gridmet_simulation_county.csv
    - county-level predictions

2. gridmet_simulation_statewide.csv
    - statewide area-weighted yield with uncertainty bands
"""

import os
import json
import numpy as np
import pandas as pd
import sys

# =========================================================
# PATHS
# =========================================================
CLIMATE_DIR = "/group/moniergrp/dbaral"

input_path = os.path.join(CLIMATE_DIR, "run_project/input_data/gridmet_hist_model_input")
output_path = os.path.join(CLIMATE_DIR, "run_project/output_data/historical_model")
area_path = os.path.join(CLIMATE_DIR, "run_project/input_data/rice_area/county_rice_area_static.csv")

# =========================================================
# IMPORT FUNCTION
# =========================================================
#add the dir that contains the function that you want to import to the sys path
function_dir = os.path.join(CLIMATE_DIR, "California_Rice_ClimateChange/historical_analysis/gridmet")
sys.path.append(function_dir)
from lasso_model import build_design_matrix_for_prediction


# =========================================================
# MAIN SIMULATION FUNCTION
# =========================================================
def simulate():

    # -----------------------------------------------------
    # Step 1: Load input data
    # -----------------------------------------------------
    df = pd.read_csv(os.path.join(input_path, "Lasso_Model_Input_Variables_1979_2023.csv"))

    # -----------------------------------------------------
    # Step 2: Load metadata (normalization parameters)
    # -----------------------------------------------------
    metadata = json.load(open(os.path.join(output_path, "model_metadata.json")))

    # -----------------------------------------------------
    # Step 3: Build prediction design matrix
    # -----------------------------------------------------
    df_model, X = build_design_matrix_for_prediction(df, metadata)

    # -----------------------------------------------------
    # Step 4: Load model coefficients
    # -----------------------------------------------------
    coef_df = pd.read_csv(os.path.join(output_path, "final_coefficients.csv"))

    coef_pivot = coef_df.pivot(
        index="iteration",
        columns="feature",
        values="coefficient"
    )
    coef_pivot = coef_pivot[X.columns]
    coef_mat = coef_pivot.values   # shape: (iterations, features)

    # -----------------------------------------------------
    # Step 5: Generate predictions
    # -----------------------------------------------------
    X_mat = X.values               # shape: (observations, features)

    preds = coef_mat @ X_mat.T     # shape: (iterations, observations)

    # -----------------------------------------------------
    # Step 6: County-level outputs
    # -----------------------------------------------------
    df_model["pred_mean"] = preds.mean(axis=0)

    # Save county-level predictions
    df_model.to_csv(
        os.path.join(output_path, "gridmet_simulation_county.csv"),
        index=False
    )

    # -----------------------------------------------------
    # Step 7: Load area data
    # -----------------------------------------------------
    area_df = pd.read_csv(area_path)

    df_model = df_model.merge(area_df, on="county", how="left")

    # -----------------------------------------------------
    # Step 8: Compute statewide area-weighted yield
    # -----------------------------------------------------
    years = df_model["year"].unique()
    iterations = preds.shape[0]

    statewide_results = []

    for year in sorted(years):

        df_year = df_model[df_model["year"] == year]
        idx = df_year.index

        # predictions for this year across all iterations
        preds_year = preds[:, idx]   # (iterations x counties)

        area = df_year["rice_area_ha"].values

        # weighted yield per iteration
        weighted_yield = np.sum(preds_year * area, axis=1) / np.sum(area)

        statewide_results.append({
            "year": year,
            "mean": np.mean(weighted_yield),
            "p67_low": np.percentile(weighted_yield, 16.5),
            "p67_high": np.percentile(weighted_yield, 83.5),
            "p95_low": np.percentile(weighted_yield, 2.5),
            "p95_high": np.percentile(weighted_yield, 97.5)
        })

    statewide_df = pd.DataFrame(statewide_results)

    # -----------------------------------------------------
    # Step 9: Save statewide results
    # -----------------------------------------------------
    statewide_df.to_csv(
        os.path.join(output_path, "gridmet_simulation_statewide.csv"),
        index=False
    )

    print("Simulation complete: county + statewide outputs saved.")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    simulate()