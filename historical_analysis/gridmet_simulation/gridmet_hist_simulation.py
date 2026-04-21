"""

=========================================================
OVERVIEW
=========================================================
This script simulates historical rice yield using 
trained Lasso models.

It uses:
    - Model coefficients (1000 iterations)
    - Training normalization parameters

=========================================================
WHY THIS STEP IS IMPORTANT
=========================================================
This allows us to:
    - validate model performance
    - compare predicted vs observed yield
    - generate uncertainty bounds

=========================================================
METHOD
=========================================================
1. Build design matrix using training normalization
2. Load all coefficient sets
3. Compute predictions using matrix multiplication
4. Aggregate results into:
    - mean prediction
    - uncertainty intervals

=========================================================
OUTPUT
=========================================================
gridmet_simulation.csv
"""

import os
import json
import numpy as np
import pandas as pd
from lasso_model import build_design_matrix_for_prediction

CLIMATE_DIR = "/group/moniergrp/dbaral"

input_path = os.path.join(CLIMATE_DIR, "run_project/input_data/gridmet_hist_model_input")
output_path = os.path.join(CLIMATE_DIR, "run_project/output_data/historical_model")


def simulate():

    df = pd.read_csv(os.path.join(input_path, "Lasso_Model_Input_Variables_1979_2023.csv"))

    metadata = json.load(open(os.path.join(output_path, "model_metadata.json")))

    df_model, X = build_design_matrix_for_prediction(df, metadata)

    coef_df = pd.read_csv(os.path.join(output_path, "final_coefficients.csv"))
    coef_mat = coef_df.pivot(index="iteration", columns="feature", values="coefficient").values

    preds = coef_mat @ X.values.T

    df_model["pred_mean"] = preds.mean(axis=0)

    df_model.to_csv(os.path.join(output_path, "gridmet_simulation.csv"), index=False)


if __name__ == "__main__":
    simulate()