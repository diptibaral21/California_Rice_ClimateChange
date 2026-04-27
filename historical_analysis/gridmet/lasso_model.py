"""

=========================================================
OVERVIEW
=========================================================
This script builds a Lasso regression model to explain 
historical rice yield variability across counties in 
California using temperature-derived climate indices 
(from GridMET data).

The goal of this modeling framework is to:
    1. Quantify how temperature variability influences yield
    2. Capture spatial heterogeneity across counties
    3. Account for long-term yield trends
    4. Allow nonlinear climate responses

This script represents the TRAINING phase of a larger workflow:
    - Training (this script)
    - Aggregation of model outputs
    - Simulation using GridMET (historical validation)
    - Simulation using LOCA (historical + future projections)

IMPORTANT:
----------
This script is ONLY used to TRAIN the model.

Prediction and simulation MUST use:
    build_design_matrix_for_prediction()

=========================================================
MODEL STRUCTURE
=========================================================
The model is specified as:

    Yield =
        County Fixed Effects
      + County-Specific Time Trends
      + Climate Variables (Normalized)
      + Squared Climate Variables (Normalized)

Each component serves a purpose:

1. County Fixed Effects (FE):
    - Captures time-invariant differences across counties
    - Includes soil, management practices, irrigation, etc.
    - No intercept is used → FE represent baseline yield

2. County-Specific Time Trends:
    - Captures long-term improvements or decline
    - Accounts for technology, adaptation, policy shifts

3. Climate Variables:
    - Represent temperature conditions during key growth stages

4. Squared Climate Terms:
    - Capture nonlinear responses such as:
        * heat stress (extreme high temps)
        * cold damage (extreme low temps)

=========================================================
KEY MODELING CHOICES
=========================================================

- No intercept:
    All county dummies are included -- avoids redundancy

- Climate normalization:
    Climate variables are standardized so that:
        - coefficients are comparable across variables
        - county FE represent yield under average climate

- Squared terms:
    Allows flexible nonlinear relationships

- Cook's Distance Filtering:
    Removes influential observations that could:
        - distort coefficients
        - bias model selection
        - reduce robustness

- County-wise 70-30 split:
    Ensures:
        - each county appears in both train and test
        - avoids spatial leakage

- Repeated training (SLURM array):
    - Each iteration uses a different random split
    - Used to quantify:
        - coefficient stability
        - predictive uncertainty



=========================================================
OUTPUTS
=========================================================

1. array_results_temp/
    - coef_*.csv -- coefficients per iteration
    - metrics_*.json -- model performance per iteration

2. model_metadata.json
    - Contains:
        * feature order
        * normalization parameters
        * base year
    - REQUIRED for simulation phase

3. gridmet_design_matrix_full.csv
    - BEFORE outlier removal
    - Used for:
        - debugging
        - reproducibility
        - comparison with LOCA data

4. gridmet_design_matrix_filtered.csv
    - AFTER Cook's filtering
    - Used for model training

=========================================================
USAGE
=========================================================

Run via SLURM array:

    sbatch --array=1-1000 

Each task runs one iteration with a different random split.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm


# =========================================================
# PATHS
# =========================================================
CLIMATE_DIR = "/group/moniergrp/dbaral"
input_path = os.path.join(CLIMATE_DIR, "run_project/input_data/gridmet_hist_model_input")
output_path = os.path.join(CLIMATE_DIR, "run_project/output_data/historical_model")
os.makedirs(output_path, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================
def load_raw_data():
    """
    Load historical GridMET dataset.

    Returns:
        df (DataFrame):
            Contains:
                - county
                - year
                - yield_kg_ha
                - climate variables
    """
    return pd.read_csv(os.path.join(input_path, "Lasso_Model_Input_Variables_1979_2023.csv"))


# =========================================================
# BUILD DESIGN MATRIX (TRAINING)
# =========================================================
def build_design_matrix(df):

    df_model = df[df["yield_kg_ha"].notna()].copy()

    cols_exclude = ["county", "year", "yield_kg_ha"]
    climate_cols = [c for c in df_model.columns if c not in cols_exclude]

    # Create squared terms
    for col in climate_cols:
        df_model[f"{col}_sq"] = df_model[col] ** 2

    # Normalize
    means, stds = {}, {}
    for col in climate_cols + [f"{c}_sq" for c in climate_cols]:
        means[col] = float(df_model[col].mean())
        stds[col] = float(df_model[col].std())
        df_model[col] = (df_model[col] - means[col]) / stds[col]

    # Fixed effects
    dummies = pd.get_dummies(df_model["county"], prefix="county").astype(float)

    # Trends
    base_year = int(df_model["year"].min())
    trend = df_model["year"] - base_year

    trends = dummies.multiply(trend, axis=0)
    trends.columns = [c.replace("county", "trend") for c in trends.columns]

    X = pd.concat(
        [df_model[climate_cols + [f"{c}_sq" for c in climate_cols]], dummies, trends],
        axis=1
    ).astype(float)

    X.to_csv(os.path.join(output_path, "gridmet_design_matrix_full.csv"), index=False)

    return df_model, X, df_model["yield_kg_ha"], list(X.columns), means, stds, base_year


# =========================================================
# REMOVE OUTLIERS
# =========================================================
def remove_outliers(df_model, X, Y):

    X_const = sm.add_constant(X).astype(float)

    model = sm.OLS(Y, X_const).fit()
    cooks_d = model.get_influence().cooks_distance[0]

    mask = cooks_d < (4 / len(Y))

    X_f = X.loc[mask].astype(float)
    df_f = df_model.loc[mask]
    Y_f = Y.loc[mask]

    X_f.to_csv(os.path.join(output_path, "gridmet_design_matrix_filtered.csv"), index=False)

    return df_f, X_f, Y_f


# =========================================================
# SAVE METADATA
# =========================================================
def save_metadata(features, means, stds, base_year):

    metadata = {
        "feature_cols_final": features,
        "climate_norm_means": means,
        "climate_norm_stds": stds,
        "base_year": base_year
    }

    json.dump(metadata, open(os.path.join(output_path, "model_metadata.json"), "w"), indent=2)


# =========================================================
# RUN ITERATION (ORIGINAL LOGIC RESTORED)
# =========================================================
def run_iteration(X, Y, df, features, iteration_id):

    # Alpha grid (original)
    alphas = [5, 4, 3, 2, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]

    train_idx, test_idx = [], []

    for c in df["county"].unique():
        idx = df.index[df["county"] == c]

        tr, te = train_test_split(
            idx,
            test_size=0.3,
            random_state=iteration_id
        )

        train_idx.extend(tr)
        test_idx.extend(te)

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = Y.loc[train_idx]
    y_test = Y.loc[test_idx]

    model = LassoCV(
        alphas=alphas,
        cv=5,
        fit_intercept=False,
        max_iter=int(1e7),
        random_state=45
    )

    model.fit(X_train, y_train)

    rmse_train = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
    rmse_test = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

    temp = os.path.join(output_path, "array_results_temp")
    os.makedirs(temp, exist_ok=True)

    pd.DataFrame({
        "iteration": iteration_id,
        "feature": features,
        "coefficient": model.coef_,
        "alpha_selected": model.alpha_
    }).to_csv(os.path.join(temp, f"coef_{iteration_id}.csv"), index=False)

    json.dump({
        "iteration": iteration_id,
        "R2_train": model.score(X_train, y_train),
        "R2_test": model.score(X_test, y_test),
        "RMSE_train": float(rmse_train),
        "RMSE_test": float(rmse_test)
    }, open(os.path.join(temp, f"metrics_{iteration_id}.json"), "w"), indent=2)


# =========================================================
# PREDICTION DESIGN MATRIX
# =========================================================
def build_design_matrix_for_prediction(df, metadata):

    df_model = df.copy()

    cols_exclude = ["county", "year", "yield_kg_ha"]
    climate_cols = [c for c in df_model.columns if c not in cols_exclude]

    for col in climate_cols:
        df_model[f"{col}_sq"] = df_model[col] ** 2

    for col in climate_cols + [f"{c}_sq" for c in climate_cols]:
        df_model[col] = (
            df_model[col] - metadata["climate_norm_means"][col]
        ) / metadata["climate_norm_stds"][col]

    dummies = pd.get_dummies(df_model["county"], prefix="county").astype(float)

    trend = df_model["year"] - metadata["base_year"]

    trends = dummies.multiply(trend, axis=0)
    trends.columns = [c.replace("county", "trend") for c in trends.columns]

    X = pd.concat(
        [df_model[climate_cols + [f"{c}_sq" for c in climate_cols]], dummies, trends],
        axis=1
    ).astype(float)

    X = X[metadata["feature_cols_final"]]

    return df_model, X


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    iteration_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))

    df = load_raw_data()

    df_model, X, Y, features, means, stds, base_year = build_design_matrix(df)

    save_metadata(features, means, stds, base_year)

    df_f, X_f, Y_f = remove_outliers(df_model, X, Y)

    run_iteration(X_f, Y_f, df_f, features, iteration_id)

    print(f"Iteration {iteration_id} complete.")