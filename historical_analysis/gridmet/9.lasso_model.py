import os
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import joblib
import sys


# =========================================================
# File paths
# =========================================================

CLIMATE_DIR = "/group/moniergrp/dbaral"
file_path = os.path.join(CLIMATE_DIR, "run_project/input_data/gridmet_hist_model_input")
save_path = os.path.join(CLIMATE_DIR, "run_project/output_data/historical_model")
os.makedirs(save_path, exist_ok=True)

# =========================================================
# Helper function 1
# Load raw data
# =========================================================

def load_raw_data():
    """
    Load the historical gridMET model input dataset.

    Returns:
        df = raw dataframe
    """
    df = pd.read_csv(os.path.join(file_path, "Lasso_Model_Input_Variables_1979_2023.csv"))
    return df


# =========================================================
# Helper function 2
# Build design matrix
# =========================================================

def build_design_matrix(df, file_prefix="gridmet"):
    """
    Build the design matrix explicitly for a no-intercept Lasso model.

    Model structure:
        Yield =
            county fixed effects
          + county-specific time trends
          + normalize climate variables
          + squared normalized climate variables

    Important modeling choices:
        1. No separate intercept in the model
        2. All county dummy variables are included
        3. County coefficients act as county fixed effects
        4. Time trend starts at 0, not 1
        5. Only climate variables are normalized
        6. County dummies and county trends are NOT scaled

    Why normalize climate variables:
        This makes county fixed effects easier to interpret.
        County FE then represent county baseline yield under average climate conditions.

    Returns:
        df_model             = dataframe used to build X and Y
        X_df                 = final design matrix dataframe
        Y                    = response series
        feature_cols_final   = final feature names in order
        climate_cols         = original climate variable names
        climate_norm_means   = means used to normalize climate variables
        climate_norm_stds    = stds used for normalization
        base_year            = base year used for time trends
    """

    df_model = df.copy()

    # -----------------------------------------------------
    # Step 1: remove rows where yield is missing
    # -----------------------------------------------------
    df_model = df_model[df_model["yield_kg_ha"].notna()].copy()

    # -----------------------------------------------------
    # Step 2: define base climate predictor columns
    # These are the continuous climate predictors only
    # -----------------------------------------------------
    cols_exclude = ["county", "year", "yield_kg_ha"]
    climate_cols = [c for c in df_model.columns if c not in cols_exclude]

    # -----------------------------------------------------
    # Step 3: create squared terms 
    # -----------------------------------------------------
    squared_cols = []

    for col in climate_cols:
        sq_col = f"{col}_sq"
        df_model[sq_col] = df_model[col] ** 2
        squared_cols.append(sq_col)

    # -----------------------------------------------------
    # Step 4: Normalize linear + squared variables separately
    # -----------------------------------------------------
    climate_norm_means = {}
    climate_norm_stds = {}

    all_climate_cols = climate_cols + squared_cols

    for col in all_climate_cols:
        mean_val = df_model[col].mean()
        std_val = df_model[col].std()

        climate_norm_means[col] = float(mean_val)
        climate_norm_stds[col] = float(std_val)

        df_model[col] = (df_model[col]- mean_val) / std_val

    # -----------------------------------------------------
    # Step 5: create all county dummy variables
    # Keep all counties because we are fitting with no intercept
    # -----------------------------------------------------
    county_dummies = pd.get_dummies(
        df_model["county"],
        prefix="county",
        drop_first=False
    ).astype(int)

    # -----------------------------------------------------
    # Step 6: create county-specific time trends
    # IMPORTANT:
    # trend starts at 0 so county coefficient stays as baseline FE
    # -----------------------------------------------------
    base_year = int(df_model["year"].min())
    time_trend = df_model["year"] - base_year

    county_trends = county_dummies.multiply(time_trend, axis=0)
    county_trends.columns = [c.replace("county", "trend") for c in county_trends.columns]

    # -----------------------------------------------------
    # Step 7: combine all components into final design matrix
    # Final X includes:
    #   - normalized climate variables
    #   - squared normalized climate variables
    #   - county fixed effects
    #   - county-specific time trends
    # -----------------------------------------------------
    X_df = pd.concat(
        [
            df_model[climate_cols + squared_cols],
            county_dummies,
            county_trends
        ],
        axis=1
    )

    X_df.to_csv(os.path.join(save_path, f"{file_prefix}_tis_v1.csv"), index=False)

    # -----------------------------------------------------
    # Step 8: define response
    # -----------------------------------------------------
    Y = df_model["yield_kg_ha"].copy()

    # -----------------------------------------------------
    # Step 9: save final feature order
    # This order must be preserved later for prediction
    # -----------------------------------------------------
    feature_cols_final = list(X_df.columns)

    return (
        df_model,
        X_df,
        Y,
        feature_cols_final,
        climate_cols,
        climate_norm_means,
        climate_norm_stds,
        base_year
    )

# =========================================================
# Helper function 3
# Remove influential outliers using Cook's distance
# =========================================================

def remove_outliers_with_cooks_distance(df_model, X_df, Y, file_prefix="gridmet"):
    """
    Remove influential observations using Cook's distance.

    Why:
        Highly influential points can distort:
            - coefficient estimates
            - selected alpha
            - stability results

    Steps:
        1. Fit OLS only for influence diagnostics
        2. Compute Cook's distance
        3. Apply threshold = 4 / n
        4. Keep observations below threshold

    Returns:
        df_filtered = filtered dataframe
        X_filtered  = filtered design matrix
        Y_filtered  = filtered response
        threshold   = Cook's distance threshold
    """

    # -----------------------------------------------------
    # Step 1: fit OLS on full design matrix
    # add_constant is okay here only for Cook's distance diagnostics
    # -----------------------------------------------------
    X_const = sm.add_constant(X_df)
    model_sm = sm.OLS(Y, X_const).fit()

    # -----------------------------------------------------
    # Step 2: compute Cook's distance
    # -----------------------------------------------------
    influence = model_sm.get_influence()
    cooks_d = influence.cooks_distance[0]

    # -----------------------------------------------------
    # Step 3: define rule-of-thumb threshold
    # -----------------------------------------------------
    threshold = 4 / len(Y)

    # -----------------------------------------------------
    # Step 4: keep non-influential rows
    # -----------------------------------------------------
    mask = cooks_d < threshold

    df_filtered = df_model.loc[mask].copy()
    X_filtered = X_df.loc[mask].copy()
    Y_filtered = Y.loc[mask].copy()

    X_filtered.to_csv(os.path.join(save_path, f"{file_prefix}_tis_filtered_v1.csv"), index=False)

    return df_filtered, X_filtered, Y_filtered, threshold


# =========================================================
# Helper function 4
# Run repeated county-wise 70/30 validation
# =========================================================

def run_single_iteration(
    X,
    Y,
    df_filtered,
    feature_cols_final,
    save_path,
    iteration_id
):
    """
    Run a single 70-30 county-wise validation iteration for a job array

    Why:
        Instead of trusting one single split, we repeat the model fitting 1000 times
        to assess:
            - coefficient stability
            - test R2 stability
            - train/test RMSE stability
            - selected alpha stability
            - use for future projection 
        We run the iterations as a job array, we can parallelize 1000 iterations across multiple
        cluster nodes. Each task fits one model independently, drastically reducing the total 
        execution time

    County-wise split:
        We split within county using the iteration_id as the random state. This ensures
        that every county is represented in both training and testing.

    Saves:
        1. iteration-specific coefficient csv in temporary dir
        2. iteration-specific metrics json in temporary save dir

    Returns:
        None
        The files saved in a temporary dir are merged to return the following files in next step
        final_coef_df = coefficient results across all iterations
        metrics_df    = validation metrics across all iterations
    """

    # -----------------------------------------------------
    # Step 1: define alpha grid
    # -----------------------------------------------------
    alphas = [5, 4, 3, 2, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]

    # -----------------------------------------------------
    # Step 2: get county names
    # -----------------------------------------------------
    unique_counties = sorted(df_filtered["county"].unique())

    # -----------------------------------------------------
    # Step 4: start repeated validation loop
    # -----------------------------------------------------

    train_indices = []
    test_indices = []

    # -------------------------------------------------
    # Step 4a: split within each county
    # -------------------------------------------------
    for county in unique_counties:
        county_indices = df_filtered.index[df_filtered["county"] == county]

        train_idx_c, test_idx_c = train_test_split(
            county_indices,
            test_size=0.3,
            random_state=iteration_id # seed changes per array task
        )

        train_indices.extend(train_idx_c)
        test_indices.extend(test_idx_c)

    # -------------------------------------------------
     # Step 4b: create train/test matrices
    # -------------------------------------------------
    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]
    y_train = Y.loc[train_indices]
    y_test = Y.loc[test_indices]

    # -------------------------------------------------
    # Step 4c: fit LassoCV with no intercept
    # -------------------------------------------------
    pipe = Pipeline([
        ("lasso", LassoCV(
            alphas=alphas,
            cv=5,
            max_iter=int(1e7),
            random_state=45,
            fit_intercept=False
        ))
    ])

    pipe.fit(X_train, y_train)

    lasso = pipe.named_steps["lasso"]

    # -------------------------------------------------
    # Step 4d: predictions and validation metrics
    # -------------------------------------------------
    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)

    rmse_train = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))

    r2_train = float(pipe.score(X_train, y_train))
    r2_test = float(pipe.score(X_test, y_test))

    # -------------------------------------------------
    # Step 4e: save coefficients in long format to a temporary dir
    # -------------------------------------------------
    coef_df_iteration = pd.DataFrame({
        "iteration": iteration_id,
        "feature": feature_cols_final,
        "coefficient": lasso.coef_,
        "alpha_selected": lasso.alpha_
    })

    #make temporary dir
    temp_dir = os.path.join(save_path, "array_results_temp")
    os.makedirs(temp_dir, exist_ok=True)

    coef_df_iteration.to_csv(os.path.join(temp_dir, f"coef_iter_{iteration_id}_v1.csv"), index=False)

    # -------------------------------------------------
    # Step 4f: save metrics for this iteration
    # -------------------------------------------------
    metrics_list = {
        "iteration": iteration_id,
        "alpha_selected": float(lasso.alpha_),
        "R2_train": r2_train,
        "R2_test": r2_test,
        "RMSE_train": rmse_train,
        "RMSE_test": rmse_test
    }

    with open(os.path.join(temp_dir, f"metrics_iter_{iteration_id}_v1.json"), "w") as f:
        json.dump(metrics_list, f, indent=2)

# =========================================================
# Helper function 5
# Aggregate array results
# =========================================================

def aggregate_array_results(save_path):
    """ 
    Combine individual iteration files into final datasets
    Why:
        After the job array completes, we have 1000 individual CSVs and 1000 JSONs
        This funciton glues them back together into the final lon-format files used for anaylsis
    Returns:
        final_coef_df = combined coefficients
        final_metrics_df = combined metrics
    """

    # -----------------------------------------------------
    # Step 1: combine all iterations
    # -----------------------------------------------------

    temp_dir = os.path.join(save_path, "array_results_temp")
    all_coef_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.startswith("coef_")]
    all_metric_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.startswith("metrics_")]
    
    coef_list = [pd.read_csv(f) for f in all_coef_files]
    final_coef_df = pd.concat(coef_list, ignore_index=True)
    
    metrics_list = []
    for f in all_metric_files:
        with open(f, "r") as j:
            metrics_list.append(json.load(j))
    final_metrics_df = pd.DataFrame(metrics_list)

    # -----------------------------------------------------
    # Step 2: save full coefficient and metrics files 
    # -----------------------------------------------------
    final_coef_df.to_csv(
        os.path.join(save_path, "gridmet_lasso_1000_iterations_coefficients_v1.csv"),
        index=False
    )

    final_metrics_df.to_csv(
        os.path.join(save_path, "gridmet_lasso_1000_iterations_metrics_v1.csv"),
        index=False
    )

    return final_coef_df, final_metrics_df


# =========================================================
# Main workflow
# =========================================================

if __name__ == "__main__":
    #get the iteration ID form the SLURM env variable
    iteration_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))

    # -----------------------------------------------------
    # Step 1: load raw historical data
    # -----------------------------------------------------
    df = load_raw_data()

    # -----------------------------------------------------
    # Step 2: build explicit design matrix
    # Here we:
    #   - normalize only climate variables
    #   - create squared climate terms
    #   - create all county fixed effects
    #   - create county-specific time trends
    # -----------------------------------------------------
    (
        df_model,
        X_df,
        Y,
        feature_cols_final,
        climate_cols,
        climate_norm_means,
        climate_norm_stds,
        base_year
    ) = build_design_matrix(df)

    # -----------------------------------------------------
    # Step 3: remove influential outliers using Cook's distance
    # -----------------------------------------------------
    df_filtered, X_filtered, Y_filtered, cooks_threshold = remove_outliers_with_cooks_distance(
        df_model,
        X_df,
        Y
    )

    # -----------------------------------------------------
    # Step 4: run repeated county-wise 70/30 validation
    # -----------------------------------------------------
    run_single_iteration(
        X = X_filtered, 
        Y = Y_filtered,
        df_filtered = df_filtered,
        feature_cols_final = feature_cols_final,
        save_path=save_path,
        iteration_id=iteration_id
    )
    # -----------------------------------------------------
    # Step 5: Print commands
    # -----------------------------------------------------
    print(f"Iteration {iteration_id} completed.")
