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
from tqdm import tqdm


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

def build_design_matrix(df):
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
        climate_norm_means = means used to normalize climate variables
        climate_norm_stds  = stds used for normalization
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

    X_df.to_csv(os.path.join(save_path, "gridmet_tis.csv"), index=False)

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

def remove_outliers_with_cooks_distance(df_model, X_df, Y):
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

    X_filtered.to_csv(os.path.join(save_path, "gridmet_tis_filtered.csv"))

    return df_filtered, X_filtered, Y_filtered, threshold


# =========================================================
# Helper function 4
# Fit one final no-intercept Lasso model
# =========================================================

def fit_and_save_full_sample_lasso_no_intercept(
    X,
    Y,
    feature_cols_final,
    climate_cols,
    climate_norm_means,
    climate_norm_stds,
    base_year,
    save_path
):
    """
    Fit one final Lasso model with no separate intercept.

    Important:
        In this specification:
            - county dummy coefficients = county fixed effects
            - trend coefficients = county-specific time trends
            - no separate intercept is estimated

    Saves:
        1. fitted pipeline
        2. coefficient csv
        3. metadata json

    Returns:
        pipe      = fitted pipeline
        coef_df   = coefficient dataframe
        metadata  = model metadata
    """

    # -----------------------------------------------------
    # Step 1: define alpha grid
    # -----------------------------------------------------
    alphas = [5, 4, 3, 2, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]

    # -----------------------------------------------------
    # Step 2: build LassoCV pipeline with no intercept
    # No StandardScaler here because:
    #   - we already normalized climate variables manually
    #   - we do NOT want to scale county dummies
    #   - we do NOT want to scale county trends
    # -----------------------------------------------------
    pipe = Pipeline([
        ("lasso", LassoCV(
            alphas=alphas,
            cv=5,
            max_iter=int(1e7),
            random_state=45,
            fit_intercept=False
        ))
    ])

    # -----------------------------------------------------
    # Step 3: fit on full filtered dataset
    # -----------------------------------------------------
    pipe.fit(X, Y)

    # -----------------------------------------------------
    # Step 4: extract fitted Lasso model
    # -----------------------------------------------------
    lasso = pipe.named_steps["lasso"]

    # -----------------------------------------------------
    # Step 5: save coefficients
    # County coefficients are directly county fixed effects
    # -----------------------------------------------------
    coef_df = pd.DataFrame({
        "feature": feature_cols_final,
        "coefficient": lasso.coef_
    })

    coef_df.to_csv(
        os.path.join(save_path, "gridmet_hist_coefficients_normalized.csv"),
        index=False
    )

    # # -----------------------------------------------------
    # # Step 6: save fitted pipeline
    # # -----------------------------------------------------
    # joblib.dump(
    #     pipe,
    #     os.path.join(save_path, "gridmet_hist_lasso_pipeline_normalized.joblib")
    # )

    # -----------------------------------------------------
    # Step 7: save metadata
    # also save normalizing means, stds, and base year because
    # future prediction must use the same transformation
    # -----------------------------------------------------
    metadata = {
        "alpha_selected": float(lasso.alpha_),
        "r2_full_sample": float(pipe.score(X, Y)),
        "fit_intercept": False,
        "n_samples": int(len(Y)),
        "n_features": int(len(feature_cols_final)),
        "base_year_for_trend": int(base_year),
        "climate_columns": climate_cols,
        "feature_columns_final": feature_cols_final,
        "climate_norm_means": climate_norm_means,
        "climate_norm_stds": climate_norm_stds
    }

    with open(os.path.join(save_path, "gridmet_hist_model_metadata_normalized.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return pipe, coef_df, metadata

# =========================================================
# Helper function 5
# Run repeated county-wise 70/30 validation
# =========================================================

def run_1000_validation_no_intercept(
    X,
    Y,
    df_filtered,
    feature_cols_final,
    save_path,
    n_iterations=1000
):
    """
    Repeat county-wise 70/30 validation many times.

    Why:
        Instead of trusting one single split, we repeat the model fitting
        to assess:
            - coefficient stability
            - test R2 stability
            - train/test RMSE stability
            - selected alpha stability
            - use for future projection 

    County-wise split:
        We split within county so that every county is represented
        in both training and testing.

    Saves:
        1. one long coefficient file across all iterations
        2. one metrics file across all iterations

    Returns:
        final_coef_df = coefficient results across all iterations
        metrics_df    = validation metrics across all iterations
    """

    # -----------------------------------------------------
    # Step 1: define alpha grid
    # -----------------------------------------------------
    alphas = [5, 4, 3, 2, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]

    # -----------------------------------------------------
    # Step 2: initialize storage containers
    # -----------------------------------------------------
    all_coef_list = []
    metrics_list = []

    # -----------------------------------------------------
    # Step 3: get county names
    # -----------------------------------------------------
    unique_counties = sorted(df_filtered["county"].unique())

    # -----------------------------------------------------
    # Step 4: start repeated validation loop
    # -----------------------------------------------------
    for i in tqdm(range(1, n_iterations + 1), desc="Running 1000 Lasso validations"):

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
                random_state=i
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

        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

        r2_train = pipe.score(X_train, y_train)
        r2_test = pipe.score(X_test, y_test)

        # -------------------------------------------------
        # Step 4e: save coefficients in long format
        # -------------------------------------------------
        coef_df_iteration = pd.DataFrame({
            "iteration": i,
            "feature": feature_cols_final,
            "coefficient": lasso.coef_,
            "alpha_selected": lasso.alpha_
        })

        all_coef_list.append(coef_df_iteration)

        # -------------------------------------------------
        # Step 4f: save metrics for this iteration
        # -------------------------------------------------
        metrics_list.append({
            "iteration": i,
            "alpha_selected": lasso.alpha_,
            "R2_train": r2_train,
            "R2_test": r2_test,
            "RMSE_train": rmse_train,
            "RMSE_test": rmse_test
        })

    # -----------------------------------------------------
    # Step 5: combine all iterations
    # -----------------------------------------------------
    final_coef_df = pd.concat(all_coef_list, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_list)

    # -----------------------------------------------------
    # Step 6: save full coefficient stability file
    # -----------------------------------------------------
    final_coef_df.to_csv(
        os.path.join(save_path, "gridmet_lasso_1000_iterations_coefficients_no_intercept.csv"),
        index=False
    )

    # -----------------------------------------------------
    # Step 7: save full metrics stability file
    # -----------------------------------------------------
    metrics_df.to_csv(
        os.path.join(save_path, "gridmet_lasso_1000_iterations_stability_metrics_no_intercept.csv"),
        index=False
    )

    return final_coef_df, metrics_df


# =========================================================
# Main workflow
# =========================================================

if __name__ == "__main__":

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
    # Step 4: fit one final full-sample Lasso model
    # In this setup:
    #   county coefficients = county fixed effects
    # -----------------------------------------------------
    pipe, coef_df, metadata = fit_and_save_full_sample_lasso_no_intercept(
        X=X_filtered,
        Y=Y_filtered,
        feature_cols_final=feature_cols_final,
        climate_cols=climate_cols,
        climate_norm_means=climate_norm_means,
        climate_norm_stds=climate_norm_stds,
        base_year=base_year,
        save_path=save_path
    )

    # -----------------------------------------------------
    # Step 5: run repeated county-wise 70/30 validation
    # -----------------------------------------------------
    coef_1000_df, metrics_1000_df = run_1000_validation_no_intercept(
        X=X_filtered,
        Y=Y_filtered,
        df_filtered=df_filtered,
        feature_cols_final=feature_cols_final,
        save_path=save_path,
        n_iterations=1000
    )

    # -----------------------------------------------------
    # Step 6: final print statements
    # -----------------------------------------------------
    print("Finished no-intercept Lasso fit and 1000-iteration stability analysis.")
    print(f"Cook's distance threshold used: {cooks_threshold:.6f}")
    print(f"Final selected alpha: {metadata['alpha_selected']:.6f}")
    print(f"Full-sample R2: {metadata['r2_full_sample']:.4f}")