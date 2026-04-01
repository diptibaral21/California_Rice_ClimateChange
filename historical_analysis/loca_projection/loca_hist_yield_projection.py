import os
import argparse
import numpy as np
import pandas as pd
DEBUG = True

# =========================================================
# Paths
# =========================================================
CLIMATE_DIR = "/group/moniergrp/dbaral"
PROJECT_DIR = os.path.join(CLIMATE_DIR, "run_project")

# LOCA input directory
loca_input_dir = os.path.join(PROJECT_DIR, "input_data", "loca_hist_model_input")

# output directory
output_dir = os.path.join(PROJECT_DIR, "output_data", "projection", "loca_hist")
os.makedirs(output_dir, exist_ok=True)

# statistical model coefficient file from gridmet training
coef_file = os.path.join(
    PROJECT_DIR,
    "output_data",
    "historical_model",
    "gridmet_lasso_1000_iterations_coefficients_no_intercept.csv"
)

# static county rice area
area_file = os.path.join(
    PROJECT_DIR,
    "input_data",
    "rice_area",
    "county_rice_area_static.csv"
)

# training means file -- recommended for consistency
center_means_file = os.path.join(
    PROJECT_DIR,
    "output_data",
    "historical_model",
    "gridmet_hist_climate_center_means.csv"
)

# =========================================================
# Argument parser
# =========================================================
parser = argparse.ArgumentParser(description="LOCA yield projection using 1000 statistical models")
parser.add_argument("--loca_model", type=str, required=True, help="LOCA climate model name")
parser.add_argument("--ssp", type=str, required=True, help="Scenario name, e.g. ssp245 or ssp585")
args = parser.parse_args()

loca_model = args.loca_model
ssp = args.ssp

print(f"Running projection for LOCA model = {loca_model}, SSP = {ssp}")

# =========================================================
# Helper 1
# Load LOCA data
# =========================================================
def load_loca_data(loca_model, ssp):
    """
    Expected file pattern:
        {LOCA_MODEL}_{SSP}_r1i1p1f1_1979_2023_Lasso_Model_Input_1979_2023.csv

    Example:
        ACCESS-CM2_ssp245_r1i1p1f1_1979_2023_Lasso_Model_Input_1979_2023.csv
    """
    filename = f"{loca_model}_{ssp}_r1i1p1f1_1979_2023_Lasso_Model_Input_1979_2023.csv"
    fp = os.path.join(loca_input_dir, filename)

    if not os.path.exists(fp):
        raise FileNotFoundError(f"LOCA input file not found: {fp}")

    df = pd.read_csv(fp)

    required_cols = {

        "county", 
        "year",     
        "cdstress_bo",
        "cdstress_fl",
        "htstress_fl",
        "tmmn_bo",
        "tmmx_bo",
        "tmean_bo",
        "tmmn_fl",
        "tmmx_fl",
        "tmean_fl",
        "tmmn_gf",
        "tmmx_gf",
        "tmean_gf",
        "tmmn_gs",
        "tmmx_gs",
        "tmean_gs"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"LOCA file missing required columns: {missing}")

    return df, fp


# =========================================================
# Helper 2
# Load static rice area
# =========================================================
def load_area_data():
    area_df = pd.read_csv(area_file)

    required_cols = {"county", "rice_area_ha"}
    missing = required_cols - set(area_df.columns)
    if missing:
        raise ValueError(f"Area file is missing required columns: {missing}")

    return area_df


# =========================================================
# Helper 3
# Load coefficient matrix
# =========================================================
def load_coefficient_matrix(coef_file):
    coef_long = pd.read_csv(coef_file)

    required_cols = {"iteration", "feature", "coefficient", "alpha_selected"}
    missing = required_cols - set(coef_long.columns)
    if missing:
        raise ValueError(f"Coefficient file missing required columns: {missing}")

    coef_wide = coef_long.pivot(index="iteration", columns="feature", values="coefficient")
    coef_wide = coef_wide.sort_index(axis=0).sort_index(axis=1)

    return coef_long, coef_wide


# =========================================================
# Helper 4
# Load training means
# =========================================================
def load_training_means():
    """
    Recommended: save climate centering means from training and use them here.
    Expected columns:
        feature
        mean
    """
    if not os.path.exists(center_means_file):
        raise FileNotFoundError(
            f"Training centering means file not found: {center_means_file}"
        )

    means_df = pd.read_csv(center_means_file)

    required_cols = {"feature", "mean"}
    missing = required_cols - set(means_df.columns)
    if missing:
        raise ValueError(f"Training means file missing required columns: {missing}")

    return dict(zip(means_df["feature"], means_df["mean"]))


# =========================================================
# Helper 5
# Build projection design matrix exactly like training
# =========================================================
def build_projection_design_matrix(df, coef_wide, training_means, base_year=1979):
    """
    Rebuild X exactly like training:
      - center climate vars using TRAINING means
      - create squared terms
      - create all county dummies
      - create county-specific time trends
      - reindex to coefficient feature columns
    """
    df_model = df.copy()

    # identify expected final features from coefficient file
    feature_cols_final = list(coef_wide.columns)

    # infer county dummy and trend feature names
    county_dummy_cols = [c for c in feature_cols_final if c.startswith("county_")]
    trend_cols = [c for c in feature_cols_final if c.startswith("trend_")]

    # climate + squared features are everything else
    non_climate = set(county_dummy_cols + trend_cols)
    climate_and_sq_cols = [c for c in feature_cols_final if c not in non_climate]

    # base climate columns = those without _sq
    climate_cols = [c for c in climate_and_sq_cols if not c.endswith("_sq")]

    # check LOCA input has all required climate columns
    missing_climate = [c for c in climate_cols if c not in df_model.columns]
    if missing_climate:
        raise ValueError(f"LOCA input is missing required climate predictors: {missing_climate}")

    # center using training means
    for col in climate_cols:
        if col not in training_means:
            raise ValueError(f"Training mean not found for feature: {col}")
        df_model[col] = df_model[col] - training_means[col]
        if DEBUG:
            print("\n=== DEBUG 3: CENTERING CHECK ===")

            for col in climate_cols[:5]:
                raw_mean = df[col].mean()
                centered_mean = df_model[col].mean()
                training_mean = training_means[col]

                print(f"\nFeature: {col}")
                print(f"  Raw LOCA mean: {raw_mean:.3f}")
                print(f"  Training mean: {training_mean:.3f}")
                print(f"  Centered mean: {centered_mean:.3f}")

            print("\n Centered mean should be ~0")

    # squared terms
    for col in climate_cols:
        df_model[f"{col}_sq"] = df_model[col] ** 2

    # county dummies
    county_dummies = pd.get_dummies(
        df_model["county"],
        prefix="county",
        drop_first=False
    ).astype(int)

    # ensure all expected county dummy columns exist
    county_dummies = county_dummies.reindex(columns=county_dummy_cols, fill_value=0)

    # county-specific time trends
    time_trend = df_model["year"] - base_year
    county_trends = county_dummies.multiply(time_trend, axis=0)
    county_trends.columns = [c.replace("county_", "trend_") for c in county_trends.columns]

    county_trends = county_trends.reindex(columns=trend_cols, fill_value=0)

    # combine
    X_df = pd.concat(
        [
            df_model[climate_cols + [f"{c}_sq" for c in climate_cols]],
            county_dummies,
            county_trends
        ],
        axis=1
    )

    # reorder exactly like coefficients
    X_df = X_df.reindex(columns=feature_cols_final)
    if DEBUG:
        print("\n=== DEBUG 4: DESIGN MATRIX ===")
        print("Shape:", X_df.shape)
        print("Any NaNs:", X_df.isna().any().any())

        print("\nSample stats:")
        print(X_df.iloc[:, :5].describe())


    if X_df.isna().any().any():
        missing_cols = X_df.columns[X_df.isna().any()].tolist()
        raise ValueError(f"Design matrix contains NaNs in columns: {missing_cols}")

    return df_model, X_df


# =========================================================
# Helper 6
# Predict all 1000 statistical models
# =========================================================
def predict_all_models(X_df, coef_wide):
    X_mat = X_df.values                         # shape: n_obs x n_features
    B_mat = coef_wide.values                   # shape: n_models x n_features
    pred_matrix = B_mat @ X_mat.T              # shape: n_models x n_obs
    return pred_matrix

#@ means matrix multiplication in python
#X_mat.T means transpose of X_mat
#if B_mat is (1000*p)
#X_mat.T is (p*n)
#then y hat = B_mat*X_mat.T -- result is (1000*n)
#so we ge tone row per statitical model
#one column per county-year observation

# =========================================================
# Helper 7
# Build county-year prediction output
# =========================================================
def build_prediction_output(df_model, pred_matrix, coef_wide):
    out_df = df_model[["county", "year"]].copy().reset_index(drop=True)
    #these percentiles give uncertainty intervals
    out_df["pred_median"] = np.median(pred_matrix, axis=0)
    out_df["pred_mean"] = np.mean(pred_matrix, axis=0)
    out_df["pred_p2_5"] = np.percentile(pred_matrix, 2.5, axis=0)
    out_df["pred_p16_5"] = np.percentile(pred_matrix, 16.5, axis=0)
    out_df["pred_p83_5"] = np.percentile(pred_matrix, 83.5, axis=0)
    out_df["pred_p97_5"] = np.percentile(pred_matrix, 97.5, axis=0)

    #turn the raw prediction matrix to dataframe
    #pred_matrix.T changes shape from models * obs to obs * models
    #ecah column becomes one model's prediction 

    pred_iter_df = pd.DataFrame(
        pred_matrix.T,
        columns=[f"pred_iter_{i}" for i in coef_wide.index]
    )

    out_all_df = pd.concat([out_df, pred_iter_df], axis=1)

    return out_df, out_all_df


# =========================================================
# Helper 8
# California area-weighted yield
# =========================================================

#this function converts county-level prediction into statewide yield

def calculate_area_weighted_california_yield(pred_all_df, area_df):
    merged_df = pred_all_df.merge(area_df, on="county", how="left")

    if merged_df["rice_area_ha"].isna().any():
        missing_counties = merged_df.loc[
            merged_df["rice_area_ha"].isna(), "county"
        ].drop_duplicates().tolist()
        raise ValueError(f"Missing rice_area_ha for counties: {missing_counties}")

    pred_cols = [c for c in merged_df.columns if c.startswith("pred_iter_")]
    years = sorted(merged_df["year"].unique())

    records = []

    for year in years:
        #for each year subset to that year only and calculate total rice area
        year_df = merged_df[merged_df["year"] == year].copy()
        total_area = year_df["rice_area_ha"].sum()

        rec = {"year": year}

        #for each iteration: state yield = summation(county yield * county area) / rice area
        #so larger county with larger area contribute more
        
        for pred_col in pred_cols:
            rec[pred_col] = (year_df[pred_col] * year_df["rice_area_ha"]).sum() / total_area

        records.append(rec)

    statewide_all_iter_df = pd.DataFrame(records)

    statewide_summary_df = pd.DataFrame({
        "year": statewide_all_iter_df["year"],
        "pred_median": statewide_all_iter_df[pred_cols].median(axis=1),
        "pred_mean": statewide_all_iter_df[pred_cols].mean(axis=1),
        "pred_p2_5": statewide_all_iter_df[pred_cols].quantile(0.025, axis=1),
        "pred_p16_5": statewide_all_iter_df[pred_cols].quantile(0.165, axis=1),
        "pred_p83_5": statewide_all_iter_df[pred_cols].quantile(0.835, axis=1),
        "pred_p97_5": statewide_all_iter_df[pred_cols].quantile(0.975, axis=1),
    })

    return statewide_summary_df, statewide_all_iter_df


# =========================================================
# Main
# =========================================================
df_loca, loca_fp = load_loca_data(loca_model, ssp)
if DEBUG:
    print("\n=== DEBUG 1: LOCA RAW DATA ===")
    print("File:", loca_fp)
    print("Shape:", df_loca.shape)

    check_cols = ["tmmn_bo", "tmmx_bo", "tmean_bo"]

    print("\nSummary stats:")
    print(df_loca[check_cols].describe())

    print("\nMeans:")
    for col in check_cols:
        print(col, df_loca[col].mean())

    print("\n If values ~280–310 → Kelvin ")
    print(" If values ~5–30 → Celsius ")

area_df = load_area_data()
coef_long, coef_wide = load_coefficient_matrix(coef_file)
training_means = load_training_means()

if DEBUG:
    print("\n=== DEBUG 2: TRAINING MEANS ===")
    print("Total features:", len(training_means))

    sample_keys = list(training_means.keys())[:10]
    print("Sample features:", sample_keys)

    for k in sample_keys:
        print(f"{k}: {training_means[k]}")

print(f"Loaded LOCA file: {loca_fp}")
print(f"LOCA rows: {len(df_loca)}")
print(f"Loaded {coef_wide.shape[0]} statistical models")
print(f"Loaded {coef_wide.shape[1]} final features")

df_model, X_df = build_projection_design_matrix(
    df=df_loca,
    coef_wide=coef_wide,
    training_means=training_means,
    base_year=1979
)

print("Projection design matrix shape:", X_df.shape)

pred_matrix = predict_all_models(X_df, coef_wide)
print("Prediction matrix shape:", pred_matrix.shape)
if DEBUG:
    print("\n=== DEBUG 5: PREDICTIONS ===")
    print("Shape:", pred_matrix.shape)
    print("Min:", np.min(pred_matrix))
    print("Max:", np.max(pred_matrix))
    print("Mean:", np.mean(pred_matrix))

    print("\n Expected yield ~7000–11000")

county_year_summary_df, county_year_all_df = build_prediction_output(
    df_model=df_model,
    pred_matrix=pred_matrix,
    coef_wide=coef_wide
)

statewide_summary_df, statewide_all_iter_df = calculate_area_weighted_california_yield(
    pred_all_df=county_year_all_df,
    area_df=area_df
)

if DEBUG:
    print("\n=== DEBUG 6: FINAL STATEWIDE ===")
    print(statewide_summary_df.head())
    print("\nSummary stats:")
    print(statewide_summary_df.describe())
    
# output naming
tag = f"{loca_model}_{ssp}"

county_year_summary_fp = os.path.join(output_dir, f"{tag}_county_year_yield_summary.csv")
county_year_all_fp = os.path.join(output_dir, f"{tag}_county_year_yield_all_1000_models.csv")
statewide_summary_fp = os.path.join(output_dir, f"{tag}_california_area_weighted_yield_summary.csv")
statewide_all_fp = os.path.join(output_dir, f"{tag}_california_area_weighted_yield_all_1000_models.csv")

county_year_summary_df.to_csv(county_year_summary_fp, index=False)
county_year_all_df.to_csv(county_year_all_fp, index=False)
statewide_summary_df.to_csv(statewide_summary_fp, index=False)
statewide_all_iter_df.to_csv(statewide_all_fp, index=False)

print("Saved files:")
print(county_year_summary_fp)
print(county_year_all_fp)
print(statewide_summary_fp)
print(statewide_all_fp)
print("Done.")