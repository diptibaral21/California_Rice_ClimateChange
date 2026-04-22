"""

=========================================================
OVERVIEW
=========================================================
This script simulates historical rice yield using LOCA
climate datasets and a previously trained Lasso model.

The trained statistical model (based on GridMET data) is
applied to LOCA-derived climate variables to generate
yield predictions across:

    - Multiple LOCA climate models (e.g., ACCESS-CM2, MIROC6)
    - Multiple SSP scenarios (e.g., ssp245, ssp585)
    - Historical period (1979–2023)

=========================================================
MODEL FRAMEWORK
=========================================================
The yield model is specified as:

    Yield =
        County Fixed Effects
      + County-Specific Time Trends
      + Climate Variables (standardized)
      + Squared Climate Variables

Where:
    - Climate variables are stage-specific temperature indices
    - Squared terms capture nonlinear responses
    - No intercept is used (absorbed by county fixed effects)

=========================================================
IMPORTANT MODELING CHOICE
=========================================================
Climate variables are normalized using LOCA-specific
mean and standard deviation:

    X_norm = (X - mean_LOCA) / std_LOCA

This differs from training (GridMET normalization) and is
intentional, based on advisor guidance.

Implication:
------------
- Coefficients are applied to re-scaled inputs
- Interpretation shifts slightly but prediction remains valid
- Must be clearly documented in thesis/paper

=========================================================
WORKFLOW
=========================================================

1. Load LOCA climate dataset
2. Load trained Lasso model coefficients (1000 iterations)
3. Compute LOCA-specific normalization parameters
4. Build design matrix:
    - normalize climate variables
    - generate squared terms
    - create county fixed effects
    - create county-specific time trends
5. Align design matrix with coefficient matrix (CRITICAL)
6. Predict yield for all 1000 models
7. Generate outputs:
    - county-level yield (summary + full ensemble)
    - statewide area-weighted yield (summary + ensemble)

=========================================================
OUTPUTS
=========================================================

For each LOCA model and SSP:

1. County-level outputs:
    - {model}_{ssp}_county_summary.csv
    - {model}_{ssp}_county_all_1000.csv

2. Statewide outputs:
    - {model}_{ssp}_statewide_summary.csv
    - {model}_{ssp}_statewide_all_1000.csv

=========================================================
CRITICAL IMPLEMENTATION NOTES
=========================================================

- Feature alignment:
    Coefficient matrix must match design matrix EXACTLY

- No intercept:
    County fixed effects represent baseline yield

- Area weighting:
    Required for statewide aggregation

- Numerical stability:
    All matrices must be float type

- Units:
    Yield remains in kg/ha

=========================================================
"""

import os
import argparse
import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================
CLIMATE_DIR = "/group/moniergrp/dbaral"
PROJECT_DIR = os.path.join(CLIMATE_DIR, "run_project")

loca_input_dir = os.path.join(PROJECT_DIR, "input_data", "loca_hist_model_input")
output_dir = os.path.join(PROJECT_DIR, "output_data", "projection", "loca_hist")
os.makedirs(output_dir, exist_ok=True)

coef_file = os.path.join(
    PROJECT_DIR,
    "output_data",
    "historical_model",
    "final_coefficients.csv"
)

area_file = os.path.join(
    PROJECT_DIR,
    "input_data",
    "rice_area",
    "county_rice_area_static.csv"
)


# =========================================================
# ARGUMENTS
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--loca_model", type=str, required=True)
parser.add_argument("--ssp", type=str, required=True)
args = parser.parse_args()

loca_model = args.loca_model
ssp = args.ssp


# =========================================================
# LOAD DATA
# =========================================================
def load_loca_data():
    filename = f"{loca_model}_{ssp}_r1i1p1f1_Lasso_Model_Input_1979_2023.csv"
    fp = os.path.join(loca_input_dir, filename)

    if not os.path.exists(fp):
        raise FileNotFoundError(fp)

    return pd.read_csv(fp)


def load_area():
    return pd.read_csv(area_file)


def load_coefficients():
    df = pd.read_csv(coef_file)
    return df.pivot(index="iteration", columns="feature", values="coefficient")


# =========================================================
# LOCA NORMALIZATION
# =========================================================
def compute_loca_stats(df, coef_wide):
    feature_cols = list(coef_wide.columns)

    county_cols = [c for c in feature_cols if c.startswith("county_")]
    trend_cols = [c for c in feature_cols if c.startswith("trend_")]

    climate_cols = [
        c for c in feature_cols
        if c not in county_cols + trend_cols and not c.endswith("_sq")
    ]

    means = {c: df[c].mean() for c in climate_cols}
    stds = {c: df[c].std() for c in climate_cols}

    return means, stds


# =========================================================
# DESIGN MATRIX
# =========================================================
def build_X(df, coef_wide, means, stds, base_year=1979):

    df = df.copy()
    feature_cols = list(coef_wide.columns)

    county_cols = [c for c in feature_cols if c.startswith("county_")]
    trend_cols = [c for c in feature_cols if c.startswith("trend_")]

    climate_cols = [
        c for c in feature_cols
        if c not in county_cols + trend_cols and not c.endswith("_sq")
    ]

    # normalize
    for col in climate_cols:
        df[col] = (df[col] - means[col]) / stds[col]

    # squared terms
    for col in climate_cols:
        df[f"{col}_sq"] = df[col] ** 2

    # county FE
    dummies = pd.get_dummies(df["county"], prefix="county").astype(float)
    dummies = dummies.reindex(columns=county_cols, fill_value=0)

    # trends
    trend = df["year"] - base_year
    trends = dummies.multiply(trend, axis=0)
    trends.columns = [c.replace("county_", "trend_") for c in trends.columns]
    trends = trends.reindex(columns=trend_cols, fill_value=0)

    X = pd.concat(
        [df[climate_cols + [f"{c}_sq" for c in climate_cols]], dummies, trends],
        axis=1
    )

    # 🔥 CRITICAL FIX
    X = X.reindex(columns=feature_cols)

    # dtype safety
    X = X.astype(float)

    return df, X


# =========================================================
# PREDICTION
# =========================================================
def predict_all_models(X, coef_wide):

    # 🔥 CRITICAL FIX: align features
    coef_wide = coef_wide[X.columns]

    return coef_wide.values @ X.values.T


# =========================================================
# COUNTY OUTPUTS
# =========================================================
def build_county_outputs(df_model, pred_matrix, coef_wide):

    df_out = df_model[["county", "year"]].copy().reset_index(drop=True)

    df_out["pred_mean"] = pred_matrix.mean(axis=0)
    df_out["pred_median"] = np.median(pred_matrix, axis=0)
    df_out["pred_p2_5"] = np.percentile(pred_matrix, 2.5, axis=0)
    df_out["pred_p16_5"] = np.percentile(pred_matrix, 16.5, axis=0)
    df_out["pred_p83_5"] = np.percentile(pred_matrix, 83.5, axis=0)
    df_out["pred_p97_5"] = np.percentile(pred_matrix, 97.5, axis=0)

    df_all = pd.DataFrame(
        pred_matrix.T,
        columns=[f"pred_iter_{i}" for i in coef_wide.index]
    )

    df_all = pd.concat([df_out[["county", "year"]], df_all], axis=1)

    return df_out, df_all


# =========================================================
# STATEWIDE AREA-WEIGHTED YIELD
# =========================================================
def compute_statewide(df_model, pred_matrix, area_df):

    df_all = pd.concat(
        [df_model[["county", "year"]].reset_index(drop=True),
         pd.DataFrame(pred_matrix.T)],
        axis=1
    )

    df_all = df_all.merge(area_df, on="county")

    pred_cols = df_all.columns[2:-1]

    records = []

    for yr in sorted(df_all["year"].unique()):
        temp = df_all[df_all["year"] == yr]

        area = temp["rice_area_ha"].values

        weighted = np.sum(temp[pred_cols].values * area[:, None], axis=0) / np.sum(area)

        records.append({
            "year": yr,
            "mean": weighted.mean(),
            "p67_low": np.percentile(weighted, 16.5),
            "p67_high": np.percentile(weighted, 83.5),
            "p95_low": np.percentile(weighted, 2.5),
            "p95_high": np.percentile(weighted, 97.5),
        })

    return pd.DataFrame(records)
# =========================================================
# STATEWIDE ALL ITERATIONS ( REQUIRED FOR ENSEMBLE)
# =========================================================
def compute_statewide_all_iterations(df_model, pred_matrix, area_df):
    """
    Compute statewide area-weighted yield for EACH iteration.

    This preserves all 1000 simulations per LOCA model,
    which is required for building the multi-model ensemble.

    Returns:
        DataFrame:
            year + pred_iter_0 ... pred_iter_999
    """

    df_all = pd.concat(
        [
            df_model[["county", "year"]].reset_index(drop=True),
            pd.DataFrame(pred_matrix.T)
        ],
        axis=1
    )

    df_all = df_all.merge(area_df, on="county")

    pred_cols = df_all.columns[2:-1]

    records = []

    for yr in sorted(df_all["year"].unique()):
        temp = df_all[df_all["year"] == yr]

        area = temp["rice_area_ha"].values

        # weighted yield for EACH iteration
        weighted = np.sum(
            temp[pred_cols].values * area[:, None],
            axis=0
        ) / np.sum(area)

        rec = {"year": yr}
        rec.update({f"pred_iter_{i}": val for i, val in enumerate(weighted)})

        records.append(rec)

    return pd.DataFrame(records)

# =========================================================
# MAIN
# =========================================================
df = load_loca_data()
area_df = load_area()
coef_wide = load_coefficients()

# ===== DEBUG BLOCK 1: COEFFICIENT CHECK =====
print("\n===== DEBUG: COEFFICIENT MATRIX =====")
print("Shape:", coef_wide.shape)
print("Sample columns:", list(coef_wide.columns[:10]))
print("Coefficient stats:")
print(coef_wide.describe().T[["mean", "std"]].head())
print("Any NaNs in coefficients?", coef_wide.isna().sum().sum())


means, stds = compute_loca_stats(df, coef_wide)

df_model, X = build_X(df, coef_wide, means, stds)

# ===== DEBUG BLOCK 2: DESIGN MATRIX CHECK =====
print("\n===== DEBUG: DESIGN MATRIX =====")
print("Shape:", X.shape)
print("Sample columns:", list(X.columns[:10]))

print("\nX stats (first few vars):")
print(X.describe().T[["mean", "std"]].head())

print("Any NaNs in X?", X.isna().sum().sum())

# Column alignment check
print("\n===== DEBUG: ALIGNMENT CHECK =====")
missing_in_X = set(coef_wide.columns) - set(X.columns)
extra_in_X = set(X.columns) - set(coef_wide.columns)

print("Missing in X:", missing_in_X)
print("Extra in X:", extra_in_X)

assert len(missing_in_X) == 0, "ERROR: Missing features in X"
assert len(extra_in_X) == 0, "ERROR: Extra features in X"

# Strict ordering check
assert list(X.columns) == list(coef_wide.columns), "ERROR: Column order mismatch"


# ===== DEBUG BLOCK 3: PRE-PREDICTION CHECK =====
print("\n===== DEBUG: PRE-PREDICTION =====")
print("X global min/max:", X.min().min(), X.max().max())
print("Coefficient global min/max:", coef_wide.min().min(), coef_wide.max().max())


pred_matrix = predict_all_models(X, coef_wide)

# ===== DEBUG BLOCK 4: PREDICTION CHECK =====
print("\n===== DEBUG: PREDICTIONS =====")
print("Prediction shape:", pred_matrix.shape)

print("Global min/max:", pred_matrix.min(), pred_matrix.max())

print("\nPer-iteration min/max (first 5):")
for i in range(5):
    print(f"Iter {i}: min={pred_matrix[i].min()}, max={pred_matrix[i].max()}")

# Count extreme values
extreme_mask = (pred_matrix < 0) | (pred_matrix > 20000)
print("\nExtreme values count:", extreme_mask.sum())

# Identify worst iteration
worst_iter = np.argmax(np.abs(pred_matrix).max(axis=1))
print("Worst iteration index:", worst_iter)
print("Worst iteration min/max:",
      pred_matrix[worst_iter].min(),
      pred_matrix[worst_iter].max())


# ===== DEBUG BLOCK 5: SINGLE MODEL TEST =====
test_pred = coef_wide.iloc[0].values @ X.values.T

print("\n===== DEBUG: SINGLE MODEL =====")
print("Single model min/max:", test_pred.min(), test_pred.max())


# ---------------------------------------------------------
# COUNTY OUTPUTS
# ---------------------------------------------------------
county_summary, county_all = build_county_outputs(df_model, pred_matrix, coef_wide)

# ---------------------------------------------------------
# STATEWIDE SUMMARY
# ---------------------------------------------------------
statewide_summary = compute_statewide(df_model, pred_matrix, area_df)

# ---------------------------------------------------------
# NEW: STATEWIDE ALL ITERATIONS (CRITICAL)
# ---------------------------------------------------------
statewide_all = compute_statewide_all_iterations(
    df_model, pred_matrix, area_df
)

# =========================================================
# SAVE OUTPUTS
# =========================================================
tag = f"{loca_model}_{ssp}"

county_summary.to_csv(os.path.join(output_dir, f"{tag}_county_summary.csv"), index=False)
county_all.to_csv(os.path.join(output_dir, f"{tag}_county_all_1000.csv"), index=False)

statewide_summary.to_csv(os.path.join(output_dir, f"{tag}_statewide_summary.csv"), index=False)

statewide_all.to_csv(os.path.join(output_dir, f"{tag}_statewide_all_1000.csv"), index=False)

print("Saved all outputs.")