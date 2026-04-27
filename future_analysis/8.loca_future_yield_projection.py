"""
=========================================================
LOCA FUTURE YIELD PROJECTION (2020–2100)
=========================================================

This script projects rice yield using:
    - LOCA climate projections
    - Previously trained Lasso model (1000 iterations)

=========================================================
KEY CONSISTENCY FEATURES
=========================================================

1. SAME cleaned coefficient set (filtered bad models)
2. SAME feature alignment as training
3. SAME normalization approach as LOCA historical

IMPORTANT:
------------
Future LOCA climate variables are normalized using
LOCA historical (1979–2023) statistics:

    X_norm = (X_future - mean_hist) / std_hist

This ensures consistency between historical and future
projections.

=========================================================
TREND SCENARIOS
=========================================================

Two options:

1. sustained
    → technological trend continues into future

2. fixed
    → trend capped at 2023 (climate-only signal)

=========================================================
NEGATIVE YIELD HANDLING
=========================================================

All predicted yields < 0 are set to 0 to ensure
physical realism.

=========================================================
OUTPUTS
=========================================================

For each LOCA model and SSP:

- County-level summary + ensemble
- Statewide area-weighted summary + ensemble

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

loca_input_dir = os.path.join(PROJECT_DIR, "input_data", "loca_future_model_input")
output_dir = os.path.join(PROJECT_DIR, "output_data", "projection", "loca_future")
os.makedirs(output_dir, exist_ok=True)

coef_file = os.path.join(
    PROJECT_DIR,
    "output_data",
    "historical_model",
    "final_cleaned_coefficients.csv"
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
parser.add_argument("--trend_mode", type=str, required=True,
                    choices=["sustained", "fixed"])
args = parser.parse_args()

loca_model = args.loca_model
ssp = args.ssp
trend_mode = args.trend_mode

# =========================================================
# LOAD DATA
# =========================================================
def load_loca_data():
    fp = os.path.join(
        loca_input_dir,
        f"{loca_model}_{ssp}_r1i1p1f1_Lasso_Model_Input_2020_2100.csv"
    )
    return pd.read_csv(fp)


def load_area():
    return pd.read_csv(area_file)


def load_coefficients():
    df = pd.read_csv(coef_file)
    return df.pivot(index="iteration", columns="feature", values="coefficient")


def load_hist_stats():
    fp = os.path.join(
        PROJECT_DIR,
        "output_data",
        "projection",
        "loca_hist",
        f"{loca_model}_historical_standardization_stats.csv"
    )
    df = pd.read_csv(fp)

    means = dict(zip(df["feature"], df["mean"]))
    stds = dict(zip(df["feature"], df["std"]))

    return means, stds


# =========================================================
# BUILD DESIGN MATRIX
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

    #create squared from RAW
    for col in climate_cols:
        df[f"{col}_sq"] = df[col] ** 2

    # Normalize
    means, stds = {}, {}
    for col in climate_cols + [f"{c}_sq" for c in climate_cols]:
        means[col] = float(df[col].mean())
        stds[col] = float(df[col].std())
        df[col] = (df[col] - means[col]) / stds[col]

    # county FE
    dummies = pd.get_dummies(df["county"], prefix="county").astype(float)
    dummies = dummies.reindex(columns=county_cols, fill_value=0)

    # trend
    raw_trend = df["year"] - base_year
    cutoff = 2023 - base_year

    if trend_mode == "sustained":
        trend = raw_trend
    else:
        trend = raw_trend.clip(upper=cutoff)

    trends = dummies.multiply(trend, axis=0)
    trends.columns = [c.replace("county_", "trend_") for c in trends.columns]
    trends = trends.reindex(columns=trend_cols, fill_value=0)

    # combine
    X = pd.concat(
        [df[climate_cols + [f"{c}_sq" for c in climate_cols]], dummies, trends],
        axis=1
    )

    X = X.reindex(columns=feature_cols)
    return df, X.astype(float)


# =========================================================
# PREDICT
# =========================================================
def predict_all(X, coef_wide):
    coef_wide = coef_wide[X.columns]
    pred = coef_wide.values @ X.values.T

    # critical fix
    pred = np.maximum(pred, 0)

    return pred


# =========================================================
# COUNTY OUTPUT
# =========================================================
def county_output(df_model, pred, coef_wide):

    df_out = df_model[["county", "year"]].copy()

    df_out["mean"] = pred.mean(axis=0)
    df_out["p67_low"] = np.percentile(pred, 16.5, axis=0)
    df_out["p67_high"] = np.percentile(pred, 83.5, axis=0)
    df_out["p95_low"] = np.percentile(pred, 2.5, axis=0)
    df_out["p95_high"] = np.percentile(pred, 97.5, axis=0)

    df_all = pd.DataFrame(pred.T,
                          columns=[f"pred_iter_{i}" for i in coef_wide.index])

    df_all = pd.concat([df_out[["county", "year"]], df_all], axis=1)

    return df_out, df_all


# =========================================================
# STATEWIDE
# =========================================================
def statewide(df_model, pred, area_df):
    """
    Returns BOTH:
        1. statewide_summary (mean + CI)
        2. statewide_all (iteration-level)
    """

    df_all = pd.concat(
        [df_model[["county", "year"]].reset_index(drop=True),
         pd.DataFrame(pred.T, columns=[f"pred_iter_{i}" for i in range(pred.shape[0])])],
        axis=1
    )

    df_all = df_all.merge(area_df, on="county")

    pred_cols = [c for c in df_all.columns if c.startswith("pred_iter_")]

    records = []
    statewide_all_list = []

    for yr in sorted(df_all["year"].unique()):
        temp = df_all[df_all["year"] == yr]
        area = temp["rice_area_ha"].values

        # weighted iterations
        weighted = np.sum(temp[pred_cols].values * area[:, None], axis=0) / np.sum(area)

        # save iteration-level
        row = {"year": yr}
        for i, val in enumerate(weighted):
            row[f"pred_iter_{i}"] = val
        statewide_all_list.append(row)

        # summary
        records.append({
            "year": yr,
            "mean": weighted.mean(),
            "p67_low": np.percentile(weighted, 16.5),
            "p67_high": np.percentile(weighted, 83.5),
            "p95_low": np.percentile(weighted, 2.5),
            "p95_high": np.percentile(weighted, 97.5),
        })

    statewide_all = pd.DataFrame(statewide_all_list)
    statewide_summary = pd.DataFrame(records)

    return statewide_summary, statewide_all

# =========================================================
# MAIN
# =========================================================
df = load_loca_data()
area_df = load_area()
coef_wide = load_coefficients()
means, stds = load_hist_stats()

df_model, X = build_X(df, coef_wide, means, stds)

pred = predict_all(X, coef_wide)

county_summary, county_all = county_output(df_model, pred, coef_wide)
statewide_summary, statewide_all = statewide(df_model, pred, area_df)


# =========================================================
# SAVE
# =========================================================
tag = f"{loca_model}_{ssp}_{trend_mode}"

county_summary.to_csv(os.path.join(output_dir, f"{tag}_county_summary.csv"), index=False)
county_all.to_csv(os.path.join(output_dir, f"{tag}_county_all_1000.csv"), index=False)
statewide_summary.to_csv(os.path.join(output_dir, f"{tag}_statewide_summary.csv"), index=False)
statewide_all.to_csv(os.path.join(output_dir, f"{tag}_statewide_all_1000.csv"), index=False)

print("Saved outputs:", tag)