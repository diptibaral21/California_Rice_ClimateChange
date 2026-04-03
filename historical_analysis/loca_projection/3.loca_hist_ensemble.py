
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# Paths
# =========================================================
CLIMATE_DIR = "/group/moniergrp/dbaral"
PROJECT_DIR = os.path.join(CLIMATE_DIR, "run_project")

# input directory: LOCA historical projection outputs
loca_output_dir = os.path.join(
    PROJECT_DIR,
    "output_data",
    "projection",
    "loca_hist"
)

# output directory for ensemble products and plot
ensemble_output_dir = os.path.join(
    PROJECT_DIR,
    "output_data",
    "projection",
    "loca_hist_ensemble"
)
os.makedirs(ensemble_output_dir, exist_ok=True)

# observed California yield file
# 
observed_file = os.path.join(
    PROJECT_DIR,
    "input_data",
    "yield",
    "rice_yield_1979_2023.csv"
)

# =========================================================
# Settings
# =========================================================

ssp = "historical"

# LOCA model names
model_list = [
    "ACCESS-CM2",
    "CNRM-ESM2-1",
    "EC-Earth3",
    "EC-Earth3-Veg",
    "FGOALS-g3",
    "GFDL-ESM4",
    "HadGEM3-GC31-LL",
    "INM-CM5-0",
    "IPSL-CM6A-LR",
    "KACE-1-0-G",
    "MIROC6",
    "MPI-ESM1-2-HR",
    "MRI-ESM2-0"
]

# =========================================================
# Helper 1
# Load observed yield
# =========================================================
def load_observed_yield():
    """
    Expected observed file:
        must contain columns:
            county
            year
            yield_kg_ha

    """
    if not os.path.exists(observed_file):
        raise FileNotFoundError(f"Observed yield file not found: {observed_file}")

    obs_df = pd.read_csv(observed_file)

    required_cols = {"year", "county", "yield_kg_ha"}
    missing = required_cols - set(obs_df.columns)
    if missing:
        raise ValueError(f"Observed yield file missing required columns: {missing}")

    return obs_df[["year", "county", "yield_kg_ha"]].copy()


# =========================================================
# Helper 2
# Load and merge all LOCA model all-iteration files
# =========================================================
def build_loca_historical_ensemble(model_list, ssp):
    """
    Reads each file:
        {model}_{ssp}_california_area_weighted_yield_all_1000_models.csv

    Each file is expected to contain:
        year
        pred_iter_1
        pred_iter_2
        ...
        pred_iter_1000

    The script renames each prediction column so that all columns remain unique
    after merging across models.

    Returns
    -------
    ensemble_all_df : DataFrame
        year + all model-specific statistical prediction columns
    ensemble_summary_df : DataFrame
        year + median/mean/67%/95% ensemble summary
    """
    merged_dfs = []

    for model in model_list:
        fp = os.path.join(
            loca_output_dir,
            f"{model}_{ssp}_california_area_weighted_yield_all_1000_models.csv"
        )

        if not os.path.exists(fp):
            print(f"WARNING: file not found, skipping - {fp}")
            continue

        df = pd.read_csv(fp)

        pred_cols = [c for c in df.columns if c.startswith("pred_iter_")]
        required_cols = {"year"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"File missing required columns {missing}: {fp}")

        if len(pred_cols) == 0:
            raise ValueError(f"No pred_iter_* columns found in: {fp}")

        # rename prediction columns to keep them unique after merge
        rename_dict = {col: f"{model}_{col}" for col in pred_cols}
        df = df.rename(columns=rename_dict)

        keep_cols = ["year"] + list(rename_dict.values())
        df = df[keep_cols].copy()

        merged_dfs.append(df)

        print(f"Loaded: {fp}")
        print(f"  Years: {df['year'].min()}-{df['year'].max()}")
        print(f"  Prediction columns: {len(rename_dict)}")

    if len(merged_dfs) == 0:
        raise ValueError("No LOCA model files were found. Check model names, ssp, and directory.")

    # merge all model dataframes by year
    ensemble_all_df = merged_dfs[0]
    for df_next in merged_dfs[1:]:
        ensemble_all_df = ensemble_all_df.merge(df_next, on="year", how="outer")

    ensemble_all_df = ensemble_all_df.sort_values("year").reset_index(drop=True)

    # all ensemble prediction columns across all models
    ensemble_pred_cols = [c for c in ensemble_all_df.columns if c != "year"]

    # calculate ensemble summary
    ensemble_summary_df = pd.DataFrame({
        "year": ensemble_all_df["year"],
        "pred_median": ensemble_all_df[ensemble_pred_cols].median(axis=1),
        "pred_mean": ensemble_all_df[ensemble_pred_cols].mean(axis=1),
        "pred_p2_5": ensemble_all_df[ensemble_pred_cols].quantile(0.025, axis=1),
        "pred_p16_5": ensemble_all_df[ensemble_pred_cols].quantile(0.165, axis=1),
        "pred_p83_5": ensemble_all_df[ensemble_pred_cols].quantile(0.835, axis=1),
        "pred_p97_5": ensemble_all_df[ensemble_pred_cols].quantile(0.975, axis=1),
    })

    return ensemble_all_df, ensemble_summary_df


# =========================================================
# Helper 3
# Save ensemble outputs
# =========================================================
def save_ensemble_outputs(ensemble_all_df, ensemble_summary_df, ssp):
    all_fp = os.path.join(
        ensemble_output_dir,
        f"loca_13model_{}_all_iterations.csv"
    )

    summary_fp = os.path.join(
        ensemble_output_dir,
        f"loca_13model_{}_ensemble_summary.csv"
    )

    ensemble_all_df.to_csv(all_fp, index=False)
    ensemble_summary_df.to_csv(summary_fp, index=False)

    print("\nSaved ensemble files:")
    print(all_fp)
    print(summary_fp)

    return all_fp, summary_fp


# =========================================================
# Helper 4
# Plot LOCA historical ensemble vs observed
# =========================================================
def plot_loca_historical_ensemble(ensemble_summary_df, obs_df, ssp):
    """
    Makes a historical LOCA ensemble plot like the paper:
    - black median line
    - dark gray 67% CI
    - light gray 95% CI
    - green observed line
    """
    df_plot = ensemble_summary_df.merge(obs_df, on="year", how="left").sort_values("year").copy()

    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    # 95% CI
    ax.fill_between(
        df_plot["year"],
        df_plot["pred_p2_5"],
        df_plot["pred_p97_5"],
        color="gray",
        alpha=0.25
    )

    # 67% CI
    ax.fill_between(
        df_plot["year"],
        df_plot["pred_p16_5"],
        df_plot["pred_p83_5"],
        color="gray",
        alpha=0.50
    )

    # LOCA ensemble median
    ax.plot(
        df_plot["year"],
        df_plot["pred_median"],
        color="black",
        linewidth=2,
        label="LOCA2-Hybrid"
    )

    # observed
    ax.plot(
        df_plot["year"],
        df_plot["yield_kg_ha"],
        color="limegreen",
        linewidth=1.8,
        label="Observed"
    )

    ax.set_ylabel("Yield ton/acre")
    ax.set_xlabel("")
    ax.set_xlim(df_plot["year"].min(), df_plot["year"].max())

    # match paper-style angled years
    ax.tick_params(axis="x", rotation=60)

    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()

    plot_fp = os.path.join(
        ensemble_output_dir,
        f"loca_13model_{ssp}_ensemble_plot.png"
    )

    plt.savefig(plot_fp, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nSaved plot:")
    print(plot_fp)

    return plot_fp


# =========================================================
# Main
# =========================================================
def main():
    print("Building LOCA historical ensemble...")
    print(f"Input directory: {loca_output_dir}")
    print(f"SSP tag used in filenames: {ssp}")

    obs_df = load_observed_yield()

    ensemble_all_df, ensemble_summary_df = build_loca_historical_ensemble(
        model_list=model_list,
        ssp=ssp
    )

    print("\nEnsemble summary:")
    print(f"  Years: {ensemble_summary_df['year'].min()}-{ensemble_summary_df['year'].max()}")
    print(f"  Number of years: {len(ensemble_summary_df)}")

    ensemble_pred_cols = [c for c in ensemble_all_df.columns if c != "year"]
    print(f"  Total ensemble prediction columns: {len(ensemble_pred_cols)}")

    save_ensemble_outputs(
        ensemble_all_df=ensemble_all_df,
        ensemble_summary_df=ensemble_summary_df,
        ssp=ssp
    )

    plot_loca_historical_ensemble(
        ensemble_summary_df=ensemble_summary_df,
        obs_df=obs_df,
        ssp=ssp
    )

    print("\nDone.")


if __name__ == "__main__":
    main()