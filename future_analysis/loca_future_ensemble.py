import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# Paths
# =========================================================
CLIMATE_DIR = "/group/moniergrp/dbaral"
PROJECT_DIR = os.path.join(CLIMATE_DIR, "run_project")

# input directory: LOCA future projection outputs
loca_output_dir = os.path.join(
    PROJECT_DIR,
    "output_data",
    "projection",
    "loca_future"
)

# output directory for ensemble products and plot
ensemble_output_dir = os.path.join(
    PROJECT_DIR,
    "output_data",
    "projection",
    "loca_future_ensemble"
)
os.makedirs(ensemble_output_dir, exist_ok=True)

# =========================================================
# Settings
# =========================================================
ssp_list = ["ssp245", "ssp585"]

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
# Load and merge all LOCA future model all-iteration files
# =========================================================
def build_loca_future_ensemble(model_list, ssp):
    """
    Reads each file:
        {model}_{ssp}_california_area_weighted_yield_all_1000_models.csv

    Returns
    -------
    ensemble_all_df : DataFrame
        year + all model-specific prediction columns
    ensemble_summary_df : DataFrame
        year + ensemble summary statistics
    """
    merged_dfs = []

    for model in model_list:
        fp = os.path.join(
            loca_output_dir,
            f"{model}_{ssp}_california_area_weighted_yield_all_1000_models.csv"
        )

        if not os.path.exists(fp):
            print(f"WARNING: file not found, skipping -> {fp}")
            continue

        df = pd.read_csv(fp)

        pred_cols = [c for c in df.columns if c.startswith("pred_iter_")]
        required_cols = {"year"}
        missing = required_cols - set(df.columns)

        if missing:
            raise ValueError(f"File missing required columns {missing}: {fp}")

        if len(pred_cols) == 0:
            raise ValueError(f"No pred_iter_* columns found in: {fp}")

        # rename prediction columns to keep names unique after merging
        rename_dict = {col: f"{model}_{col}" for col in pred_cols}
        df = df.rename(columns=rename_dict)

        keep_cols = ["year"] + list(rename_dict.values())
        df = df[keep_cols].copy()

        merged_dfs.append(df)

        print(f"Loaded: {fp}")
        print(f"  Years: {df['year'].min()}-{df['year'].max()}")
        print(f"  Prediction columns: {len(rename_dict)}")

    if len(merged_dfs) == 0:
        raise ValueError(f"No LOCA files found for {ssp} in {loca_output_dir}")

    # merge all model dataframes by year
    ensemble_all_df = merged_dfs[0]
    for df_next in merged_dfs[1:]:
        ensemble_all_df = ensemble_all_df.merge(df_next, on="year", how="outer")

    ensemble_all_df = ensemble_all_df.sort_values("year").reset_index(drop=True)

    # all ensemble prediction columns
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
# Helper 2
# Save ensemble outputs
# =========================================================
def save_ensemble_outputs(ensemble_all_df, ensemble_summary_df, ssp):
    all_fp = os.path.join(
        ensemble_output_dir,
        f"loca_13model_{ssp}_all_iterations.csv"
    )

    summary_fp = os.path.join(
        ensemble_output_dir,
        f"loca_13model_{ssp}_ensemble_summary.csv"
    )

    ensemble_all_df.to_csv(all_fp, index=False)
    ensemble_summary_df.to_csv(summary_fp, index=False)

    print("\nSaved ensemble files:")
    print(all_fp)
    print(summary_fp)

    return all_fp, summary_fp


# =========================================================
# Helper 3
# Plot future LOCA ensemble
# =========================================================
def plot_loca_future_ensemble(ensemble_summary_df, ssp):
    """
    Plot future LOCA ensemble only:
    - black median line
    - dark gray 67% CI
    - light gray 95% CI
    """
    df_plot = ensemble_summary_df.sort_values("year").copy()

    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    # 95% CI
    ax.fill_between(
        df_plot["year"],
        df_plot["pred_p2_5"],
        df_plot["pred_p97_5"],
        color="gray",
        alpha=0.25,
        label="95% CI"
    )

    # 67% CI
    ax.fill_between(
        df_plot["year"],
        df_plot["pred_p16_5"],
        df_plot["pred_p83_5"],
        color="gray",
        alpha=0.50,
        label="67% CI"
    )

    # ensemble median
    ax.plot(
        df_plot["year"],
        df_plot["pred_median"],
        color="black",
        linewidth=2,
        label=ssp.upper()
    )

    ax.set_ylabel("Yield kg/ha")
    ax.set_xlabel("")
    ax.set_xlim(df_plot["year"].min(), df_plot["year"].max())
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
    print("Building LOCA future ensembles...")
    print(f"Input directory: {loca_output_dir}")

    for ssp in ssp_list:
        print("\n" + "=" * 60)
        print(f"Processing {ssp}")

        ensemble_all_df, ensemble_summary_df = build_loca_future_ensemble(
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

        plot_loca_future_ensemble(
            ensemble_summary_df=ensemble_summary_df,
            ssp=ssp
        )

    print("\nDone.")


if __name__ == "__main__":
    main()