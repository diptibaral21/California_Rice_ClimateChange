import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# PATHS
# =========================================================
CLIMATE_DIR = "/group/moniergrp/dbaral"
PROJECT_DIR = os.path.join(CLIMATE_DIR, "run_project")

# Input directory (LOCA future outputs)
loca_output_dir = os.path.join(
    PROJECT_DIR,
    "output_data",
    "projection",
    "loca_future"
)

# Output directory (ensemble + plots)
ensemble_output_dir = os.path.join(
    PROJECT_DIR,
    "output_data",
    "projection",
    "loca_future_ensemble"
)
os.makedirs(ensemble_output_dir, exist_ok=True)


# =========================================================
# SETTINGS
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
# FUNCTION 1: BUILD ENSEMBLE
# =========================================================
def build_loca_future_ensemble(model_list, ssp, trend_mode):
    """
    Build ensemble for one SSP:
    - Merge all models
    - Compute median + uncertainty bands
    """

    merged_dfs = []

    for model in model_list:
        if trend_mode == "sustained":
            filename = f"{model}_{ssp}_sustained_california_area_weighted_yield_all_1000_models.csv"
        elif trend_mode == "stopped":
            filename = f"{model}_{ssp}_stopped_california_area_weighted_yield_all_1000_models.csv"
        else:
            raise ValueError("Invalid trend_mode")

        fp = os.path.join(loca_output_dir, filename)

        if not os.path.exists(fp):
            print(f"WARNING: missing -> {fp}")
            continue

        df = pd.read_csv(fp)

        # find prediction columns
        pred_cols = [c for c in df.columns if c.startswith("pred_iter_")]

        if len(pred_cols) == 0:
            raise ValueError(f"No pred_iter_* columns in {fp}")

        # rename to avoid collision across models
        rename_dict = {col: f"{model}_{col}" for col in pred_cols}
        df = df.rename(columns=rename_dict)

        df = df[["year"] + list(rename_dict.values())]

        merged_dfs.append(df)

        print(f"Loaded {model} ({ssp})")

    if len(merged_dfs) == 0:
        raise ValueError(f"No files found for {ssp}")

    # merge all models
    ensemble_all_df = merged_dfs[0]
    for df_next in merged_dfs[1:]:
        ensemble_all_df = ensemble_all_df.merge(df_next, on="year", how="outer")

    ensemble_all_df = ensemble_all_df.sort_values("year").reset_index(drop=True)

    pred_cols = [c for c in ensemble_all_df.columns if c != "year"]

    # summary statistics
    ensemble_summary_df = pd.DataFrame({
        "year": ensemble_all_df["year"],
        "pred_median": ensemble_all_df[pred_cols].median(axis=1),
        "pred_mean": ensemble_all_df[pred_cols].mean(axis=1),
        "pred_p2_5": ensemble_all_df[pred_cols].quantile(0.025, axis=1),
        "pred_p16_5": ensemble_all_df[pred_cols].quantile(0.165, axis=1),
        "pred_p83_5": ensemble_all_df[pred_cols].quantile(0.835, axis=1),
        "pred_p97_5": ensemble_all_df[pred_cols].quantile(0.975, axis=1),
    })

    return ensemble_all_df, ensemble_summary_df


# =========================================================
# FUNCTION 2: SAVE OUTPUTS
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

    print(f"Saved outputs for {ssp}")


# =========================================================
# FUNCTION 3: SIDE-BY-SIDE PLOT
# =========================================================
def plot_side_by_side(all_summaries, y_min, y_max):
    """
    LEFT: SSP245
    RIGHT: SSP585

    Each panel:
        sustained (blue)
        stopped (red)
    """

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for i, ssp in enumerate(ssp_list):
        ax = axes[i]

        df_sustained = all_summaries[ssp]["sustained"].sort_values("year")
        df_stopped = all_summaries[ssp]["stopped"].sort_values("year")

        # sustained
        ax.fill_between(
            df_sustained["year"],
            df_sustained["pred_p2_5"],
            df_sustained["pred_p97_5"],
            color="blue",
            alpha=0.2
        )

        ax.plot(
            df_sustained["year"],
            df_sustained["pred_median"],
            color="blue",
            linewidth=2,
            label="Sustained"
        )

        # stopped
        ax.fill_between(
            df_stopped["year"],
            df_stopped["pred_p2_5"],
            df_stopped["pred_p97_5"],
            color="red",
            alpha=0.2
        )

        ax.plot(
            df_stopped["year"],
            df_stopped["pred_median"],
            color="red",
            linewidth=2,
            label="Stopped"
        )

        ax.set_title(ssp.upper())
        ax.set_xlim(df_sustained["year"].min(), df_sustained["year"].max())
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis="x", rotation=60)

        if i == 0:
            ax.set_ylabel("Yield (kg/ha)")

        ax.legend()

    plt.tight_layout()

    plot_fp = os.path.join(
        ensemble_output_dir,
        "loca_13model_side_by_side_trend_comparison.png"
    )

    plt.savefig(plot_fp, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved comparison plot")

# =========================================================
# MAIN WORKFLOW
# =========================================================
def main():

    print("\n=== BUILDING ENSEMBLES ===")

    all_summaries = {}

    # -----------------------------------------------------
    # STEP 1: Build each SSP
    # -----------------------------------------------------
    
    
    for ssp in ssp_list:
        print(f"\nProcessing {ssp}")

        all_summaries[ssp] = {}

        for trend_mode in ["sustained", "stopped"]:

            print(f"  Trend: {trend_mode}")

            ensemble_all_df, ensemble_summary_df = build_loca_future_ensemble(
                model_list,
                ssp,
                trend_mode
            )

            # save outputs separately
            save_ensemble_outputs(
                ensemble_all_df,
                ensemble_summary_df,
                f"{ssp}_{trend_mode}"
            )

            all_summaries[ssp][trend_mode] = ensemble_summary_df

    # -----------------------------------------------------
    # STEP 2: GLOBAL Y RANGE (CRITICAL)
    # -----------------------------------------------------
    all_dfs = []

    for ssp in ssp_list:
        for trend_mode in ["sustained", "stopped"]:
            all_dfs.append(all_summaries[ssp][trend_mode])

    combined_df = pd.concat(all_dfs, ignore_index=True)

    y_min = combined_df["pred_p2_5"].min()
    y_max = combined_df["pred_p97_5"].max()

    # small buffer → nicer plot
    buffer = 0.05 * (y_max - y_min)
    y_min -= buffer
    y_max += buffer

    print(f"\nGlobal y-range: {y_min:.2f} to {y_max:.2f}")

    # -----------------------------------------------------
    # STEP 3: SIDE-BY-SIDE PLOT
    # -----------------------------------------------------
    plot_side_by_side(all_summaries, y_min, y_max)

    print("\n=== DONE ===")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    main()