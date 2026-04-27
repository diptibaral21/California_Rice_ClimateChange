"""
=========================================================
LOCA FUTURE YIELD PLOTTING SCRIPT (FINAL FIGURE)
=========================================================

This script:
1. Loads ensemble summary outputs
2. Loads historical data
3. Aligns datasets
4. Plots publication-style figure

OUTPUT:
    - 2-panel figure (SSP245, SSP585)
    - Mean + 67% CI + 95% CI

=========================================================
"""

# =========================================================
# IMPORT LIBRARIES
# =========================================================
import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PATHS
# =========================================================
CLIMATE_DIR = "/group/moniergrp/dbaral"
PROJECT_DIR = os.path.join(CLIMATE_DIR, "run_project")

# Ensemble outputs
ensemble_dir = os.path.join(
    PROJECT_DIR,
    "output_data",
    "projection",
    "loca_future_ensemble"
)

# Historical simulation (LOCA historical)
hist_path = os.path.join(
    PROJECT_DIR,
    "output_data",
    "projection",
    "loca_hist_ensemble",
    "loca_13model_historical_ensemble_summary.csv"
)

area_path = os.path.join(PROJECT_DIR, "input_data", "rice_area", "county_rice_area_static.csv")

yield_path = os.path.join(PROJECT_DIR, "input_data", "yield", "rice_yield_1979_2023.csv")


output_dir = os.path.join(PROJECT_DIR, "output_data/projection/loca_future_ensemble")
os.makedirs(output_dir, exist_ok=True)
# =========================================================
# LOAD DATA
# =========================================================

print("Loading datasets...")

# SSP245
df_245_sustained = pd.read_csv(
    os.path.join(ensemble_dir,
    "loca_13model_ssp245_sustained_ensemble_summary.csv")
)

df_245_fixed = pd.read_csv(
    os.path.join(ensemble_dir,
    "loca_13model_ssp245_fixed_ensemble_summary.csv")
)

# SSP585
df_585_sustained = pd.read_csv(
    os.path.join(ensemble_dir,
    "loca_13model_ssp585_sustained_ensemble_summary.csv")
)

df_585_fixed = pd.read_csv(
    os.path.join(ensemble_dir,
    "loca_13model_ssp585_fixed_ensemble_summary.csv")
)

# Historical
hist_df = pd.read_csv(hist_path)

print("Datasets loaded successfully.")


#Historical observed yield 
area_df = pd.read_csv(area_path)

obs_yield = pd.read_csv(yield_path)
# merge
obs_df = obs_yield.merge(area_df, on="county")

# weighted average per year
obs_state = (
    obs_df
    .groupby("year")
    .apply(lambda x: (x["yield_kg_ha"] * x["rice_area_ha"]).sum() / x["rice_area_ha"].sum())
    .reset_index(name="yield_kg_ha")
)
 
# =========================================================
# CLEAN + ALIGN YEARS
# =========================================================

# Keep historical up to transition year
hist_df = hist_df[hist_df["year"] <= 2023]
obs_state = obs_state[obs_state["year"] <= 2023]

# Keep future from transition onward
df_245_sustained = df_245_sustained[df_245_sustained["year"] >= 2023]
df_245_fixed     = df_245_fixed[df_245_fixed["year"] > 2023]

df_585_sustained = df_585_sustained[df_585_sustained["year"] >= 2023]
df_585_fixed     = df_585_fixed[df_585_fixed["year"] >= 2023]


# =========================================================
# SANITY CHECK
# =========================================================
print("\nChecking data...")
print(hist_df.head())
print(df_245_sustained.head())

assert "mean" in hist_df.columns, "Historical file missing 'mean'"
assert "p16_5" in df_245_sustained.columns, "Future file missing CI columns"


# =========================================================
# PLOTTING FUNCTION
# =========================================================

def plot_panel(ax, hist_df, sustained_df, fixed_df, title):

    # ---------------------------
    # HISTORICAL (BLACK)
    # ---------------------------
    ax.plot(hist_df["year"], hist_df["mean"],
            color="black", linewidth=2, label="Historical Simulated")

    ax.fill_between(
        hist_df["year"],
        hist_df["p16_5"],
        hist_df["p83_5"],
        color="gray",
        alpha=0.3
    )
    # ---------------------------
    # OBSERVED (GREEN)
    # ---------------------------
    ax.plot(obs_state["year"], obs_state["yield_kg_ha"],
            color="green", linewidth=2, label="Observed Yield")

    # ---------------------------
    # SUSTAINED (BLUE)
    # ---------------------------
    ax.plot(sustained_df["year"], sustained_df["mean"],
            color="blue", linewidth=2, label="Sustained")

    ax.fill_between(
        sustained_df["year"],
        sustained_df["p2_5"],
        sustained_df["p97_5"],
        color="blue",
        alpha=0.15
    )

    ax.fill_between(
        sustained_df["year"],
        sustained_df["p16_5"],
        sustained_df["p83_5"],
        color="blue",
        alpha=0.35
    )

    # ---------------------------
    # FIXED (RED)
    # ---------------------------
    ax.plot(fixed_df["year"], fixed_df["mean"],
            color="red", linewidth=2, label="Fixed")

    ax.fill_between(
        fixed_df["year"],
        fixed_df["p2_5"],
        fixed_df["p97_5"],
        color="red",
        alpha=0.15
    )

    ax.fill_between(
        fixed_df["year"],
        fixed_df["p16_5"],
        fixed_df["p83_5"],
        color="red",
        alpha=0.35
    )

    # ---------------------------
    # FORMATTING
    # ---------------------------
    ax.axvline(x=2023, linestyle="--", color="gray", alpha=0.7)

    ax.set_title(title)
    ax.set_ylabel("Yield (kg/ha)")
    ax.grid(True)


# =========================================================
# CREATE FIGURE
# =========================================================

print("Creating figure...")

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# SSP245
plot_panel(
    axes[0],
    hist_df,
    df_245_sustained,
    df_245_fixed,
    "SSP245"
)

# SSP585
plot_panel(
    axes[1],
    hist_df,
    df_585_sustained,
    df_585_fixed,
    "SSP585"
)

axes[1].set_xlabel("Year")

# Legend only once
axes[0].legend()

plt.tight_layout()


# =========================================================
# SAVE FIGURE
# =========================================================

save_file = os.path.join(output_dir, "future_yield_projection.png")

plt.savefig(save_file, dpi=300)
print(f"Figure saved to: {save_file}")
plt.show()


# =========================================================
# DONE
# =========================================================
print("Done.")