"""

=========================================================
OVERVIEW
=========================================================
This script aggregates outputs from SLURM array jobs.

Each SLURM task produces:
    - One coefficient file
    - One metrics file

This script:
    - Combines all coefficient files
    - Combines all metric files
    - Saves final datasets for analysis

=========================================================
WHY THIS STEP IS IMPORTANT
=========================================================
Instead of relying on a single model, we:
    - train 1000 models
    - capture variability in results

This allows us to:
    - estimate uncertainty
    - compute confidence intervals
    - assess model stability

=========================================================
OUTPUTS
=========================================================
1. final_coefficients.csv
2. final_metrics.csv
"""

import os
import json
import pandas as pd

CLIMATE_DIR = "/group/moniergrp/dbaral"
OUTPUT_PATH = os.path.join(CLIMATE_DIR, "run_project/output_data/historical_model")


def aggregate():

    temp = os.path.join(OUTPUT_PATH, "array_results_temp")

    coef_files = [f for f in os.listdir(temp) if f.startswith("coef")]
    metric_files = [f for f in os.listdir(temp) if f.startswith("metrics")]

    coef_df = pd.concat([pd.read_csv(os.path.join(temp, f)) for f in coef_files])

    metrics = []
    for f in metric_files:
        metrics.append(json.load(open(os.path.join(temp, f))))

    pd.DataFrame(coef_df).to_csv(os.path.join(OUTPUT_PATH, "final_coefficients.csv"), index=False)
    pd.DataFrame(metrics).to_csv(os.path.join(OUTPUT_PATH, "final_metrics.csv"), index=False)


if __name__ == "__main__":
    aggregate()