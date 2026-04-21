# =========================================================
# import libraries and functions
# =========================================================
import os
import pandas as pd
from lasso_model import aggregate_array_results

# =========================================================
# File paths
# =========================================================

CLIMATE_DIR = "/group/moniergrp/dbaral"
save_path = os.path.join(CLIMATE_DIR, "run_project/output_data/historical_model")

# =========================================================
# Run function 
# =========================================================

aggregate_array_results(save_path)

print("All 1000 results merged into final CSVs")