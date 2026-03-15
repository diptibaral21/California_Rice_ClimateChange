#!/bin/bash

# submit LOCA future yield projection jobs
# one job per (loca model * ssp)

models=(
ACCESS-CM2
CNRM-ESM2-1
EC-Earth3
EC-Earth3-Veg
GFDL-ESM4
INM-CM5-0
MPI-ESM1-2-HR
MRI-ESM2-0
FGOALS-g3
HadGEM3-GC31-LL
IPSL-CM6A-LR
KACE-1-0-G
MIROC6
)

ssps=(
ssp245
ssp585
)

# -----------------------------
# Loop over all climate models
# "${models[@]}" expands the array and iterates through each model
# -----------------------------
for model in "${models[@]}"; do

    # Print message to terminal showing which model is being processed
    echo "Submitting jobs for model: ${model}"

    # -----------------------------
    # Loop over SSP scenarios for the current model
    # -----------------------------
    for ssp in "${ssps[@]}"; do

        # Print which SSP scenario is being submitted
        echo " -- scenario: ${ssp}"

        # -----------------------------
        # Submit a SLURM job
        # -----------------------------
         # --export passes environment variables to the SLURM job
        # ALL means export all existing environment variables
        # We also pass two custom variables:
        #   LOCA_MODEL = current climate model
        #   LOCA_SSP   = current SSP scenario

        sbatch --export=ALL,LOCA_MODEL="${model}",LOCA_SSP="${ssp}" \
            run_loca_future_projection.slurm

    done  # end SSP loop
done  # end model loop