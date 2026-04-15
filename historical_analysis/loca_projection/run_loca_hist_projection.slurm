#!/bin/bash -l

#SBATCH -A moniergrp
#SBATCH -J loca_yield_proj
#SBATCH -D /group/moniergrp/dbaral/run_project
#SBATCH -p high
#SBATCH --cpus-per-task=4 
#SBATCH --mem=10G
#SBATCH -t 12:00:00
#SBATCH -o /group/moniergrp/dbaral/run_project/slurm_log/slurm_output/loca_proj_%j.out
#SBATCH -e /group/moniergrp/dbaral/run_project/slurm_log/slurm_error/loca_proj_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email_here

echo "Job started on $(date)"

echo "LOCA_MODEL = ${LOCA_MODEL}"
echo "LOCA_SSP   = ${LOCA_SSP}"

# activate environment
source ~/.bashrc
conda activate myenv

srun python -u /group/moniergrp/dbaral/California_Rice_ClimateChange/historical_analysis/loca_projection/2.loca_hist_yield_projection.py \
    --loca_model "${LOCA_MODEL}" \
    --ssp "${LOCA_SSP}"

echo "Job finished on $(date)"