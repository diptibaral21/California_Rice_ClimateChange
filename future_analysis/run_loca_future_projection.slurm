#!/bin/bash -l

#SBATCH -A moniergrp
#SBATCH -J loca_future_proj
#SBATCH -D /group/moniergrp/dbaral/run_project
#SBATCH -p high
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH -t 24:00:00
#SBATCH -o /group/moniergrp/dbaral/run_project/slurm_log/slurm_output/loca_future_proj_%j.out
#SBATCH -e /group/moniergrp/dbaral/run_project/slurm_log/slurm_error/loca_future_proj_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email_here

echo "Job started on $(date)"

echo "LOCA_MODEL = ${LOCA_MODEL}"
echo "LOCA_SSP   = ${LOCA_SSP}"

# activate environment
source ~/.bashrc
conda activate myenv

srun python -u /group/moniergrp/dbaral/California_Rice_ClimateChange/future_analysis/loca_future_yield_projection.py \
    --loca_model "${LOCA_MODEL}" \
    --ssp "${LOCA_SSP}"

echo "Job finished on $(date)"