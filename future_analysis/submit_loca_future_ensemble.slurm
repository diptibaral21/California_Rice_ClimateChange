#!/bin/bash -l

#SBATCH -A moniergrp
#SBATCH -J loca_future_ensemble
#SBATCH -D /group/moniergrp/dbaral/California_Rice_ClimateChange
#SBATCH -p high
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH -t 24:00:00
#SBATCH -o /group/moniergrp/dbaral/run_project/slurm_log/slurm_output/loca_future_ensemble_%j.out
#SBATCH -e /group/moniergrp/dbaral/run_project/slurm_log/slurm_error/loca_future_ensemble_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email_here

echo "Job started on $(date)"

# activate environment
source ~/.bashrc
conda activate myenv

srun python -u /group/moniergrp/dbaral/California_Rice_ClimateChange/future_analysis/loca_future_ensemble.py \

echo "Job finished on $(date)"