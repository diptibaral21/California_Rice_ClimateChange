#!/bin/bash -l

#setting account name 

#SBATCH -A moniergrp

# setting name of job
#SBATCH -J 'merge_array'

# setting home directory
#SBATCH -D /group/moniergrp/dbaral/California_Rice_ClimateChange/historical_analysis/gridmet

# setting priority
#SBATCH -p high

# setting standard error output
#SBATCH -e /group/moniergrp/dbaral/run_project/slurm_log/slurm_error/sterror_%j.txt

# setting standard output
#SBATCH -o /group/moniergrp/dbaral/run_project/slurm_log/slurm_output/stoutput_%j.txt

# setting memory and cpus
#SBATCH --mem=20G
#SBATCH --cpus-per-task=10

# setting the max time
#SBATCH -t 12:00:00

srun python -u 10.merge_array_results_lasso_model.py 
