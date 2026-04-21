#!/bin/bash -l

#setting account name 

#SBATCH -A moniergrp

# setting name of job
#SBATCH -J 'lasso_model'

# setting home directory
#SBATCH -D /group/moniergrp/dbaral/California_Rice_ClimateChange/historical_analysis/gridmet

# setting priority
#SBATCH -p high

# setting standard error output
#SBATCH -e /group/moniergrp/dbaral/run_project/slurm_log/slurm_error/sterror_%j.txt

# setting standard output
#SBATCH -o /group/moniergrp/dbaral/run_project/slurm_log/slurm_output/stoutput_%j.txt

# setting memory and cpus
#SBATCH --array=1-1000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# setting the max time
#SBATCH -t 1:00:00

# SLURM_ARRAY_TASK_ID will be passed to the Python script automatically
python 9.lasso_model.py