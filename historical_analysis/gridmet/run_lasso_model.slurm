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
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4

# setting the max time
#SBATCH -t 24:00:00

srun python -u 9.lasso_model.py 
