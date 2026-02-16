#!/bin/bash -l

#setting account name 

#SBATCH -A moniergrp

# setting name of job
#SBATCH -J 'lasso_model_job_array'

# setting home directory
#SBATCH -D /group/moniergrp/dbaral/run_project/historical_analysis

# setting medium priority
#SBATCH -p high

# setting standard error output
#SBATCH -e /group/moniergrp/dbaral/run_project/slurm_log/slurm_error/sterror_%j.txt

# setting standard output
#SBATCH -o /group/moniergrp/dbaral/run_project/slurm_log/slurm_output/stoutput_%j.txt

# setting memory and cpus
#SBATCH --mem=80G
#SBATCH --cpus-per-task=6

#SBATCH --array=1

# setting the max time
#SBATCH -t 12:00:00

srun python -u lasso_model_job_array.py $SLURM_ARRAY_TASK_ID
