#!/bin/bash -l

#setting account name 

#SBATCH -A moniergrp

# setting name of job
#SBATCH -J 'gridmet_simulation'

# setting home directory
#SBATCH -D /group/moniergrp/dbaral/California_Rice_ClimateChange/historical_analysis/gridmet_simulation

# setting priority
#SBATCH -p high

# setting standard error output
#SBATCH -e /group/moniergrp/dbaral/run_project/slurm_log/slurm_error/gridmet_simulation_error_%j.txt

# setting standard output
#SBATCH -o /group/moniergrp/dbaral/run_project/slurm_log/slurm_output/gridmet_simulation_output_%j.txt

# setting memory and cpus
#SBATCH --mem=210G
#SBATCH --cpus-per-task=10

# setting the max time
#SBATCH -t 12:00:00

srun python -u gridmet_hist_simulation.py 
