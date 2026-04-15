#!/bin/bash -l

#setting account name 

#SBATCH -A moniergrp

# setting name of job
#SBATCH -J 'loca_county_daily_means'

# setting home directory
#SBATCH -D /group/moniergrp/dbaral/California_Rice_ClimateChange/future_analysis

# setting high priority
#SBATCH -p high

# setting standard error output
#SBATCH -e /group/moniergrp/dbaral/run_project/slurm_log/slurm_error/loca_daily_means_%j.txt

# setting standard output
#SBATCH -o /group/moniergrp/dbaral/run_project/slurm_log/slurm_output/loca_daily_means_%j.txt

#setting memory
#SBATCH --mem=50G

#setting cpus
#SBATCH --cpus-per-task=2

# setting the max time
#SBATCH -t 12:00:00

# mail alerts at beginning and end of job
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

# send mail here
#SBATCH --mail-user=

srun python -u 2.loca_county_daily_means.py