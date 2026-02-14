#!/bin/bash -l

#setting account name 

#SBATCH -A moniergrp

# setting name of job
#SBATCH -J 'download_models'

# setting home directory
#SBATCH -D /group/moniergrp/LOCA2_CA

# setting priority
#SBATCH -p high

# setting standard error output
#SBATCH -e /group/moniergrp/dbaral/run_project/slurm_log/slurm_error/sterror_%j.txt

# setting standard output
#SBATCH -o /group/moniergrp/dbaral/run_project/slurm_log/slurm_output/sterror_%j.txt

#setting memory
#SBATCH --mem=10G

#setting cpus
#SBATCH --cpus-per-task=6

# setting the max time
#SBATCH -t 6:00:00

# mail alerts at beginning and end of job
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

# send mail here
#SBATCH --mail-user=

srun python -u /group/moniergrp/dbaral/run_project/historical_analysis/loca/download_models_10.py 