#!/bin/bash -l

#setting account name 

#SBATCH -A moniergrp

# setting name of job
#SBATCH -J 'plot_loca_hist_ensemble'

# setting home directory
#SBATCH -D /group/moniergrp/dbaral/California_Rice_ClimateChange/historical_analysis/loca_projection

# setting high priority
#SBATCH -p high

# setting standard error output
#SBATCH -e /group/moniergrp/dbaral/run_project/slurm_log/slurm_error/plot_%j.txt

# setting standard output
#SBATCH -o /group/moniergrp/dbaral/run_project/slurm_log/slurm_output/plot_%j.txt

#setting memory
#SBATCH --mem=2G

#setting cpus
#SBATCH --cpus-per-task=1

# setting the max time
#SBATCH -t 12:00:00

# mail alerts at beginning and end of job
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

# send mail here
#SBATCH --mail-user=

srun python -u loca_hist_plot.py