#!/bin/bash -l

#setting account name 

#SBATCH -A moniergrp

# setting name of job
#SBATCH -J 'download LOCA models'

# setting home directory
#SBATCH -D /group/moniergrp/LOCA2_CA

# setting medium priority
#SBATCH -p bmh

# setting standard error output
#SBATCH -e /group/moniergrp/dbaral/run_project/slurm_log/sterror_%j.txt

# setting standard output
#SBATCH -o /group/moniergrp/dbaral/run_project/slurm_log/sterror_%j.txt

#setting memory
#SBATCH --mem=50G

#setting cpus
#SBATCH --cpus-per-task=6

# setting the max time
#SBATCH -t 12:00:00

# mail alerts at beginning and end of job
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

# send mail here
#SBATCH --mail-user=

srun python -u download_1model.py