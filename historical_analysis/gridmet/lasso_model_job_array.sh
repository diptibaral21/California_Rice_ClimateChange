# # Get job_id number from HPC job array
# #job_id is the ID of this job in HPC job array
# # for ex. if we submit 1000 jobs in SLURM, job_id = 1 for the first job and so on
# #each job_id represents - one full modeling on the same dataset but with a different random seed for reproducibility 
# #inside each job_id, we will still run 1000 diff 70/30 splits

# #right now we are using job_id = 1 becuase we are just using one dataset
# #In the future , we will have multiple job_id for 
#     #13 diff climate models * 2 scenarios = 26 trials
#     #each job_id will give a separate pipeline and validation results

#this goes in bash script
# #define datasets 
#     #list all gcms and scenarios and create all model-scenario combinations

# gcms = ['ACCESS-CM2', 'CNRM-ESM2-1', 'EC-Earth_3', 'EC-Earth3-Veg', 'GFDL-ESM4', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0','FGOALS-g3', 'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'KACE-1-0-G',  'MIROC6']

# scenarios = ['historical', 'ssp245', 'ssp585']

# datasets = [(gcsm, sc) for gcm in gcms for sc in scenarios]

# #get slurm task id
# try:
#     job_id = int(sys.argv[1]) - 1 #zero index
# except IndexError:
#     raise ValueError("Pass SLURM_ARRAY_TASK_ID")

# #select model + scenario for this job
# gcm, scenario = datasets[job_id]
# model_name =f"{gcm}_{scenario}"
# print("Running:", gcm, scenario)