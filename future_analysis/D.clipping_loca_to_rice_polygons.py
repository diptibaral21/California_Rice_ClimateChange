#importing libraries
import os
import pandas as pd
import numpy as np
import glob
import xarray as xr
import netCDF4 as nc
import geopandas as gpd
import rioxarray 
import warnings

#file paths -- shared_dir to access data and my_dir to save data
shared_dir = '/group/moniergrp/LOCA2_CA'
my_dir = '/group/moniergrp/LOCA2_future_rice_clipped'

models = ['ACCESS-CM2', 'CNRM-ESM2-1', 'EC-Earth3', 'EC-Earth3-Veg', 'GFDL-ESM4', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0','FGOALS-g3', 'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'KACE-1-0-G',  'MIROC6']
variables = ['tasmin', 'tasmax']
scenarios = ['ssp245', 'ssp585']

output_dir = "/group/moniergrp/dbaral/run_project/input_data/loca_future"

for model in models:
    for scenario in scenarios:
        datasets = []
        for var in variables:
            file_path = os.path.join(my_dir,f"{model}_{scenario}_r1i1p1f1_{var}_rice.nc") 
            if os.path.exists(file_path):
                ds = xr.open_dataset(file_path)
                datasets.append(ds)
            else:
                print(f"File not found: {file_path}")
        if datasets:
            combined_ds = xr.merge(datasets, compat = "override")
            #compute mean temperature
            combined_ds['tmean'] = (combined_ds['tasmin']+combined_ds['tasmax'])/2
            
            #growing season mask(May to october)
            mask = (
                ((combined_ds.time.dt.month >= 5) & (combined_ds.time.dt.month <=10)) 
            )
            # subset the data for all years
            subset = combined_ds.sel(time =mask)
        #save the combine dataset
        output_file = os.path.join(output_dir, f"{model}_{scenario}_r1i1p1f1_rice_temp.nc")
        subset.to_netcdf(output_file)
        print(f"Saved combined file: {output_file}")

