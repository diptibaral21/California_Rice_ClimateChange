## Clip the loca models by rice growing polygons
# Here we take the netcdf files, select years from 2000 to 2020 and clipe it to ricegrowing areas and save a new netcdf that only contains rice-area data

#this will remove unnecessary pixels and later we can calculate growing degree days, count heat stress, and run spatial analysis

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
my_dir = '/group/moniergrp/LOCA2_CA_clipped_historical'

#load rice polygons

shape_files = "/group/moniergrp/dbaral/run_project/input_data/shape_files"
rice = gpd.read_file(shape_files + "/Rice_Growing_Areas_30m.shp")


#loop through files in shared dir

for f in os.listdir(shared_dir):
    #only process if files that meet all below critera
    if (f.endswith(".nc") and
        ("tasmin" in f or "tasmax" in f) and 
        ("historical" in f)):

        file_path = os.path.join(shared_dir, f)
        print(f"processing {f} ..")

        #open LOCA file
        ds = xr.open_dataset(file_path, engine = 'netcdf4', chunks={'time': 365})
        #select time slice 2000-2020
        ds = ds.sel(time=slice('2000-01-01', '2020-12-31'))
        clipped_vars={}
        #loop through all variables in the dataset
        for var in ds.data_vars:
            print(f" - Clipping variable: {var}")
            da = ds[var]
            da = da.rio.write_crs("EPSG:4326")
            clipped = da.rio.clip(rice.geometry, rice.crs)
            clipped_vars[var] = clipped
        #combine all clipped variables into one dataset
        clipped_ds = xr.Dataset(clipped_vars)
        #save output file
        out_file = os.path.join(my_dir, f.replace(".nc", "_rice.nc"))
        clipped_ds.to_netcdf(out_file)
        print(f"saved {out_file}\n")

print("Done all variables clipped and time-sliced for all files.") 