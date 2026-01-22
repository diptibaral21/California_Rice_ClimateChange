#download the LOCA models
#check the dependencies 
#this is the code to download the 10 models for tasmin and tasmax for ssp245
#to download ssp585 - change ssp245.day.d03 to ssp585.day.d03
#for more details about the data, variables and download visit - https://loca.ucsd.edu/loca-version-2-for-north-america-ca-jan-2023/

import xarray as xr
import intake
cat = intake.open_esm_datastore(
    'https://cadcat.s3.amazonaws.com/cae-collection.json'
)
#
var_list=  ['hursmax', 'hursmin', 'huss', 'pr', 'rsds', 'tasmax', 'tasmin', 'uas', 'vas', 'wspeed']
model_list = [ 
    'ACCESS-CM2', 
    'CNRM-ESM2-1',
    'EC-Earth 3',
    'EC-Earth3-Veg', 
    'FGOALS-g3', 
    'GFDL-ESM4',
    'INM-CM5-0',
    'KACE-1-0-G',
    'MPI-ESM1-2-HR', 
    'MRI-ESM2-0'
]
for model in model_list:
    for i in range(5,7):
        ds = xr.open_zarr(cat['LOCA2.UCSD.'+str(model)+'.ssp245.day.d03'].df.path[i],storage_options={'anon': True})
        ds.to_netcdf(str(model)+'_ssp245_r1i1p1f1_'+str(var_list[i])+'.nc')