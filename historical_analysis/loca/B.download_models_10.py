#download the LOCA models
#check the dependencies 
#this is the code to download the 10 models for tasmin and tasmax 

import xarray as xr
import intake
cat = intake.open_esm_datastore(
    'https://cadcat.s3.amazonaws.com/cae-collection.json'
)
#
var_list=  ['hursmax', 'hursmin', 'huss', 'pr', 'rsds', 'tasmax', 'tasmin', 'uas', 'vas', 'wspeed']
model_list = [   
    'KACE-1-0-G'
]
for model in model_list:
    for i in range(5,7):
        ds = xr.open_zarr(cat['LOCA2.UCSD.'+str(model)+'.historical.day.d03'].df.path[i],storage_options={'anon': True})
        ds.to_netcdf(str(model)+'_historical_r1i1p1f1_'+str(var_list[i])+'.nc')