#download the LOCA models
#check the dependencies 
#this is the code to download the remaining 2 models for tasmin and tasmax 
#because these two models are missing variables before tasmin and tasmax, the variable number for tasmin and tasmax is diff
#(notice the range(2,4))

import xarray as xr
import intake
cat = intake.open_esm_datastore(
    'https://cadcat.s3.amazonaws.com/cae-collection.json'
)
#
var_list=  ['pr', 'rsds', 'tasmax', 'tasmin', 'uas', 'vas', 'wspeed']
model_list = [ 
    'IPSL-CM6A-LR', 
    'MIROC6'
]
for model in model_list:
    for i in range(2,4): # because these two model don't have first 3 variables 
        ds = xr.open_zarr(cat['LOCA2.UCSD.'+str(model)+'.ssp585.day.d03'].df.path[i],storage_options={'anon': True})
        ds.to_netcdf(str(model)+'_ssp585_r1i1p1f1_'+str(var_list[i])+'.nc')