#download the LOCA models
#check the dependencies 
#this is the code to download the HadGEM model for tasmin and tasmax -- using r1i1p1f3 as it dosn't have r1i1p1f1 variant

import xarray as xr
import intake
cat = intake.open_esm_datastore(
    'https://cadcat.s3.amazonaws.com/cae-collection.json'
)
#
var_list=  ['hursmax', 'hursmin', 'pr', 'rsds', 'tasmax', 'tasmin', 'uas', 'vas', 'wspeed']

for i in range(4,6): 
    ds = xr.open_zarr(cat['LOCA2.UCSD.'+ 'HadGEM3-GC31-LL'+'.ssp585.day.d03'].df.path[i],storage_options={'anon': True})
    ds.to_netcdf('HadGEM3-GC31-LL'+'_ssp585_r1i1p1f1_'+str(var_list[i])+'.nc')