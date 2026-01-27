## Calculating the county level mean temperatures (min, max, and mean) 
    #netcdf files are daily records for each pixel,
    #we need to calculate mean temps(min, max, mean) by averging over all pixels in each county
    #we would need this mean value later to calculate gdd, and then based on that we define the growing stages
    #stress indices however is calculated pixel-wise first and then only aggregated over the county

import xarray as xr
import rioxarray
import geopandas as gpd
import numpy as np
import pandas as pd
import os

#define file paths

CLIMATE_DIR = os.environ.get("CLIMATE_DIR") # base directory defined in .bashrc

shape_file = os.path.join(CLIMATE_DR, "run_project/input_data/shape_files")
loca_file = os.path.join(CLIMATE_DIR, "run_project/input_data/loca2")
output_dir = os.path.join(CLIMATE_DIR, "run_project/intermediate_data/LOCA_csv")

#load shape file
counties = gpd.read_file(os.path.join(shape_file, "CA_9counties_shapefile.shp"))
#print(counties.columns)
#print(counties.head())

#process each netcdf file

all_files = [f for f in os.listdir(loca_file) if f.endswith(".nc")]
    #this creates a list of all netcdf files in the directory 

for file in all_files:
    print(f"Processing: {file}")
    #load loca file
    ds = xr.open_dataset(os.path.join(loca_file, file))
    ds = ds.rio.write_crs("EPSG:4326")
    
    combined_list = []
    #lopp through 9 counties
    for idx, row in counties.iterrows():
        county_name = row["NAME"]
        #spatially clip dataset using the county polygon
        masked = ds.rio.clip([row.geometry], counties.crs)
        #remove empty results 
        if masked.tasmin.count().item() == 0:
            print(f"No data for {county_name} in this file.")
            continue
        #mean over space - take the mean over spatial dimensions(lat and lon) for each day 
        df = (
            masked[['tasmin', 'tasmax', 'tmean']]
            .mean(dim=['lat', 'lon'])
            .to_dataframe()
        )

        #convert K to C
        df["tasmin"] = df["tasmin"] - 273.15
        df["tasmax"] = df["tasmax"] - 273.15
        df["tmean"] = df["tmean"] - 273.15
        #add county, year, month and day column
        df["county"] = county_name
        df["year"] = df.index.year
        df["month"] = df.index.month
        df["day"] = df.index.day
        #rename index to date
        df = df.rename_axis("date").reset_index()
        df["date"] = df["date"].dt.date
        #rename columns
        df = df.rename(columns ={
            "tasmin": "tmmn",
            "tasmax": "tmmx"})
        combined_list.append(df)
                            

    #combine all counties 
    final_df = pd.concat(combined_list)
    final_df.sort_index(inplace = True)

    #save one csv per file
    output_file = os.path.join(
        output_dir, 
        f"{file.replace('.nc', '')}_county_daily_means.csv"
    )
    final_df.to_csv(output_file)

    print(f"Saved: {output_file}")
    
        
        