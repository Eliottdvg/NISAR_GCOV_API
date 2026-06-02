import earthaccess
import xarray as xr
import rioxarray
import rasterio as rio
import numpy as np
import os
import yaml

import re
from urllib.parse import urlparse
from datetime import datetime
from pathlib import PurePosixPath

import argparse

# requires pip-system-certs
auth = earthaccess.login(persist=True) # authentication 

########## Parametres ##########
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", required=True, type=str, help="Enter path to yaml config file")
args = parser.parse_args()
config_path = args.config_path

with open(config_path) as stream:
    try:
        file = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

AOI = file['AOI']
Zones = file['Zones']
temp_extent = tuple(file['temp_extent'].split(', '))
FILTERS = file['FILTERS']
product = file['product']
group_path = file['group_path']
R_BOI = file['R_BOI']

Save_path = f'./Data/NISAR_GCOV/{AOI}/{group_path.split('/')[-1]}'
Zones = Zones[AOI].split(', ')
spatial_extent = (float(Zones[0]), float(Zones[1]), float(Zones[2]), float(Zones[3]))
bbox_crs = Zones[-1]
print(bbox_crs, spatial_extent)

proceed = input(f'Processing Area {AOI}\n '
                f'temporal extent : {temp_extent}\n '
                f'Filters: {FILTERS}\n '
                f'product: {product}, {group_path}\n '
                f'Bands : {R_BOI}\n '
                f'Save path : {Save_path}\n'
                'Proceed ? (Y/n)\n')

if proceed == 'n' :
    quit()
elif proceed == 'Y' :
    pass

######### Requests ########
results = earthaccess.search_data(
    short_name=product, # The product to access, there is many more, see doc.
    count=-1, 
    bounding_box=spatial_extent, 
    temporal=temp_extent,
    cloud_hosted=True,
)

# Filtering the URL 
mask = np.where([all(f in result["meta"]['native-id'] for f in FILTERS) for result in results])[0]
results = [results[id] for id in mask]
names = [result["meta"]['native-id'] for result in results]

print(len(results), 'files found') 
# print([r['meta']['native-id'] for r in results]) # Printing the name of the files 

# Configuration connection
fsspec_config = {
    'cache_type': 'background',
    'block_size': 16*1024*1024,  # 16 MB
}
fs = earthaccess.get_fsspec_https_session()

https_links = [result.data_links(access='external') for result in results]
files = [fs.open(https_link[0], **fsspec_config) for https_link in https_links]

# Creating a list of datatrees, takes a few minutes
datatrees = [xr.open_datatree(
                f,
                engine='h5netcdf',
                decode_timedelta=False,
                phony_dims="access", 
                chunks = 'auto',
                )[group_path]
for f in files]

############ Preprocessing ###############
NISAR_TS_RE = re.compile(r"_(\d{8}T\d{6})_")

def nisar_start_time_from_url(url: str) -> datetime:
    path = urlparse(url).path
    name = PurePosixPath(path).name
    
    m = NISAR_TS_RE.search(name)
    if not m:
        raise ValueError(f"No NISAR timestamp found in: {url}")
    
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")

dts = [nisar_start_time_from_url(url[0]) for url in https_links]

datasets = [
    tree.ds.assign_coords(time=dt).expand_dims(time=1)
    for dt, tree in zip(dts, datatrees)
]

########### Projection #########
dataarrays = [
    ds[R_BOI].to_dataarray(dim="band").assign_coords(projection = ds.projection)
    for ds in datasets
]

dataarrays_proj =  [da.rio.write_crs(f"EPSG:{da.isel(time=0).projection.item()}") for da in dataarrays]

bbox4326 = dict(minx=spatial_extent[0], miny=spatial_extent[1], maxx=spatial_extent[2], maxy=spatial_extent[3], crs=bbox_crs)
da_subset = [da.rio.clip_box(**bbox4326) for da in dataarrays_proj]
ts: xr.DataArray = xr.concat(da_subset, dim="time")

####### Saving ##########
print('Saving file...')
os.makedirs(Save_path, exist_ok=True)
ts.to_netcdf(os.path.join(Save_path, '_'.join(FILTERS) + '_' + temp_extent[0] + '.cdf'))