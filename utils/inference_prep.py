import numpy as np
import xarray as xr
import os

DATA_DIR = "data"

def normalize(array):
    """Applies Min-Max normalization, scaling values between 0 and 1."""
    min_val = np.nanmin(array)
    max_val = np.nanmax(array)
    if max_val == min_val:
        return np.zeros_like(array)
    return (array - min_val) / (max_val - min_val)

def get_inference_tensor(date_str):
    """
    Loads NetCDF datasets, aligns and regrids them at the xarray level,
    and returns a single inference tensor of shape (1, 7, 1, 301, 720).
    """
    # Load individual datasets
    ds_temp = xr.open_dataset(os.path.join(DATA_DIR, "temp_sample.nc"))
    ds_sal = xr.open_dataset(os.path.join(DATA_DIR, "salinity_sample.nc"))
    ds_curr = xr.open_dataset(os.path.join(DATA_DIR, "currents_sample.nc"))
    ds_ssh = xr.open_dataset(os.path.join(DATA_DIR, "sealevel_sample.nc"))
    ds_wind = xr.open_dataset(os.path.join(DATA_DIR, "wind_sample.nc"))

    # Establish the master reference grid using temperature
    temp = ds_temp['thetao']
    
    # Align and regrid all datasets to match the temp grid structure directly
    sal = ds_sal['so'].interp_like(temp)
    curr_u = ds_curr['uo'].interp_like(temp)
    curr_v = ds_curr['vo'].interp_like(temp)
    ssh = ds_ssh['total_sea_level'].interp_like(temp)
    wind_u = ds_wind['eastward_wind'].interp_like(temp)
    wind_v = ds_wind['northward_wind'].interp_like(temp)

    # Extract the first time index cleanly (or slice by date safely)
    sub_temp = temp.isel(time=0).values
    sub_sal = sal.isel(time=0).values
    sub_curr_u = curr_u.isel(time=0).values
    sub_curr_v = curr_v.isel(time=0).values
    sub_ssh = ssh.isel(time=0).values
    sub_wind_u = wind_u.isel(time=0).values
    sub_wind_v = wind_v.isel(time=0).values

    # Helper to guarantee every array has a depth dimension: (1, Lat, Lon)
    def ensure_3d(arr):
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)
        return arr

    arrays = [
        ensure_3d(sub_temp),
        ensure_3d(sub_sal),
        ensure_3d(sub_curr_u),
        ensure_3d(sub_curr_v),
        ensure_3d(sub_ssh),
        ensure_3d(sub_wind_u),
        ensure_3d(sub_wind_v)
    ]

    # Stack channels -> (Channels, Depth, Lat, Lon)
    input_features = np.stack(arrays, axis=0)

    # Convert landmass NaNs to 0.0 and normalize
    input_features = np.nan_to_num(input_features, nan=0.0)
    input_features_norm = normalize(input_features)

    # Add batch dimension -> (1, 7, 1, 301, 720)
    inference_tensor = np.expand_dims(input_features_norm, axis=0)
    
    return inference_tensor