import xarray as xr
import numpy as np
import torch

# Geographic boundaries specified by MoES PS 26066
LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0

def load_and_preprocess_surface(nc_file_path):
    """
    Ingests daily satellite NetCDF file, crops to Indian Ocean,
    handles NaNs over land, and stacks 5 surface variables into a tensor.
    """
    ds = xr.open_dataset(nc_file_path)
    
    # Subset geography
    cropped = ds.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
    
    # Extract surface variables (SST, SSS, SLA, U-wind/curr, V-wind/curr)
    sst = np.nan_to_num(cropped['sst'].values, nan=0.0)
    sss = np.nan_to_num(cropped['sss'].values, nan=0.0)
    sla = np.nan_to_num(cropped['sla'].values, nan=0.0)
    u_curr = np.nan_to_num(cropped['u'].values, nan=0.0)
    v_curr = np.nan_to_num(cropped['v'].values, nan=0.0)
    
    # Stack into 5-channel array -> Shape: (5, H, W)
    stacked = np.stack([sst, sss, sla, u_curr, v_curr], axis=0)
    
    # Channel-wise Min-Max Normalization
    for c in range(5):
        c_min, c_max = stacked[c].min(), stacked[c].max()
        if c_max > c_min:
            stacked[c] = (stacked[c] - c_min) / (c_max - c_min)
            
    # Convert to 4D Tensor -> Shape: (1, 5, H, W)
    tensor = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)
    
    return tensor, cropped.lat.values, cropped.lon.values