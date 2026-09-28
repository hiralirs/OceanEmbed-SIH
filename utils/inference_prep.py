import os
import numpy as np
import xarray as xr

DATA_DIR = "data"


def normalize(array):
  """Applies Min-Max normalization, scaling values between 0 and 1."""
  min_val = np.nanmin(array)
  max_val = np.nanmax(array)
  if max_val == min_val:
    return np.zeros_like(array)
  return (array - min_val) / (max_val - min_val)


def get_inference_tensor(date_str):
  """Loads NetCDF datasets, aligns them to the 0.25 deg SIH resolution,

  and returns an inference tensor of shape (1, 7, 101, 241).
  """
  ds_temp = xr.open_dataset(os.path.join(DATA_DIR, "temp_sample.nc"))
  ds_sal = xr.open_dataset(os.path.join(DATA_DIR, "salinity_sample.nc"))
  ds_curr = xr.open_dataset(os.path.join(DATA_DIR, "currents_sample.nc"))
  ds_ssh = xr.open_dataset(os.path.join(DATA_DIR, "sealevel_sample.nc"))
  ds_wind = xr.open_dataset(os.path.join(DATA_DIR, "wind_sample.nc"))

  # Define the official 0.25 degree grid
  new_lat = np.arange(5.0, 30.25, 0.25)
  new_lon = np.arange(45.0, 105.25, 0.25)

  # Establish master reference grid using temperature
  temp_master = ds_temp["thetao"].interp(latitude=new_lat, longitude=new_lon)

  # Align and regrid all datasets using interp_like to handle differing coordinate names
  sal_regridded = ds_sal["so"].interp_like(temp_master)
  curr_u_regridded = ds_curr["uo"].interp_like(temp_master)
  curr_v_regridded = ds_curr["vo"].interp_like(temp_master)
  ssh_regridded = ds_ssh["total_sea_level"].interp_like(temp_master)
  wind_u_regridded = ds_wind["eastward_wind"].interp_like(temp_master)
  wind_v_regridded = ds_wind["northward_wind"].interp_like(temp_master)

  # Extract the slice safely
  try:
    sub_temp = temp_master.sel(time=date_str)
    sub_sal = sal_regridded.sel(time=date_str)
    sub_curr_u = curr_u_regridded.sel(time=date_str)
    sub_curr_v = curr_v_regridded.sel(time=date_str)
    sub_ssh = ssh_regridded.sel(time=date_str)
    sub_wind_u = wind_u_regridded.sel(time=date_str)
    sub_wind_v = wind_v_regridded.sel(time=date_str)
  except Exception:
    sub_temp = temp_master.isel(time=0)
    sub_sal = sal_regridded.isel(time=0)
    sub_curr_u = curr_u_regridded.isel(time=0)
    sub_curr_v = curr_v_regridded.isel(time=0)
    sub_ssh = ssh_regridded.isel(time=0)
    sub_wind_u = wind_u_regridded.isel(time=0)
    sub_wind_v = wind_v_regridded.isel(time=0)

  def ensure_2d(da):
    return np.squeeze(da.values)

  arrays = [
      ensure_2d(sub_temp),
      ensure_2d(sub_sal),
      ensure_2d(sub_curr_u),
      ensure_2d(sub_curr_v),
      ensure_2d(sub_ssh),
      ensure_2d(sub_wind_u),
      ensure_2d(sub_wind_v),
  ]

  # Stack channels -> (Channels, Lat, Lon)
  input_features = np.stack(arrays, axis=0)

  # Clean and normalize
  input_features = np.nan_to_num(input_features, nan=0.0)
  input_features_norm = normalize(input_features)

  # Add batch dimension -> (1, 7, 101, 241)
  return np.expand_dims(input_features_norm, axis=0)