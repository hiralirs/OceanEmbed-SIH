import os
import numpy as np
import xarray as xr

# Define Bounding Box & Paths
DATA_DIR = "data"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize(array):
  """Applies Min-Max normalization, scaling values between 0 and 1."""
  min_val = np.nanmin(array)
  max_val = np.nanmax(array)
  if max_val == min_val:
    return np.zeros_like(array)
  return (array - min_val) / (max_val - min_val)


def load_and_preprocess():
  print("Loading NetCDF sample files...")

  # Load individual datasets
  ds_temp = xr.open_dataset(os.path.join(DATA_DIR, "temp_sample.nc"))
  ds_sal = xr.open_dataset(os.path.join(DATA_DIR, "salinity_sample.nc"))
  ds_curr = xr.open_dataset(os.path.join(DATA_DIR, "currents_sample.nc"))
  ds_ssh = xr.open_dataset(os.path.join(DATA_DIR, "sealevel_sample.nc"))
  ds_wind = xr.open_dataset(os.path.join(DATA_DIR, "wind_sample.nc"))
  ds_target = xr.open_dataset(os.path.join(DATA_DIR, "glorys_sample.nc"))

  print("Regridding to official 0.25 degree SIH resolution...")
  # Define the official 0.25 degree grid for the North Indian Ocean region
  new_lat = np.arange(5.0, 30.25, 0.25)
  new_lon = np.arange(45.0, 105.25, 0.25)

  # Establish a new master grid using the correct 'latitude' and 'longitude' names
  master_grid = ds_temp.interp(latitude=new_lat, longitude=new_lon)

  sst = master_grid["thetao"].values
  sal = ds_sal["so"].interp_like(master_grid).values
  u_curr = ds_curr["uo"].interp_like(master_grid).values
  v_curr = ds_curr["vo"].interp_like(master_grid).values
  ssh = ds_ssh["total_sea_level"].interp_like(master_grid).values

  u_wind_raw = ds_wind["eastward_wind"].interp_like(master_grid).values
  v_wind_raw = ds_wind["northward_wind"].interp_like(master_grid).values

  def fix_shape(arr):
    arr = np.squeeze(arr)
    if arr.ndim == 2:  # Shape: (Lat, Lon)
      arr = np.expand_dims(arr, axis=0)  # Shape: (1, Lat, Lon)
    elif arr.ndim == 3:  # Shape: (Time, Lat, Lon)
      pass
    return arr

  arrays = [
      fix_shape(sst),
      fix_shape(sal),
      fix_shape(u_curr),
      fix_shape(v_curr),
      fix_shape(ssh),
      fix_shape(u_wind_raw),
      fix_shape(v_wind_raw),
  ]

  print("Extracting target depths...")
  target_regridded = ds_target.interp(latitude=new_lat, longitude=new_lon)

  # Safely handle the depth extraction depending on what is currently in the file
  target_var = "thetao" if "thetao" in target_regridded else "bottomT"
  if target_regridded.dims.get("depth", 1) > 1:
    target_depths = [
        0,
        5,
        10,
        20,
        30,
        50,
        75,
        100,
        125,
        150,
        200,
        300,
        500,
        700,
        1000,
    ]
    target_thetao = (
        target_regridded[target_var]
        .sel(depth=target_depths, method="nearest")
        .values
    )
  else:
    print(
        "WARNING: glorys_sample.nc only contains 1 depth layer. Proceeding with"
        " 1 layer."
    )
    target_thetao = target_regridded[target_var].values

  print("Stacking features, masking landmass NaNs, and normalizing...")
  # Stack the 7 surface inputs into one array along the channel axis -> (Time, Channels, Lat, Lon)
  input_features = np.stack(arrays, axis=1)

  # Convert landmass NaNs to 0.0 so they don't break the neural network
  input_features = np.nan_to_num(input_features, nan=0.0)
  target_thetao = np.nan_to_num(target_thetao, nan=0.0)

  # Normalize inputs for stable deep learning training
  input_features_norm = normalize(input_features)
  target_thetao_norm = normalize(target_thetao)

  print("Saving processed tensors...")
  # Export as clean NumPy arrays
  np.save(os.path.join(OUTPUT_DIR, "X_inputs.npy"), input_features_norm)
  np.save(os.path.join(OUTPUT_DIR, "Y_target.npy"), target_thetao_norm)

  print(
      f"Success! Saved X_inputs {input_features_norm.shape} and Y_target"
      f" {target_thetao_norm.shape} to {OUTPUT_DIR}/"
  )


# --- STREAMLIT PIPELINE BRIDGE (MATCHES APP.PY EXPECTATION) ---
import torch


def load_and_preprocess_surface(nc_path):
  """Bridge function expected by app.py to load real inference data."""
  new_lat = np.arange(5.0, 30.25, 0.25)
  new_lon = np.arange(45.0, 105.25, 0.25)

  # Load sample datasets to construct inference tensor
  ds_temp = xr.open_dataset(os.path.join(DATA_DIR, "temp_sample.nc"))
  ds_sal = xr.open_dataset(os.path.join(DATA_DIR, "salinity_sample.nc"))
  ds_curr = xr.open_dataset(os.path.join(DATA_DIR, "currents_sample.nc"))
  ds_ssh = xr.open_dataset(os.path.join(DATA_DIR, "sealevel_sample.nc"))
  ds_wind = xr.open_dataset(os.path.join(DATA_DIR, "wind_sample.nc"))

  master_grid = ds_temp.interp(latitude=new_lat, longitude=new_lon)

  sst = np.nan_to_num(np.squeeze(master_grid["thetao"].values), nan=0.0)
  sal = np.nan_to_num(
      np.squeeze(ds_sal["so"].interp_like(master_grid).values), nan=0.0
  )
  u_curr = np.nan_to_num(
      np.squeeze(ds_curr["uo"].interp_like(master_grid).values), nan=0.0
  )
  v_curr = np.nan_to_num(
      np.squeeze(ds_curr["vo"].interp_like(master_grid).values), nan=0.0
  )
  ssh = np.nan_to_num(
      np.squeeze(ds_ssh["total_sea_level"].interp_like(master_grid).values),
      nan=0.0,
  )
  u_wind = np.nan_to_num(
      np.squeeze(ds_wind["eastward_wind"].interp_like(master_grid).values),
      nan=0.0,
  )
  v_wind = np.nan_to_num(
      np.squeeze(ds_wind["northward_wind"].interp_like(master_grid).values),
      nan=0.0,
  )

  # Ensure each has a leading time/batch dimension if 2D
  def ensure_3d(arr):
    if arr.ndim == 2:
      return np.expand_dims(arr, axis=0)
    return arr

  features = np.stack(
      [
          ensure_3d(sst),
          ensure_3d(sal),
          ensure_3d(u_curr),
          ensure_3d(v_curr),
          ensure_3d(ssh),
          ensure_3d(u_wind),
          ensure_3d(v_wind),
      ],
      axis=1,
  )

  # Take the first time step for single-frame UI inference
  tensor_data = torch.tensor(
      features[0:1], dtype=torch.float32
  )  # Shape: (1, 7, 101, 241)
  return tensor_data, new_lat, new_lon


if __name__ == "__main__":
  load_and_preprocess()