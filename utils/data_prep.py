import numpy as np
import xarray as xr
import os

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

    print("Extracting and dynamically regridding arrays...")
    
    # These four are already perfectly aligned
    sst = ds_temp['thetao'].values
    sal = ds_sal['so'].values
    u_curr = ds_curr['uo'].values
    v_curr = ds_curr['vo'].values
    
    # SSH is hourly. We use interp_like() to automatically resample it to daily, matching the temp file.
    ssh = ds_ssh['total_sea_level'].interp_like(ds_temp).values
    
    # Wind is hourly AND a different grid resolution. interp_like() fixes both instantly.
    u_wind_raw = ds_wind['eastward_wind'].interp_like(ds_temp).values
    v_wind_raw = ds_wind['northward_wind'].interp_like(ds_temp).values
    
    # Wind is also missing the 'depth' dimension, so we add a dummy dimension using np.expand_dims
    u_wind = np.expand_dims(u_wind_raw, axis=1)
    v_wind = np.expand_dims(v_wind_raw, axis=1)
    
    # Target variable
    target_thetao = ds_target['bottomT'].values

    print("Stacking features, masking landmass NaNs, and normalizing...")
    # Stack the 7 surface inputs into one array along the channel axis (Time, Channels, Lat, Lon)
    input_features = np.stack([sst, sal, u_curr, v_curr, ssh, u_wind, v_wind], axis=1)

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
    
    print(f"Success! Saved X_inputs {input_features_norm.shape} and Y_target {target_thetao_norm.shape} to {OUTPUT_DIR}/")

if __name__ == "__main__":
    load_and_preprocess()