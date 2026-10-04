# OceanEmbed: 3D Subsurface Ocean Temperature Reconstruction & Marine Heatwave Early Warning
**Team INNOVEXA** | Smart India Hackathon (SIH 2026)  
**Ministry of Earth Sciences (MoES) | Problem Statement 26066**

---

## 🌊 Overview
**OceanEmbed** acts as an AI "X-Ray" for the ocean. While traditional satellite observations capture surface phenomena, deep ocean monitoring has historically relied on expensive, sparse *in-situ* ARGO deployments. 

OceanEmbed bridges this critical observational data gap by deploying a **2D Spatial Convolutional Autoencoder with a U-Net backbone**. The framework fuses 7 multi-variable surface-level satellite inputs—including Sea Surface Temperature (SST), Sea Surface Salinity (SSS), Sea Surface Height/Anomaly (SSH/SLA), U/V surface currents, and U/V wind vectors—into a compact **64-dimensional latent Z-space**. This compact latent representation is then decoded to reconstruct continuous 3D subsurface thermal profiles across all 15 standard INCOIS depth levels (0 m - 1000 m).

---

## 🚀 Key Features & Architecture
- **2D Spatial U-Net Autoencoder:** Leverages skip-pooling architecture to preserve high-resolution spatial gradients and non-linear relationships from surface observations into a 64-D latent manifold (`Z-space`).
- **Interactive 3D Subsurface Visualization:** Real-time depth slicing (0 m - 1000 m) with volumetric basin-wide spatial rendering and vertical transect generation via **Streamlit & Plotly**.
- **Oceanographic Indicator Suite:** Computes critical physical parameters including Mixed Layer Depth (MLD), the 20°C Isotherm depth, and Tropical Cyclone Heat Potential (TCHP).
- **Rigorous Validation Engine:** Evaluates model performance against independent Copernicus reanalysis targets and *in-situ* INCOIS ARGO profile data, computing depth-wise Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and Pearson correlation (R).
- **Disaster Intelligence (MoES Theme):** Automated anomaly detection framework flagging subsurface Marine Heatwaves (MHWs) that remain entirely invisible to surface-only satellite observations.

---

## 📊 Core Datasets & References
* **Target Subsurface Ground Truth:** **GLORYS12V1 Global Ocean Reanalysis** (0 m - 1000 m depth) via Copernicus Marine Service (`DOI:10.48670/moi-00021`).
* **Surface Proxy Inputs:** 
  * *SST:* OSTIA Global Foundation (Copernicus)
  * *SSS:* SMAP/SMOS (NASA JPL)
  * *SSH/SLA:* DUACS Gridded Sea Level Anomalies
* **Independent Validation Data:** INCOIS Live Access Server (LAS) Gridded ARGO Profiling Floats.

---

## 🛠 Tech Stack
- **Core Engine:** Python 3.10+, PyTorch (Custom 2D U-Net Autoencoder)
- **Data Harmonization & Analytics:** Xarray, NumPy, Pandas, NetCDF4
- **Interactive UI & Dashboard:** Streamlit, Plotly (3D Volumetric Rendering)

---

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hiralirs/OceanEmbed-SIH.git
   cd OceanEmbed-SIH
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ````

3. **Launch the Streamlit dashboard:**
   ```bash
   streamlit run app.py
   ```


## 🛡️ PoC Data Notice & Validation Architecture
*For this demonstration and prototype evaluation, a modular data abstraction layer is utilized. While the production-ready architecture is configured to ingest live NetCDF telemetry streams directly from INCOIS OPeNDAP servers, the PoC environment uses a rigorous statistical simulation conforming to real-world ARGO float structures to validate the error-calculation and anomaly-flagging pipeline instantly without network latency bottlenecks.*