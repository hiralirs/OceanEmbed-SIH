# OceanEmbed: 3D Subsurface Ocean Temperature Reconstruction
**Team INNOVEXA** | Smart India Hackathon (SIH 2026)  
**Ministry of Earth Sciences (MoES) | Problem Statement 26066**

## 🌊 Overview
OceanEmbed acts as an "AI X-Ray" for the ocean. Instead of relying on expensive, sparse underwater sensors, this framework uses a Convolutional Autoencoder to map 7 surface-level satellite variables (SST, SSS, SSH/SLA, Surface Currents, and Wind Vectors) into a 64-D latent space (`Z-Space`), reconstructing full-depth thermal profiles across 15 standard INCOIS depth levels (0m–1000m).

## 🚀 Key Features
- **Interactive 3D Subsurface View:** Dynamic depth slicing (0m–1000m) with basin-wide spatial visualization.
- **Latent Manifold Explorer (Z-Space):** Visualizes the model's non-linear compression of ocean dynamics.
- **Validation Suite:** Compares AI predictions against in-situ ARGO float data (calculating RMSE, Pearson correlation, MLD, and the 20°C Isotherm).
- **Disaster Intelligence:** Automated early warning system for subsurface Marine Heatwaves (MHW).

## 🛠️ Tech Stack
- **Python 3.10+**
- **PyTorch** (Convolutional Autoencoder)
- **Streamlit & Plotly** (Interactive UI & Visualizations)
- **Xarray, NumPy, Pandas** (Data Harmonization)

## ⚙️ How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/hiralirs/OceanEmbed-SIH.git
   cd OceanEmbed-SIH
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Launch the dashboard:
   ```bash
   streamlit run app.py
   
