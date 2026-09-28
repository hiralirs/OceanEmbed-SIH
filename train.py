import os
import numpy as np
from models.oceanembed_net import OceanEmbedNet
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_model():
  print("Loading Track A data...")
  X = np.load("data/processed/X_inputs.npy")
  Y = np.load("data/processed/Y_target.npy")

  X_tensor = torch.tensor(X, dtype=torch.float32)
  Y_tensor = torch.tensor(Y, dtype=torch.float32)

  # Ensure Y matches the batch size of X if it was saved with a single sample
  if Y_tensor.shape[0] == 1 and X_tensor.shape[0] > 1:
    Y_tensor = Y_tensor.repeat(X_tensor.shape[0], 1, 1, 1)

  # Ensure Y has a channel dimension -> (Batch, Channels, Lat, Lon)
  if Y_tensor.ndim == 3:
    Y_tensor = torch.unsqueeze(Y_tensor, dim=1)

  dataset = TensorDataset(X_tensor, Y_tensor)
  dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

  model = OceanEmbedNet(in_channels=7, out_channels=1)
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  epochs = 30
  print(f"Starting training loop for {epochs} epochs...")

  for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_X, batch_Y in dataloader:
      optimizer.zero_grad()
      # The model returns (reconstruction, latent_features)
      predictions, _ = model(batch_X)
      loss = criterion(predictions, batch_Y)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    if (epoch + 1) % 5 == 0 or epoch == 0:
      print(f"Epoch [{epoch+1}/{epochs}], MSE Loss: {avg_loss:.6f}")

  print("\nCalculating Final Metrics...")
  model.eval()
  with torch.no_grad():
    all_preds, _ = model(X_tensor)
    final_mse = criterion(all_preds, Y_tensor).item()
    rmse = np.sqrt(final_mse)

    # Flatten arrays to calculate Pearson correlation
    preds_flat = all_preds.numpy().flatten()
    y_flat = Y_tensor.numpy().flatten()
    pearson_corr, _ = pearsonr(preds_flat, y_flat)

  print(f"RMSE: {rmse:.4f}")
  print(f"Pearson Correlation: {pearson_corr:.4f}")

  os.makedirs("models", exist_ok=True)
  torch.save(model.state_dict(), "models/best_model.pth")
  print("\nTraining complete! Weights saved to models/best_model.pth")


if __name__ == "__main__":
  train_model()