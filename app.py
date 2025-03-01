import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta

# ---------------------------------
# Title and Overview
# ---------------------------------
st.title("Energy Consumption Optimizer Dashboard (PyTorch)")
st.write("""
This dashboard simulates energy consumption data, detects anomalies using a PyTorch autoencoder, 
and forecasts future usage with a PyTorch regression model.
""")

# ---------------------------------
# Simulated Energy Consumption Data
# ---------------------------------
st.header("Simulated Energy Consumption Data")
np.random.seed(42)
n_days = 100
dates = pd.date_range(end=datetime.today(), periods=n_days)

# Simulate a gradual increasing trend with noise
base_consumption = 20 + np.linspace(0, 5, n_days)
noise = np.random.normal(0, 2, n_days)
energy_usage = base_consumption + noise

# Inject random anomaly spikes
anomaly_indices = np.random.choice(n_days, size=5, replace=False)
energy_usage[anomaly_indices] += np.random.normal(15, 5, 5)

data = pd.DataFrame({"date": dates, "energy_usage": energy_usage})
data.set_index("date", inplace=True)
st.line_chart(data["energy_usage"])

# ---------------------------------
# Anomaly Detection using a PyTorch Autoencoder
# ---------------------------------
st.header("Anomaly Detection with PyTorch Autoencoder")

# Create sliding windows from the energy usage data
window_size = 5
X_windows = []
center_indices = []
for i in range(len(energy_usage) - window_size + 1):
    window = energy_usage[i:i+window_size]
    X_windows.append(window)
    center_indices.append(i + window_size // 2)
X_windows = np.array(X_windows)  # shape: (num_windows, window_size)

# Convert windows to a PyTorch tensor
X_tensor = torch.tensor(X_windows, dtype=torch.float32)

# Define a simple autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_size // 2, input_size),
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

input_size = window_size
autoencoder = Autoencoder(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

# Train the autoencoder
n_epochs = 200
for epoch in range(n_epochs):
    optimizer.zero_grad()
    outputs = autoencoder(X_tensor)
    loss = criterion(outputs, X_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        st.write(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")

# Compute reconstruction error for each window
with torch.no_grad():
    reconstructed = autoencoder(X_tensor)
    mse_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()

# Determine threshold (mean + 2*std)
threshold = mse_errors.mean() + 2 * mse_errors.std()
anomaly_windows = np.where(mse_errors > threshold)[0]
anomaly_days = [center_indices[i] for i in anomaly_windows]

st.write("Detected anomaly days (by index):", anomaly_days)

# Plot energy usage and highlight anomalies
fig, ax = plt.subplots()
ax.plot(data.index, data["energy_usage"], label="Energy Usage")
for i, idx in enumerate(anomaly_days):
    ax.scatter(data.index[idx], data["energy_usage"].iloc[idx], color='red', 
               label="Anomaly" if i == 0 else "")
ax.set_xlabel("Date")
ax.set_ylabel("Energy Usage (kWh)")
ax.legend()
st.pyplot(fig)

# ---------------------------------
# Predictive Modeling using a PyTorch Regression Model
# ---------------------------------
st.header("Predictive Modeling with PyTorch Regression")

# Prepare data: use day index as feature and energy usage as target
X_reg = np.arange(n_days).reshape(-1, 1).astype(np.float32)
y_reg = energy_usage.reshape(-1, 1).astype(np.float32)
X_reg_tensor = torch.tensor(X_reg)
y_reg_tensor = torch.tensor(y_reg)

# Define a simple regression model
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)

reg_model = RegressionModel()
criterion_reg = nn.MSELoss()
optimizer_reg = optim.Adam(reg_model.parameters(), lr=0.01)

# Train the regression model
n_epochs_reg = 1000
for epoch in range(n_epochs_reg):
    optimizer_reg.zero_grad()
    outputs = reg_model(X_reg_tensor)
    loss_reg = criterion_reg(outputs, y_reg_tensor)
    loss_reg.backward()
    optimizer_reg.step()
    if (epoch + 1) % 200 == 0:
        st.write(f"Regression Epoch [{epoch+1}/{n_epochs_reg}], Loss: {loss_reg.item():.4f}")

# Forecast next 7 days
future_indices = np.arange(n_days, n_days + 7).reshape(-1, 1).astype(np.float32)
future_tensor = torch.tensor(future_indices)
with torch.no_grad():
    future_predictions = reg_model(future_tensor).numpy()

future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, 8)]
future_df = pd.DataFrame({"date": future_dates, "predicted_energy_usage": future_predictions.flatten()})
future_df.set_index("date", inplace=True)
st.write("Energy Usage Forecast for Next 7 Days:")
st.line_chart(future_df)

# ---------------------------------
# Personalized Insights
# ---------------------------------
st.header("Personalized Insights")
avg_usage = data["energy_usage"].mean()
if future_predictions[-1] > avg_usage * 1.1:
    st.write("**Alert:** The predicted energy usage for the coming days is higher than average. Consider reviewing your peak hour consumption and exploring energy-saving measures!")
else:
    st.write("The predicted energy usage is within a normal range. Keep up your current efficient energy practices!")

st.write("""
This demo integrates PyTorch for anomaly detection via an autoencoder and predictive modeling through a simple regression model,
providing actionable insights to help optimize energy consumption.
""")
