import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from google import genai

st.title("Ontario Household Energy Dashboard & Carbon Footprint Calculator")

################################################################################
# Section 1: Load Dataset
################################################################################

st.header("Dataset Overview")
try:
    df = pd.read_csv("factors.csv")
    st.write("Loaded factors.csv successfully.")
    st.dataframe(df.head(10))
except Exception as e:
    st.error("Failed to load factors.csv: " + str(e))
    st.stop()

################################################################################
# Outlier Removal (using Total_kWh)
################################################################################

st.header("Outlier Removal")
# Remove outliers based on Total_kWh using the IQR method
Q1 = df["Total_kWh"].quantile(0.25)
Q3 = df["Total_kWh"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_filtered = df[(df["Total_kWh"] >= lower_bound) & (df["Total_kWh"] <= upper_bound)]
st.write(f"Removed {len(df) - len(df_filtered)} outlier record(s) based on Total_kWh.")
df = df_filtered.reset_index(drop=True)
st.dataframe(df.head(10))

################################################################################
# Section 2: Anomaly Detection & Forecasting (using PyTorch)
################################################################################

st.header("Anomaly Detection on Total Consumption")
window_size = 3  # window size in months
energy_values = df["Total_kWh"].values

if len(energy_values) < window_size:
    st.error("Not enough data for anomaly detection after outlier removal.")
else:
    X_windows = []
    center_indices = []
    for i in range(len(energy_values) - window_size + 1):
        window = energy_values[i : i + window_size]
        X_windows.append(window)
        center_indices.append(i + window_size // 2)
    
    # Only proceed if we have at least one window
    if len(X_windows) == 0:
        st.error("No valid windows found for anomaly detection.")
    else:
        X_windows = np.array(X_windows, dtype=np.float32)
        X_tensor = torch.tensor(X_windows)

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

        autoencoder = Autoencoder(window_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

        n_epochs = 200
        st.write("Training autoencoder on Total_kWh windows...")
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            outputs = autoencoder(X_tensor)
            loss = criterion(outputs, X_tensor)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 50 == 0:
                st.write(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")

        with torch.no_grad():
            reconstructed = autoencoder(X_tensor)
            mse_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
        threshold = mse_errors.mean() + 2 * mse_errors.std()
        anomaly_windows = np.where(mse_errors > threshold)[0]
        anomaly_months = [center_indices[i] for i in anomaly_windows]

        st.write("Detected anomaly indices (by record index):", anomaly_months)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df["Total_kWh"], label="Total_kWh")
        for i, idx in enumerate(anomaly_months):
            ax.scatter(df.index[idx], df["Total_kWh"].iloc[idx], color='red', label="Anomaly" if i == 0 else "")
        ax.set_xlabel("Record Index")
        ax.set_ylabel("Total_kWh")
        ax.legend()
        st.pyplot(fig)

    st.header("Predictive Modeling for Total Consumption")
    n_samples = len(df)
    X_reg = np.arange(n_samples).reshape(-1, 1).astype(np.float32)
    y_reg = df["Total_kWh"].values.reshape(-1, 1).astype(np.float32)
    X_reg_tensor = torch.tensor(X_reg)
    y_reg_tensor = torch.tensor(y_reg)

    class RegressionModel(nn.Module):
        def __init__(self):
            super(RegressionModel, self).__init__()
            self.linear = nn.Linear(1, 1)
        def forward(self, x):
            return self.linear(x)

    reg_model = RegressionModel()
    criterion_reg = nn.MSELoss()
    optimizer_reg = optim.Adam(reg_model.parameters(), lr=0.01)

    n_epochs_reg = 1000
    st.write("Training regression model for forecasting Total_kWh...")
    for epoch in range(n_epochs_reg):
        optimizer_reg.zero_grad()
        outputs = reg_model(X_reg_tensor)
        loss_reg = criterion_reg(outputs, y_reg_tensor)
        loss_reg.backward()
        optimizer_reg.step()
        if (epoch + 1) % 200 == 0:
            st.write(f"Epoch [{epoch+1}/{n_epochs_reg}], Loss: {loss_reg.item():.4f}")

    future_indices = np.arange(n_samples, n_samples + 3).reshape(-1, 1).astype(np.float32)
    future_tensor = torch.tensor(future_indices)
    with torch.no_grad():
        future_predictions = reg_model(future_tensor).numpy()

    future_records = [f"Record {n_samples+j+1}" for j in range(3)]
    future_df = pd.DataFrame({
        "Record": future_records,
        "Predicted_Total_kWh": future_predictions.flatten()
    })
    st.write("Forecast for the next 3 records (Total_kWh):")
    st.line_chart(future_df.set_index("Record"))

################################################################################
# Section 3: Dimensionality Reduction (PCA) & Correlation Analysis
################################################################################

st.header("Dimensionality Reduction (PCA)")
features = ["AvgTemp_C", "AvgHumidity_%", "Occupants", "HouseSize_sqft",
            "OffPeak_kWh", "MidPeak_kWh", "OnPeak_kWh", "Total_kWh", "Total_cost_$"]
X_features = df[features].values.astype(np.float32)
X_tensor_feat = torch.tensor(X_features)
X_centered = X_tensor_feat - X_tensor_feat.mean(dim=0, keepdim=True)
U, S, V = torch.linalg.svd(X_centered, full_matrices=False)

# Set k = 3, but if effective rank is less than 3, pad with zeros for plotting
k = 3
if V.shape[0] < k:
    k = V.shape[0]
principal_components = V[:k, :].T
X_projected = X_centered @ principal_components

# If we got less than 3 dimensions, pad with zeros to allow 3D plotting.
if X_projected.shape[1] < 3:
    missing = 3 - X_projected.shape[1]
    pad = torch.zeros(X_projected.shape[0], missing)
    X_projected = torch.cat([X_projected, pad], dim=1)

# Create a 3D scatter plot
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
fig3d = plt.figure(figsize=(8, 6))
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.scatter(X_projected[:, 0].numpy(), X_projected[:, 1].numpy(), X_projected[:, 2].numpy(), 
             c='blue', alpha=0.7)
ax3d.set_xlabel("PC 1")
ax3d.set_ylabel("PC 2")
ax3d.set_zlabel("PC 3")
ax3d.set_title("PCA Projection (3D)")
st.pyplot(fig3d)

st.header("Pearson Correlation Analysis")
def pearson_correlation_matrix(X: torch.Tensor) -> torch.Tensor:
    n = X.size(0)
    X_center = X - X.mean(dim=0, keepdim=True)
    cov = torch.mm(X_center.T, X_center) / (n - 1)
    std = torch.sqrt(torch.diag(cov))
    denom = std.unsqueeze(0) * std.unsqueeze(1)
    corr = cov / denom
    return corr

corr_matrix = pearson_correlation_matrix(X_tensor_feat)
corr_np = corr_matrix.numpy()
fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
cax = ax_corr.matshow(corr_np, cmap='coolwarm')
fig_corr.colorbar(cax)
ax_corr.set_xticks(range(len(features)))
ax_corr.set_yticks(range(len(features)))
ax_corr.set_xticklabels(features, rotation=90, fontsize=8)
ax_corr.set_yticklabels(features, fontsize=8)
ax_corr.set_title("Pearson Correlation Matrix", pad=20)
st.pyplot(fig_corr)

################################################################################
# Section 4: Carbon Footprint Calculator & Gemini API Recommendations
################################################################################

st.header("Carbon Footprint Calculator & Recommendations")

st.markdown("""
Enter your monthly consumption data below:
- **Electricity (kWh)**
- **Natural Gas (therms)**
- **Vehicle Miles Driven**

The app will estimate your monthly CO₂ emissions and then use the Gemini API to offer recommendations to reduce your carbon footprint.
""")

electricity = st.number_input("Monthly Electricity Consumption (kWh)", value=500.0, min_value=0.0)
gas = st.number_input("Monthly Natural Gas Consumption (therms)", value=50.0, min_value=0.0)
vehicle = st.number_input("Monthly Vehicle Miles Driven", value=1000.0, min_value=0.0)

# Conversion factors (typical values)
electricity_factor = 0.075  # kg CO₂ per kWh
gas_factor = 5.3            # kg CO₂ per therm
vehicle_factor = 0.411      # kg CO₂ per mile

carbon_electricity = electricity * electricity_factor
carbon_gas = gas * gas_factor
carbon_vehicle = vehicle * vehicle_factor
total_carbon = carbon_electricity + carbon_gas + carbon_vehicle

st.write(f"**Estimated Monthly Carbon Emissions:** {total_carbon:.2f} kg CO₂")

if st.button("Get Recommendations to Reduce Emissions"):
    prompt = (
        f"I'm a sustainability expert. A user has a monthly carbon footprint of {total_carbon:.2f} kg CO₂. "
        f"They consume {electricity} kWh of electricity, {gas} therms of natural gas, and drive {vehicle} miles per month. "
        f"Provide actionable, friendly recommendations to lower their carbon emissions."
    )
    client = genai.Client(api_key="AIzaSyB4VHerf7dMxw35V6TS8tru3E_i7_wkLVU")
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt])
    st.write(response.text)

st.write("""
This app integrates data analysis (with outlier removal), anomaly detection, forecasting, PCA visualization, 
and a carbon footprint calculator with personalized sustainability recommendations powered by the Gemini API.
""")

################################################################################
# Section 5: Conversion Factor Regression (Using PyTorch)
################################################################################

st.header("Conversion Factor Regression (Using PyTorch)")

# Ensure the required columns are present
required_cols = ["OffPeak_kWh", "MidPeak_kWh", "OnPeak_kWh", "Total_cost_$"]
if all(col in df.columns for col in required_cols):
    features_conv = ["OffPeak_kWh", "MidPeak_kWh", "OnPeak_kWh"]
    X_factors = df[features_conv].values.astype(np.float32)
    y_factors = df["Total_cost_$"].values.reshape(-1, 1).astype(np.float32)
    
    X_factors_tensor = torch.tensor(X_factors)
    y_factors_tensor = torch.tensor(y_factors)
    
    # Define a simple linear regression model for conversion factors
    class ConversionRegression(nn.Module):
        def __init__(self):
            super(ConversionRegression, self).__init__()
            self.linear = nn.Linear(3, 1)
        def forward(self, x):
            return self.linear(x)
    
    conversion_model = ConversionRegression()
    criterion_conv = nn.MSELoss()
    optimizer_conv = optim.Adam(conversion_model.parameters(), lr=0.01)
    
    n_epochs_conv = 1000
    st.write("Training conversion factor regression model...")
    for epoch in range(n_epochs_conv):
        optimizer_conv.zero_grad()
        outputs_conv = conversion_model(X_factors_tensor)
        loss_conv = criterion_conv(outputs_conv, y_factors_tensor)
        loss_conv.backward()
        optimizer_conv.step()
        if (epoch + 1) % 200 == 0:
            st.write(f"Epoch [{epoch+1}/{n_epochs_conv}], Loss: {loss_conv.item():.4f}")
    
    # Extract estimated conversion factors and fixed charge
    coefficients = conversion_model.linear.weight.data.numpy().flatten()
    intercept = conversion_model.linear.bias.data.numpy().item()
    
    st.write("### Estimated Conversion Factors (in $ per kWh):")
    st.write(f"OffPeak: {coefficients[0]:.4f}")
    st.write(f"MidPeak: {coefficients[1]:.4f}")
    st.write(f"OnPeak: {coefficients[2]:.4f}")
    st.write("### Fixed Charge (Intercept): {:.4f} $".format(intercept))
else:
    st.error("The dataset does not contain the required columns: " + ", ".join(required_cols))
