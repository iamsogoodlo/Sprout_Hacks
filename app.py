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
# Section 1: Synthetic Energy Consumption Dataset (10 Years, Monthly)
################################################################################

st.header("Synthetic Energy Consumption Dataset (2010-2019)")
n_months = 120  # 10 years
records = []
for i in range(n_months):
    year = 2010 + i // 12
    month = (i % 12) + 1
    
    # Weather: Base values by season (with noise)
    if month in [12, 1, 2]:
        base_temp = -5
        base_humidity = 75
    elif month in [3, 4, 5]:
        base_temp = 10
        base_humidity = 65
    elif month in [6, 7, 8]:
        base_temp = 25
        base_humidity = 60
    else:
        base_temp = 15
        base_humidity = 70
    AvgTemp_C = np.round(base_temp + np.random.normal(0, 2), 1)
    AvgHumidity = np.round(base_humidity + np.random.normal(0, 5), 1)
    
    # Household characteristics
    if year < 2013:
        occupants = 2
        AC = 0
    elif year < 2016:
        occupants = 3
        AC = 1
    else:
        occupants = 4
        AC = 1
    if year < 2015:
        house_size = 1500
        ElectricHeating = 0
    else:
        house_size = 2500
        ElectricHeating = 1
    EV = 1 if year >= 2018 else 0
    
    # Base energy consumption (kWh) varies by season and appliance use
    if month in [12, 1, 2] and ElectricHeating:
        base_usage = 1800
    elif month in [6, 7, 8] and AC:
        base_usage = 800
    else:
        base_usage = 400
    usage = base_usage * (1 + (occupants - 2) * 0.2) * (house_size / 1500)
    usage = usage * (1 + np.random.normal(0, 0.1))  # ±10% noise
    
    # Allocate usage to Time-Of-Use periods
    OffPeak_kWh = usage * 0.65
    MidPeak_kWh = usage * 0.18
    OnPeak_kWh  = usage * 0.17
    
    # Historical TOU rates (in cents/kWh)
    if year <= 2014:
        OffPeak_rate = 5.0; MidPeak_rate = 8.0; OnPeak_rate  = 9.3
    elif year <= 2016:
        OffPeak_rate = 7.7; MidPeak_rate = 11.4; OnPeak_rate  = 14.0
    elif year == 2017:
        OffPeak_rate = 6.5; MidPeak_rate = 9.5; OnPeak_rate  = 13.2
    elif year == 2018:
        OffPeak_rate = 6.5; MidPeak_rate = 9.4; OnPeak_rate  = 13.2
    else:
        OffPeak_rate = 10.1; MidPeak_rate = 14.4; OnPeak_rate  = 20.8

    # Calculate cost ($); convert cents to dollars
    OffPeak_cost = OffPeak_kWh * (OffPeak_rate / 100)
    MidPeak_cost = MidPeak_kWh * (MidPeak_rate / 100)
    OnPeak_cost  = OnPeak_kWh  * (OnPeak_rate / 100)
    Fixed_charge = 30.0  # constant monthly fee
    
    Total_kWh = OffPeak_kWh + MidPeak_kWh + OnPeak_kWh
    Total_cost = OffPeak_cost + MidPeak_cost + OnPeak_cost + Fixed_charge
    
    records.append({
        "Year": year,
        "Month": month,
        "AvgTemp_C": AvgTemp_C,
        "AvgHumidity_%": AvgHumidity,
        "Occupants": occupants,
        "HouseSize_sqft": house_size,
        "ElectricHeating": ElectricHeating,
        "AC": AC,
        "EV": EV,
        "OffPeak_kWh": np.round(OffPeak_kWh, 1),
        "MidPeak_kWh": np.round(MidPeak_kWh, 1),
        "OnPeak_kWh": np.round(OnPeak_kWh, 1),
        "OffPeak_rate_c/kWh": OffPeak_rate,
        "MidPeak_rate_c/kWh": MidPeak_rate,
        "OnPeak_rate_c/kWh": OnPeak_rate,
        "OffPeak_cost_$": np.round(OffPeak_cost, 2),
        "MidPeak_cost_$": np.round(MidPeak_cost, 2),
        "OnPeak_cost_$": np.round(OnPeak_cost, 2),
        "Fixed_charge_$": Fixed_charge,
        "Total_kWh": np.round(Total_kWh, 1),
        "Total_cost_$": np.round(Total_cost, 2)
    })

df = pd.DataFrame(records)
st.dataframe(df.head(10))

################################################################################
# Section 2: Anomaly Detection & Forecasting (using PyTorch)
################################################################################

st.header("Anomaly Detection on Total Consumption")
window_size = 3  # window size in months
energy_values = df["Total_kWh"].values
X_windows = []
center_indices = []
for i in range(len(energy_values) - window_size + 1):
    window = energy_values[i : i + window_size]
    X_windows.append(window)
    center_indices.append(i + window_size // 2)
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

st.write("Detected anomaly indices (by month index):", anomaly_months)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.index, df["Total_kWh"], label="Total_kWh")
for i, idx in enumerate(anomaly_months):
    ax.scatter(df.index[idx], df["Total_kWh"].iloc[idx], color='red', label="Anomaly" if i == 0 else "")
ax.set_xlabel("Record Index (Month)")
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

future_records = []
for j in range(3):
    future_records.append(f"Month {n_samples+j+1}")
future_df = pd.DataFrame({
    "Month": future_records,
    "Predicted_Total_kWh": future_predictions.flatten()
})
st.write("Forecast for the next 3 months (Total_kWh):")
st.line_chart(future_df.set_index("Month"))

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
k = 2  # reduce to 2 dimensions
principal_components = V[:k, :].T
X_projected = X_centered @ principal_components

fig2, ax2 = plt.subplots()
ax2.scatter(X_projected[:, 0].numpy(), X_projected[:, 1].numpy(), c='blue', alpha=0.7)
ax2.set_xlabel("PC 1")
ax2.set_ylabel("PC 2")
ax2.set_title("PCA Projection")
st.pyplot(fig2)

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
fig3, ax3 = plt.subplots(figsize=(8, 6))
cax = ax3.matshow(corr_np, cmap='coolwarm')
fig3.colorbar(cax)
ax3.set_xticks(range(len(features)))
ax3.set_yticks(range(len(features)))
ax3.set_xticklabels(features, rotation=90, fontsize=8)
ax3.set_yticklabels(features, fontsize=8)
ax3.set_title("Pearson Correlation Matrix", pad=20)
st.pyplot(fig3)

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
    # client = genai.Client(api_key="AIzaSyB4VHerf7dMxw35V6TS8tru3E_i7_wkLVU")
    prompt = (
        f"I'm a sustainability expert. A user has a monthly carbon footprint of {total_carbon:.2f} kg CO₂. "
        f"They consume {electricity} kWh of electricity, {gas} therms of natural gas, and drive {vehicle} miles per month. "
        f"Provide actionable, friendly recommendations to lower their carbon emissions."
    )
    # Configure the API key
    client = genai.Client(api_key="AIzaSyB4VHerf7dMxw35V6TS8tru3E_i7_wkLVU")
    # Generate content using the specified model and prompt
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt])
    st.write(response.text)


    
    
    
st.write("""
This app integrates energy consumption data analysis, anomaly detection, forecasting, PCA visualization, 
and a carbon footprint calculator with personalized sustainability recommendations powered by the Gemini API.
""")




