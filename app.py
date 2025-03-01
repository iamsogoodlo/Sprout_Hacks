import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from google import genai
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# Set up page configuration and sidebar navigation
st.set_page_config(page_title="Ontario Energy Dashboard", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "User Input"])

###############################################################################
# Dashboard Page: Data Analysis and Key Graphs
###############################################################################
if page == "Dashboard":
    st.title("Ontario Household Energy Dashboard")
    
    # --- Dataset Loading ---
    st.header("Dataset Overview")
    try:
        df = pd.read_csv("factors.csv")
        st.write("Loaded factors.csv successfully.")
        st.dataframe(df.head(10))
    except Exception as e:
        st.error("Failed to load factors.csv: " + str(e))
        st.stop()

    # --- Outlier Removal ---
    st.header("Outlier Removal")
    Q1 = df["Total_kWh"].quantile(0.25)
    Q3 = df["Total_kWh"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df["Total_kWh"] >= lower_bound) & (df["Total_kWh"] <= upper_bound)]
    st.write(f"Removed {len(df) - len(df_filtered)} outlier record(s) based on Total_kWh.")
    df = df_filtered.reset_index(drop=True)
    st.dataframe(df.head(10))

    # --- Anomaly Detection ---
    st.header("Anomaly Detection on Total Consumption")
    window_size = 3  # window size in records
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
        
        if len(X_windows) == 0:
            st.error("No valid windows found for anomaly detection.")
        else:
            X_windows = np.array(X_windows, dtype=np.float32)
            X_tensor = torch.tensor(X_windows)

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
            anomaly_indices = [center_indices[i] for i in anomaly_windows]

            st.write("Detected anomaly indices (by record index):", anomaly_indices)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df.index, df["Total_kWh"], label="Total_kWh")
            for i, idx in enumerate(anomaly_indices):
                ax.scatter(df.index[idx], df["Total_kWh"].iloc[idx], color='red', label="Anomaly" if i == 0 else "")
            ax.set_xlabel("Record Index")
            ax.set_ylabel("Total_kWh")
            ax.legend()
            st.pyplot(fig)

    # --- PCA Projection (3D) ---
    st.header("PCA Projection")
    pca_features = ["AvgTemp_C", "AvgHumidity_%", "Occupants", "HouseSize_sqft",
                    "OffPeak_kWh", "MidPeak_kWh", "OnPeak_kWh", "Total_kWh", "Total_cost_$"]
    X_features = df[pca_features].values.astype(np.float32)
    X_tensor_feat = torch.tensor(X_features)
    X_centered = X_tensor_feat - X_tensor_feat.mean(dim=0, keepdim=True)
    U, S, V = torch.linalg.svd(X_centered, full_matrices=False)
    k = 3
    if V.shape[0] < k:
        k = V.shape[0]
    principal_components = V[:k, :].T
    X_projected = X_centered @ principal_components
    if X_projected.shape[1] < 3:
        missing = 3 - X_projected.shape[1]
        pad = torch.zeros(X_projected.shape[0], missing)
        X_projected = torch.cat([X_projected, pad], dim=1)

    fig3d = plt.figure(figsize=(8, 6))
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.scatter(X_projected[:, 0].numpy(), X_projected[:, 1].numpy(), X_projected[:, 2].numpy(), 
                 c='blue', alpha=0.7)
    ax3d.set_xlabel("PC 1")
    ax3d.set_ylabel("PC 2")
    ax3d.set_zlabel("PC 3")
    ax3d.set_title("PCA Projection (3D)")
    st.pyplot(fig3d)

    st.write("""
    **Note:** This dashboard displays only the most relevant graphs from the dataset analysis.
    """)

###############################################################################
# User Input Page: Detailed Carbon Footprint Calculation
###############################################################################
elif page == "User Input":
    st.title("Carbon Footprint Calculator")
    st.markdown("Please select the consumption levels for each category below. Each option represents a typical amount:")

    # -- Electricity --
    st.subheader("Electricity Consumption")
    elec_choice = st.radio("Electricity Level", 
                           options=["Low (300 kWh)", "Medium (500 kWh)", "High (700 kWh)"])
    if "300" in elec_choice:
        electricity_val = 300.0
    elif "500" in elec_choice:
        electricity_val = 500.0
    else:
        electricity_val = 700.0

    # -- Natural Gas --
    st.subheader("Natural Gas Consumption")
    gas_choice = st.radio("Natural Gas Level", 
                          options=["Low (30 therms)", "Medium (50 therms)", "High (80 therms)"])
    if "30" in gas_choice:
        gas_val = 30.0
    elif "50" in gas_choice:
        gas_val = 50.0
    else:
        gas_val = 80.0

    # -- Vehicle Miles --
    st.subheader("Vehicle Miles Driven")
    vehicle_choice = st.radio("Vehicle Miles Level", 
                              options=["Low (500 miles)", "Medium (1000 miles)", "High (1500 miles)"])
    if "500" in vehicle_choice:
        vehicle_val = 500.0
    elif "1000" in vehicle_choice:
        vehicle_val = 1000.0
    else:
        vehicle_val = 1500.0

    # -- Heating Oil --
    st.subheader("Heating Oil Consumption")
    oil_choice = st.radio("Heating Oil Level", 
                          options=["Low (10 gallons)", "Medium (20 gallons)", "High (30 gallons)"])
    if "10" in oil_choice:
        oil_val = 10.0
    elif "20" in oil_choice:
        oil_val = 20.0
    else:
        oil_val = 30.0

    # -- Propane --
    st.subheader("Propane Consumption")
    propane_choice = st.radio("Propane Level", 
                              options=["Low (5 gallons)", "Medium (10 gallons)", "High (15 gallons)"])
    if "5" in propane_choice:
        propane_val = 5.0
    elif "10" in propane_choice:
        propane_val = 10.0
    else:
        propane_val = 15.0

    # -- Air Travel --
    st.subheader("Air Travel Distance")
    air_choice = st.radio("Air Travel Level", 
                          options=["Low (100 miles)", "Medium (300 miles)", "High (600 miles)"])
    if "100" in air_choice:
        air_val = 100.0
    elif "300" in air_choice:
        air_val = 300.0
    else:
        air_val = 600.0

    st.markdown("### Review Your Selections")
    st.write(f"**Electricity:** {electricity_val} kWh")
    st.write(f"**Natural Gas:** {gas_val} therms")
    st.write(f"**Vehicle Miles:** {vehicle_val} miles")
    st.write(f"**Heating Oil:** {oil_val} gallons")
    st.write(f"**Propane:** {propane_val} gallons")
    st.write(f"**Air Travel:** {air_val} miles")

    # Track the carbon footprint in session state so it can be accessed after calculation
    if 'total_carbon' not in st.session_state:
        st.session_state.total_carbon = None

    # Placeholders for messages
    carbon_placeholder = st.empty()       # for the carbon footprint result
    suggestion_placeholder = st.empty()   # for the "Creating suggestions..." / final suggestions

    # Button to calculate carbon footprint
    calc_button = st.button("Calculate Carbon Footprint")
    if calc_button:
        # Conversion factors
        elec_factor = 0.075
        gas_factor = 5.3
        vehicle_factor = 0.411
        oil_factor = 10.16
        propane_factor = 5.74
        air_factor = 0.2

        carbon_elec = electricity_val * elec_factor
        carbon_gas = gas_val * gas_factor
        carbon_vehicle = vehicle_val * vehicle_factor
        carbon_oil = oil_val * oil_factor
        carbon_propane = propane_val * propane_factor
        carbon_air = air_val * air_factor

        total_carbon = (carbon_elec + carbon_gas + carbon_vehicle +
                        carbon_oil + carbon_propane + carbon_air)

        # Store in session_state so it can be used after user clicks next button
        st.session_state.total_carbon = total_carbon
        carbon_placeholder.success(
            f"**Estimated Monthly Carbon Emissions:** {total_carbon:.2f} kg CO₂"
        )

    # Button to get sustainability recommendations
    suggest_button = st.button("Get Sustainability Recommendations")
    if suggest_button:
        if st.session_state.total_carbon is None:
            st.error("Please calculate your carbon footprint first!")
        else:
            # Keep the carbon footprint on screen
            carbon_placeholder.success(
                f"**Estimated Monthly Carbon Emissions:** {st.session_state.total_carbon:.2f} kg CO₂"
            )
            # Show a placeholder while we call the API
            suggestion_placeholder.info("Creating suggestions, please wait...")

            prompt = (
                f"I'm a sustainability expert. A user has a monthly carbon footprint of {st.session_state.total_carbon:.2f} kg CO₂. "
                f"They consume {electricity_val} kWh of electricity, {gas_val} therms of natural gas, drive {vehicle_val} miles, "
                f"use {oil_val} gallons of heating oil, {propane_val} gallons of propane, and travel {air_val} miles by air each month. "
                f"Provide actionable, friendly recommendations to lower their carbon emissions."
            )

            # Make the Gemini API call (replace with your actual API key)
            client = genai.Client(api_key="AIzaSyB4VHerf7dMxw35V6TS8tru3E_i7_wkLVU")
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt]
            )

            # Display final suggestions
            suggestion_placeholder.success(response.text)
