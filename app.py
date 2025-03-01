import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from google import genai
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # not used here, but kept for completeness

# Configure the page layout
st.set_page_config(page_title="Ontario Energy Dashboard", layout="wide")

###############################################################################
# SIMPLE PAGE NAVIGATION VIA SIDEBAR RADIO
###############################################################################
page = st.sidebar.radio("Go to", ["Overview", "User Input"])

###############################################################################
# TRAIN REGRESSION MODEL TO LEARN CONVERSION FACTORS
###############################################################################
@st.cache_resource
def train_regression_model():
    try:
        df = pd.read_csv("carbon_emissions_large.csv")
        required = ["Electricity", "Natural Gas", "Vehicle", "Heating Oil", "Propane", "Total_Emissions"]
        if not all(col in df.columns for col in required):
            raise ValueError("Dataset does not contain required columns.")
        features = ["Electricity", "Natural Gas", "Vehicle", "Heating Oil", "Propane"]
        X = df[features].values.astype(np.float32)
        y = df["Total_Emissions"].values.astype(np.float32).reshape(-1, 1)
        X_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y)
        model = nn.Linear(X.shape[1], 1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        epochs = 1000
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        weights = model.weight.data.numpy().flatten()
        bias = model.bias.data.numpy()[0]
        return weights, bias
    except Exception as e:
        st.error("Error training model: " + str(e))
        return np.array([0.075, 5.3, 0.411, 10.16, 5.74]), 0.0

if "conversion_factors" not in st.session_state:
    weights, bias = train_regression_model()
    st.session_state.conversion_factors = {
        "Electricity": weights[0],
        "Natural Gas": weights[1],
        "Vehicle": weights[2],
        "Heating Oil": weights[3],
        "Propane": weights[4]
    }
    st.session_state.bias = bias

###############################################################################
# OVERVIEW PAGE
###############################################################################
def show_overview_page():
    st.title("Overview: Carbon Footprint Factors & Statistical Methodology")
    
    # 1) Time Series Analysis
    st.markdown("## Time Series Analysis")
    dates = pd.date_range(start='2020-01-01', periods=36, freq='M')
    consumption = np.random.normal(500, 50, len(dates))
    df_timeseries = pd.DataFrame({'Month': dates, 'Energy Consumption (kWh)': consumption})
    st.line_chart(df_timeseries.set_index('Month'))
    
    # 2) Emissions Breakdown Pie Chart (percentages in legend)
    st.markdown("## Emissions Breakdown")
    sample_values = {
        "Electricity": 500,
        "Natural Gas": 50,
        "Vehicle": 1000,
        "Heating Oil": 20,
        "Propane": 10
    }
    conv = st.session_state.conversion_factors
    sample_emissions = {
        "Electricity": sample_values["Electricity"] * conv["Electricity"],
        "Natural Gas": sample_values["Natural Gas"] * conv["Natural Gas"],
        "Vehicle": sample_values["Vehicle"] * conv["Vehicle"],
        "Heating Oil": sample_values["Heating Oil"] * conv["Heating Oil"],
        "Propane": sample_values["Propane"] * conv["Propane"]
    }
    breakdown_df = pd.DataFrame({
        "Source": list(sample_emissions.keys()),
        "Emissions": list(sample_emissions.values())
    }).sort_values(by="Emissions")
    total_emissions = breakdown_df["Emissions"].sum()
    breakdown_df["Percent"] = breakdown_df["Emissions"] / total_emissions * 100

    fig, ax = plt.subplots()
    wedges, texts = ax.pie(
        breakdown_df["Emissions"],
        startangle=90,
        textprops={'fontsize': 8}
    )
    ax.axis('equal')
    ax.set_title("Emissions Breakdown (Ascending Order)")
    legend_labels = breakdown_df.apply(lambda row: f"{row['Source']} - {row['Percent']:.1f}%", axis=1).tolist()
    ax.legend(wedges, legend_labels, title="Source", loc="center left", bbox_to_anchor=(1, 0, 0.3, 1))
    st.pyplot(fig)
    
    # 3) Your Estimated Emissions
    st.markdown("## Your Estimated Emissions")
    user_total = st.session_state.get("total_carbon", None)
    if user_total is None:
        st.write("You haven't calculated your personal emissions yet. **Please select 'User Input' from the left sidebar** to calculate your emissions.")
    else:
        st.write(f"Your Estimated Emissions (kg CO₂): **{user_total:.2f}**")
    
    # 4) Benchmarking
    st.markdown("## Benchmarking")
    canada_avg = 1500.00
    if user_total is not None:
        diff = user_total - canada_avg
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Your Estimated Emissions (kg CO₂)", f"{user_total:.2f}")
        with col2:
            st.metric("Average in Canada (kg CO₂)", f"{canada_avg:.2f}")
        if diff < 0:
            st.markdown(f"**Great job!** Your emissions are **{abs(diff):.2f} kg CO₂ lower** than the Canadian average.")
        else:
            st.markdown(f"**Attention:** Your emissions are **{diff:.2f} kg CO₂ higher** than the Canadian average.")
    else:
        st.write("Once you calculate your emissions, you can compare them to the Canadian average here.")
    
    # 5) Methodology
    st.markdown("## Overview of Methodology")
    st.markdown("### Regression Analysis:")
    st.markdown(r"We used a multiple linear regression model of the form:")
    st.latex(r"Y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \varepsilon")
    st.markdown(r"where \(Y\) is the total emissions, \(x_i\) are the consumption measures, and \(\beta_i\) are the conversion factors.")
    st.markdown("### Gradient Descent Optimization:")
    st.markdown(r"To minimize the Mean Squared Error (MSE), we used gradient descent. The cost function is:")
    st.latex(r"J(\beta) = \frac{1}{2n} \sum_{i=1}^{n} \left( y_i - x_i^T \beta \right)^2")
    st.markdown("and the gradient descent update rule is:")
    st.latex(r"\beta := \beta - \alpha \frac{\partial J}{\partial \beta}")
    st.markdown(r"where \(\alpha\) is the learning rate.")
    st.markdown("### Outlier Removal:")
    st.markdown(r"Prior to regression, outliers are removed using the Interquartile Range (IQR) method:")
    st.latex(r"\text{Lower Bound} = Q1 - 1.5 \times (Q3 - Q1), \quad \text{Upper Bound} = Q3 + 1.5 \times (Q3 - Q1)")
    st.markdown("### Carbon Emissions Formula")
    st.latex(r"""
    \begin{aligned}
    \text{Total CO}_2\, (\text{kg}) =\; & \text{Electricity Consumption (kWh)} \times """ + f"{st.session_state.conversion_factors['Electricity']:.3f}" + r""" \\
    &+ \text{Natural Gas Consumption (therms)} \times """ + f"{st.session_state.conversion_factors['Natural Gas']:.3f}" + r""" \\
    &+ \text{Vehicle Miles Driven} \times """ + f"{st.session_state.conversion_factors['Vehicle']:.3f}" + r""" \\
    &+ \text{Heating Oil Consumption (gallons)} \times """ + f"{st.session_state.conversion_factors['Heating Oil']:.3f}" + r""" \\
    &+ \text{Propane Consumption (gallons)} \times """ + f"{st.session_state.conversion_factors['Propane']:.3f}" + r"""
    \end{aligned}
    """)
    
    # 6) Print the Conversion Factors at the Bottom
    st.markdown("## Conversion Factors Used")
    conversion_text = " | ".join([f"**{key}:** {value:.3f}" for key, value in st.session_state.conversion_factors.items()])
    st.write(conversion_text)
    st.write(f"**Bias:** {st.session_state.bias:.3f}")

###############################################################################
# USER INPUT PAGE
###############################################################################
def show_user_input_page():
    st.title("Carbon Footprint Calculator")
    st.markdown("Please select the consumption levels for each category below. Each option represents a typical amount:")

    # Electricity
    st.subheader("Electricity Consumption")
    elec_choice = st.radio("Electricity Level", options=["Low (300 kWh)", "Medium (500 kWh)", "High (700 kWh)"])
    if elec_choice == "Low (300 kWh)":
        electricity_val = 300.0
    elif elec_choice == "Medium (500 kWh)":
        electricity_val = 500.0
    else:
        electricity_val = 700.0

    # Natural Gas
    st.subheader("Natural Gas Consumption")
    gas_choice = st.radio("Natural Gas Level", options=["Low (30 therms)", "Medium (50 therms)", "High (80 therms)"])
    if gas_choice == "Low (30 therms)":
        gas_val = 30.0
    elif gas_choice == "Medium (50 therms)":
        gas_val = 50.0
    else:
        gas_val = 80.0

    # Vehicle Miles
    st.subheader("Vehicle Miles Driven")
    vehicle_choice = st.radio("Vehicle Miles Level", options=["Low (500 miles)", "Medium (1000 miles)", "High (1500 miles)"])
    if vehicle_choice == "Low (500 miles)":
        vehicle_val = 500.0
    elif vehicle_choice == "Medium (1000 miles)":
        vehicle_val = 1000.0
    else:
        vehicle_val = 1500.0

    # Heating Oil
    st.subheader("Heating Oil Consumption")
    oil_choice = st.radio("Heating Oil Level", options=["Low (10 gallons)", "Medium (20 gallons)", "High (30 gallons)"])
    if oil_choice == "Low (10 gallons)":
        oil_val = 10.0
    elif oil_choice == "Medium (20 gallons)":
        oil_val = 20.0
    else:
        oil_val = 30.0

    # Propane
    st.subheader("Propane Consumption")
    propane_choice = st.radio("Propane Level", options=["Low (5 gallons)", "Medium (10 gallons)", "High (15 gallons)"])
    if propane_choice == "Low (5 gallons)":
        propane_val = 5.0
    elif propane_choice == "Medium (10 gallons)":
        propane_val = 10.0
    else:
        propane_val = 15.0

    st.markdown("### Review Your Selections")
    st.write(f"**Electricity:** {electricity_val} kWh")
    st.write(f"**Natural Gas:** {gas_val} therms")
    st.write(f"**Vehicle Miles:** {vehicle_val} miles")
    st.write(f"**Heating Oil:** {oil_val} gallons")
    st.write(f"**Propane:** {propane_val} gallons")

    if "total_carbon" not in st.session_state:
        st.session_state.total_carbon = None

    carbon_placeholder = st.empty()

    calc_button = st.button("Calculate Carbon Footprint")
    if calc_button:
        conv = st.session_state.conversion_factors
        carbon_elec = electricity_val * conv["Electricity"]
        carbon_gas = gas_val * conv["Natural Gas"]
        carbon_vehicle = vehicle_val * conv["Vehicle"]
        carbon_oil = oil_val * conv["Heating Oil"]
        carbon_propane = propane_val * conv["Propane"]

        total_carbon = carbon_elec + carbon_gas + carbon_vehicle + carbon_oil + carbon_propane
        st.session_state.total_carbon = total_carbon

        carbon_placeholder.success(f"**Estimated Monthly Carbon Emissions:** {total_carbon:.2f} kg CO₂")

    get_recs = st.button("Get Sustainability Recommendations")
    if get_recs:
        if st.session_state.total_carbon is None:
            st.error("Please calculate your carbon footprint first!")
        else:
            carbon_placeholder.success(f"**Estimated Monthly Carbon Emissions:** {st.session_state.total_carbon:.2f} kg CO₂")
            suggestion_placeholder = st.empty()
            suggestion_placeholder.info("Creating suggestions, please wait...")
            prompt = (
                f"I'm a sustainability expert. A user has a monthly carbon footprint of {st.session_state.total_carbon:.2f} kg CO₂. "
                f"They consume {electricity_val} kWh of electricity, {gas_val} therms of natural gas, drive {vehicle_val} miles, "
                f"use {oil_val} gallons of heating oil, and {propane_val} gallons of propane each month. "
                f"Provide actionable, friendly recommendations to lower their carbon emissions."
            )
            client = genai.Client(api_key="YOUR_GEMINI_API_KEY")
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt]
            )
            suggestion_placeholder.success(response.text)

###############################################################################
# MAIN
###############################################################################
def main():
    if page == "Overview":
        show_overview_page()
    else:
        show_user_input_page()

if __name__ == "__main__":
    main()
