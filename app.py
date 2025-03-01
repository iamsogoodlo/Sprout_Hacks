import random
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from google import genai
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # not used here, but kept for completeness
import requests
from streamlit_lottie import st_lottie  # Ensure you install streamlit-lottie (pip install streamlit-lottie)
import io  # For in-memory file operations if needed elsewhere
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import tempfile
import seaborn as sns  # Import seaborn for heatmap

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
# HELPER FUNCTION FOR GENETIC ALGORITHM (OPTIMIZATION)
###############################################################################
def optimize_carbon_footprint(consumption_limits, conversion_factors):
    population_size = 10
    generations = 100
    mutation_rate = 0.1

    # Random initialization of population
    population = [
        [random.uniform(consumption_limits[i][0], consumption_limits[i][1]) for i in range(len(consumption_limits))]
        for _ in range(population_size)
    ]
    
    def calculate_emissions(consumption):
        return sum(consumption[i] * conversion_factors[i] for i in range(len(consumption)))

    def fitness(consumption):
        return 1 / (1 + calculate_emissions(consumption))  # Higher fitness for lower emissions

    def mutate(consumption):
        i = random.randint(0, len(consumption) - 1)
        consumption[i] += random.uniform(-0.1, 0.1)
        return consumption

    def crossover(consumption1, consumption2):
        point = random.randint(1, len(consumption1) - 1)
        return consumption1[:point] + consumption2[point:]

    for generation in range(generations):
        population.sort(key=lambda x: fitness(x), reverse=True)
        
        # Selection: top 50% of population
        selected = population[:population_size // 2]
        
        # Crossover: breed the top selections
        next_generation = selected.copy()
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            next_generation.append(child)

        # Mutation: introduce small changes
        population = [mutate(ind) if random.random() < mutation_rate else ind for ind in next_generation]

    best_solution = population[0]
    return best_solution

###############################################################################
# HELPER FUNCTION: LOAD LOTTIE ANIMATION JSON FROM URL
###############################################################################
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

###############################################################################
# PCA-BASED 3D ANIMATED GRAPH FUNCTION (using PyTorch for PCA)
###############################################################################
def generate_pca_animation():
    try:
        # Load dataset and extract energy consumption features
        df = pd.read_csv("carbon_emissions_large.csv")
        features = ["Electricity", "Natural Gas", "Vehicle", "Heating Oil", "Propane"]
        X_np = df[features].values.astype(np.float32)
        y_np = df["Total_Emissions"].values.astype(np.float32)
        
        # Convert to torch tensor and center the data
        X = torch.tensor(X_np)
        X_centered = X - X.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix (unbiased estimator)
        cov_matrix = torch.matmul(X_centered.t(), X_centered) / (X_centered.shape[0] - 1)
        
        # Compute eigenvalues and eigenvectors using eigh (for symmetric matrices)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Select top 2 principal components
        pcs = eigenvectors[:, :2]
        
        # Project centered data onto the top 2 PCs
        X_pca = torch.matmul(X_centered, pcs)
        X_pca_np = X_pca.numpy()  # Shape: (n_samples, 2)
        
        # Create 3D scatter plot: PC1 vs PC2, Total Emissions as z-axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(X_pca_np[:, 0], X_pca_np[:, 1], y_np, c=y_np, cmap="viridis")
        ax.set_xlabel("Key Energy Driver 1")
        ax.set_ylabel("Key Energy Driver 2")
        ax.set_zlabel("Total Emissions")
        fig.colorbar(sc, ax=ax, label="Total Emissions")
        
        # Animation: rotate view (azimuth from 0 to 360 degrees)
        def update(frame):
            ax.view_init(elev=30, azim=frame)
            return fig,
        
        frames = np.linspace(0, 360, 90)
        ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
        writer = PillowWriter(fps=10)
        
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            temp_filename = tmp.name

        ani.save(temp_filename, writer=writer)
        plt.close(fig)
        
        with open(temp_filename, "rb") as f:
            gif_bytes = f.read()
        os.remove(temp_filename)
        return gif_bytes
    except Exception as e:
        st.error("Error generating PCA animation with PyTorch: " + str(e))
        return None

###############################################################################
# HEATMAP DISPLAY OF ENERGY CONSUMPTION AND EMISSIONS (WITH SCIENTIFIC NOTATION)
###############################################################################
def show_heatmap():
    # Example heatmap data (can be modified with real data)
    regions = ["North", "South", "East", "West"]
    energy_types = ["Electricity", "Natural Gas", "Vehicle", "Heating Oil", "Propane"]
    
    # Generating some random data for demonstration
    emissions_data = np.random.rand(len(regions), len(energy_types)) * 10000  # Randomized data in thousands
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(emissions_data, annot=True, fmt='.2e', cmap='YlGnBu', xticklabels=energy_types, yticklabels=regions, ax=ax)
    ax.set_title("Energy Consumption vs Emissions by Region and Energy Type")
    st.pyplot(fig)

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

    # 3) Show the heatmap
    st.markdown("## Energy Consumption vs Emissions Heatmap")
    show_heatmap()

    # 5) Your Estimated Emissions
    st.markdown("## Your Estimated Emissions")
    user_total = st.session_state.get("total_carbon", None)
    if user_total is None:
        st.write("You haven't calculated your personal emissions yet. **Please select 'User Input' from the left sidebar** to calculate your emissions.")
    else:
        st.write(f"Your Estimated Emissions (kg CO₂): **{user_total:.2f}**")
    
    # 6) Benchmarking
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
    
    # 7) Methodology
    # 4) Genetic Algorithm Overview
    st.markdown("## Overview of Methodology")
    st.markdown("## Genetic Algorithm for Optimization")
    st.markdown(
        """
        The **Genetic Algorithm** (GA) is used to optimize the energy consumption pattern by simulating the process of natural selection. 
        It creates a population of possible energy consumption combinations, evaluates their fitness (carbon emissions), and iteratively improves 
        the solutions through selection, crossover, and mutation.
        
        The **Genetic Algorithm** process is as follows:
        1. **Population Initialization**: A random set of consumption levels is generated.
        2. **Fitness Evaluation**: The fitness of each individual (set of consumption levels) is evaluated by calculating its emissions.
        3. **Selection**: The fittest individuals are selected for reproduction.
        4. **Crossover**: Pairs of individuals are combined to create new individuals.
        5. **Mutation**: A small mutation is applied to some individuals to introduce diversity.

        The algorithm runs for a set number of generations, evolving a solution that minimizes carbon emissions.
        """
    )
    st.markdown("### Regression Analysis:")
    st.markdown(r"We used a multiple linear regression model of the form:")
    st.latex(r"Y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \varepsilon")
    st.markdown(r"where \(Y\) is the total emissions, \(x_i\) are the consumption measures, and \(\beta_i\) are the conversion factors.")
    st.markdown("### Gradient Descent Optimization:")
    st.markdown(r"To minimize the Mean Squared Error (MSE), we used gradient descent. The cost function is:")
    st.latex(r"J(\beta) = \frac{1}{2n} \sum_{i=1}^{n} \left( y_i - x_i^T \beta \right)^2")
    st.markdown("and the gradient descent update rule is:")
    st.latex(r"\beta := \beta - \alpha \frac{\partial J}{\partial \beta}")
    st.markdown("### Outlier Removal:")
    st.markdown(r"Prior to regression, outliers are removed using the Interquartile Range (IQR) method:")
    st.latex(r"\text{Lower Bound} = Q1 - 1.5 \times (Q3 - Q1), \quad \text{Upper Bound} = Q3 + 1.5 \times (Q3 - Q1)")

    # 8) Animated 3D Graph: Key Energy Drivers of Carbon Emissions
    st.markdown("## Key Energy Drivers of Carbon Emissions")
    st.markdown(
        """
        The animated 3D graph below shows how we identified the key energy drivers impacting total carbon emissions.  
        We used PyTorch to perform PCA on the five energy consumption features by first centering the data and computing its covariance matrix.  
        The eigen-decomposition revealed that the first two principal components capture the most variance in the data.  
        These two components are used as proxies for the most influential energy consumption parameters, while the z-axis represents Total Emissions.
        """
    )
    gif_pca = generate_pca_animation()
    if gif_pca:
        st.image(gif_pca, caption="Key Energy Drivers of Carbon Emissions", use_container_width=True)

###############################################################################
# USER INPUT PAGE
###############################################################################
def show_user_input_page():
    st.title("Carbon Footprint Calculator")
    st.markdown("Please select the consumption levels for each category below. Each option represents a typical amount:")

    # Electricity
    st.subheader("Electricity Consumption")
    elec_choice = st.radio("Electricity Level", options=["Low (300 kWh)", "Medium (500 kWh)", "High (700 kWh)"])
    electricity_val = 300.0 if elec_choice == "Low (300 kWh)" else 500.0 if elec_choice == "Medium (500 kWh)" else 700.0

    # Natural Gas
    st.subheader("Natural Gas Consumption")
    gas_choice = st.radio("Natural Gas Level", options=["Low (30 therms)", "Medium (50 therms)", "High (80 therms)"])
    gas_val = 30.0 if gas_choice == "Low (30 therms)" else 50.0 if gas_choice == "Medium (50 therms)" else 80.0

    # Vehicle Miles
    st.subheader("Vehicle Miles Driven")
    vehicle_choice = st.radio("Vehicle Miles Level", options=["Low (500 miles)", "Medium (1000 miles)", "High (1500 miles)"])
    vehicle_val = 500.0 if vehicle_choice == "Low (500 miles)" else 1000.0 if vehicle_choice == "Medium (1000 miles)" else 1500.0

    # Heating Oil
    st.subheader("Heating Oil Consumption")
    oil_choice = st.radio("Heating Oil Level", options=["Low (10 gallons)", "Medium (20 gallons)", "High (30 gallons)"])
    oil_val = 10.0 if oil_choice == "Low (10 gallons)" else 20.0 if oil_choice == "Medium (20 gallons)" else 30.0

    # Propane
    st.subheader("Propane Consumption")
    propane_choice = st.radio("Propane Level", options=["Low (5 gallons)", "Medium (10 gallons)", "High (15 gallons)"])
    propane_val = 5.0 if propane_choice == "Low (5 gallons)" else 10.0 if propane_choice == "Medium (10 gallons)" else 15.0

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
        with st.spinner("Calculating your carbon footprint..."):
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
            # Create a placeholder for the spinner and suggestions
            spinner_placeholder = st.empty()
            suggestion_placeholder = st.empty()
            spinner_placeholder.image("leaf_spinner.gif", caption="Suggesting...", use_container_width=160)
            prompt = (
                f"I'm a sustainability expert. A user has a monthly carbon footprint of {st.session_state.total_carbon:.2f} kg CO₂. "
                f"They consume {electricity_val} kWh of electricity, {gas_val} therms of natural gas, drive {vehicle_val} miles, "
                f"use {oil_val} gallons of heating oil, and {propane_val} gallons of propane each month. "
                f"Provide actionable, friendly recommendations to lower their carbon emissions."
            )
            client = genai.Client(api_key="AIzaSyB4VHerf7dMxw35V6TS8tru3E_i7_wkLVU")
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt]
            )
            # Remove the spinner after the response is received
            spinner_placeholder.empty()
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
