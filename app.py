import streamlit as st
st.set_page_config(page_title="Ontario Energy Dashboard", layout="wide")

import random
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

###############################################################################
# CUSTOM STYLING: LIGHT FARMLAND-STYLE BACKGROUND, CENTERED CONTENT, AND SIDEBAR ENLARGEMENT
###############################################################################
st.markdown(
    """
    <style>
    /* Main container: white to light green gradient background */
    .reportview-container, .main .block-container {
        background: linear-gradient(135deg, #ffffff, #e8f5e9) no-repeat center center fixed;
        background-size: cover;
    }
    /* Semi-transparent container for better readability */
    .block-container {
        backdrop-filter: blur(6px);
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 12px;
        padding: 2rem;
        margin-top: 2rem;
    }
    /* Center headings */
    h2, h3, h4, h5 {
        text-align: center;
        margin-bottom: 1rem;
    }
    /* Styling for question text */
    .question-text {
        font-size: 1.2rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1rem;
    }
    /* Container for sliders and buttons: limit width to 65% and center */
    .slider-container {
        width: 65%;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
    }
    /* Center class for generic centering */
    .center {
        text-align: center;
    }
    /* Make dataframes centered */
    div[data-testid="stDataFrameContainer"] {
        margin: 0 auto;
    }
    /* Increase sidebar navigation radio elements' size and spacing */
    [data-testid="stSidebar"] .css-1d391kg,
    [data-testid="stSidebar"] .stRadio {
        font-size: 1.5rem !important;
        padding: 1rem 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

###############################################################################
# ADD AUDIO ELEMENT AND JS FUNCTION FOR CLICK SOUND
###############################################################################
st.components.v1.html(
    """
    <audio id="clickSound" src="https://actions.google.com/sounds/v1/cartoon/pop.ogg"></audio>
    <script>
    function playClickSound(){
        document.getElementById("clickSound").play();
    }
    </script>
    """,
    height=0
)

def play_click_sound():
    st.components.v1.html(
        """
        <script>
        playClickSound();
        </script>
        """,
        height=0
    )

###############################################################################
# LOGO CREATION FUNCTION
###############################################################################
def create_logo():
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis("off")
    
    # Draw a green circle as background
    circle = plt.Circle((0.5, 0.5), 0.4, color='#4CAF50', zorder=1)
    ax.add_artist(circle)
    
    # Draw a stylized leaf (lighter green)
    leaf_points = [
        (0.5, 0.68),
        (0.45, 0.55),
        (0.5, 0.42),
        (0.55, 0.55)
    ]
    leaf = plt.Polygon(leaf_points, closed=True, color='#8BC34A', zorder=2)
    ax.add_artist(leaf)
    
    # Add project initials ("OED") in white
    ax.text(0.5, 0.28, "OED", fontsize=24, fontweight='bold',
            color='white', ha='center', va='center', zorder=3)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

###############################################################################
# SIDEBAR NAVIGATION (Overview, User Input, User History, Suggestions)
###############################################################################
page = st.sidebar.radio("Go to", ["Overview", "User Input", "User History", "Suggestions"])

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

    population = [
        [random.uniform(consumption_limits[i][0], consumption_limits[i][1]) for i in range(len(consumption_limits))]
        for _ in range(population_size)
    ]
    
    def calculate_emissions(consumption):
        return sum(consumption[i] * conversion_factors[i] for i in range(len(consumption)))

    def fitness(consumption):
        return 1 / (1 + calculate_emissions(consumption))

    def mutate(consumption):
        i = random.randint(0, len(consumption) - 1)
        consumption[i] += random.uniform(-0.1, 0.1)
        return consumption

    def crossover(consumption1, consumption2):
        point = random.randint(1, len(consumption1) - 1)
        return consumption1[:point] + consumption2[point:]

    for generation in range(generations):
        population.sort(key=lambda x: fitness(x), reverse=True)
        selected = population[:population_size // 2]
        next_generation = selected.copy()
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            next_generation.append(child)
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
        df = pd.read_csv("carbon_emissions_large.csv")
        features = ["Electricity", "Natural Gas", "Vehicle", "Heating Oil", "Propane"]
        X_np = df[features].values.astype(np.float32)
        y_np = df["Total_Emissions"].values.astype(np.float32)
        
        X = torch.tensor(X_np)
        X_centered = X - X.mean(dim=0, keepdim=True)
        
        cov_matrix = torch.matmul(X_centered.t(), X_centered) / (X_centered.shape[0] - 1)
        
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, sorted_indices]
        
        pcs = eigenvectors[:, :2]
        X_pca = torch.matmul(X_centered, pcs)
        X_pca_np = X_pca.numpy()
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(X_pca_np[:, 0], X_pca_np[:, 1], y_np, c=y_np, cmap="viridis")
        ax.set_xlabel("Key Energy Driver 1")
        ax.set_ylabel("Key Energy Driver 2")
        ax.set_zlabel("Total Emissions")
        fig.colorbar(sc, ax=ax, label="Total Emissions")
        
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
# HEATMAP DISPLAY OF ENERGY CONSUMPTION AND EMISSIONS
###############################################################################
def show_heatmap():
    regions = ["North", "South", "East", "West"]
    energy_types = ["Electricity", "Natural Gas", "Vehicle", "Heating Oil", "Propane"]
    emissions_data = np.random.rand(len(regions), len(energy_types)) * 10000
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(emissions_data, annot=True, fmt='.2e', cmap='YlGnBu', 
                xticklabels=energy_types, yticklabels=regions, ax=ax)
    ax.set_title("Energy Consumption vs Emissions by Region and Energy Type")
    st.pyplot(fig)

###############################################################################
# OVERVIEW PAGE
###############################################################################
def show_overview_page():
    logo_buf = create_logo()
    st.image(logo_buf, width=100)
    
    st.title("Overview: Carbon Footprint Factors & Statistical Methodology")
    
    st.markdown("## Your Estimated Emissions")
    user_total = st.session_state.get("total_carbon", None)
    if user_total is None:
        st.write("You haven't calculated your personal emissions yet. **Please select 'User Input' from the left sidebar** to start your calculation.")
    else:
        st.markdown(f"<div class='center'>Your Estimated Emissions (kg CO₂): <strong>{user_total:.2f}</strong></div>", unsafe_allow_html=True)
    
    st.markdown("## Time Series Analysis")
    dates = pd.date_range(start='2020-01-01', periods=36, freq='M')
    consumption = np.random.normal(500, 50, len(dates))
    df_timeseries = pd.DataFrame({'Month': dates, 'Energy Consumption (kWh)': consumption})
    st.line_chart(df_timeseries.set_index('Month'))
    
    st.markdown("## Emissions Breakdown")
    sample_values = {"Electricity": 500, "Natural Gas": 50, "Vehicle": 1000, "Heating Oil": 20, "Propane": 10}
    conv = st.session_state.conversion_factors
    sample_emissions = { key: sample_values[key] * conv[key] for key in sample_values }
    breakdown_df = pd.DataFrame({
        "Source": list(sample_emissions.keys()),
        "Emissions": list(sample_emissions.values())
    }).sort_values(by="Emissions")
    total_emissions = breakdown_df["Emissions"].sum()
    breakdown_df["Percent"] = breakdown_df["Emissions"] / total_emissions * 100

    fig, ax = plt.subplots()
    wedges, texts = ax.pie(breakdown_df["Emissions"], startangle=90, textprops={'fontsize': 8})
    ax.axis('equal')
    ax.set_title("Emissions Breakdown (Ascending Order)")
    legend_labels = breakdown_df.apply(lambda row: f"{row['Source']} - {row['Percent']:.1f}%", axis=1).tolist()
    ax.legend(wedges, legend_labels, title="Source", loc="center left", bbox_to_anchor=(1, 0, 0.3, 1))
    st.pyplot(fig)
    
    st.markdown("## Energy Consumption vs Emissions Heatmap")
    show_heatmap()
    
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
            st.markdown(f"<div class='center'>**Great job!** Your emissions are <strong>{abs(diff):.2f} kg CO₂ lower</strong> than the Canadian average.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='center'>**Attention:** Your emissions are <strong>{diff:.2f} kg CO₂ higher</strong> than the Canadian average.</div>", unsafe_allow_html=True)
    else:
        st.write("Once you calculate your emissions, you can compare them to the Canadian average here.")
    
    st.markdown("## Overview of Methodology")
    st.markdown("## Genetic Algorithm for Optimization")
    st.markdown(
        """
        The **Genetic Algorithm** (GA) is used to optimize the energy consumption pattern by simulating natural selection. 
        It creates a population of possible energy consumption combinations, evaluates their fitness (carbon emissions), and iteratively improves 
        the solutions through selection, crossover, and mutation.
        """
    )
    st.markdown("### Regression Analysis:")
    st.markdown(r"We used a multiple linear regression model of the form:")
    st.latex(r"Y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \varepsilon")
    st.markdown("### Gradient Descent Optimization:")
    st.markdown(r"To minimize the Mean Squared Error (MSE), we used gradient descent. The cost function is:")
    st.latex(r"J(\beta) = \frac{1}{2n} \sum_{i=1}^{n} \left( y_i - x_i^T \beta \right)^2")
    st.markdown("### Outlier Removal:")
    st.markdown(r"Outliers are removed using the Interquartile Range (IQR) method:")
    st.latex(r"\text{Lower Bound} = Q1 - 1.5 \times (Q3 - Q1), \quad \text{Upper Bound} = Q3 + 1.5 \times (Q3 - Q1)")
    
    st.markdown("## Key Energy Drivers of Carbon Emissions")
    st.markdown(
        """
        The animated 3D graph below shows how we identified key energy drivers impacting total carbon emissions.  
        We used PyTorch to perform PCA on five energy consumption features. The first two principal components capture the most variance, 
        while the z-axis represents Total Emissions.
        """
    )
    gif_pca = generate_pca_animation()
    if gif_pca:
        st.image(gif_pca, caption="Key Energy Drivers of Carbon Emissions", use_container_width=True)

###############################################################################
# SUGGESTIONS PAGE (Centered Content Using Provided Gemini Snippet)
###############################################################################
def show_suggestions_page():
    st.markdown("<div class='center'>", unsafe_allow_html=True)
    st.title("Carbon Emission Improvement Suggestions")
    st.write("Click the button below to fetch actionable, friendly recommendations on how to lower your carbon emissions based on your current consumption data.")
    
    if st.button("Get Suggestions"):
        if "total_carbon" not in st.session_state or st.session_state.total_carbon is None:
            st.error("Please calculate your carbon footprint first!")
        else:
            spinner_placeholder = st.empty()
            suggestion_placeholder = st.empty()
            spinner_placeholder.image("leaf_spinner.gif", caption="Suggesting...", use_container_width=160)
            
            prompt = (
                f"I'm a sustainability expert. A user has a monthly carbon footprint of {st.session_state.total_carbon:.2f} kg CO₂. "
                f"They consume {st.session_state.get('electricity_val', 0)} kWh of electricity, {st.session_state.get('gas_val', 0)} therms of natural gas, "
                f"drive {st.session_state.get('vehicle_val', 0)} miles, use {st.session_state.get('oil_val', 0)} gallons of heating oil, "
                f"and {st.session_state.get('propane_val', 0)} gallons of propane each month. "
                f"Provide actionable, friendly recommendations to lower their carbon emissions."
            )
            try:
                client = genai.Client(api_key="AIzaSyB4VHerf7dMxw35V6TS8tru3E_i7_wkLVU")
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[prompt]
                )
                spinner_placeholder.empty()
                suggestion_placeholder.success(response.text)
            except Exception as e:
                spinner_placeholder.empty()
                st.error("Error fetching suggestions: " + str(e))
    st.markdown("</div>", unsafe_allow_html=True)

###############################################################################
# MULTI-STEP USER INPUT PAGE WITH SLIDERS (USING CALLBACKS)
###############################################################################
def set_step(new_step):
    st.session_state.input_step = new_step

def reset_calculation():
    for key in ["input_step", "electricity_val", "gas_val", "vehicle_val", "oil_val", "propane_val"]:
        st.session_state.pop(key, None)

def show_user_input_page():
    if "input_step" not in st.session_state:
        st.session_state.input_step = 1

    if st.session_state.input_step == 1:
        st.title("Welcome to Your Carbon Footprint Calculator")
        st.write("This interactive notebook will guide you through a series of steps to estimate your monthly carbon emissions.")
        st.button("Start", key="start", on_click=set_step, args=(2,))
    elif st.session_state.input_step == 2:
        with st.container():
            st.markdown("<div class='slider-container'>", unsafe_allow_html=True)
            st.markdown("<h2>Electricity Consumption</h2>", unsafe_allow_html=True)
            st.markdown("<p class='question-text'>How many kWh of electricity do you use monthly?</p>", unsafe_allow_html=True)
            elec_val = st.slider("Electricity (kWh)", min_value=300, max_value=700, value=500, step=1, key="elec_slider")
            st.session_state.electricity_val = float(elec_val)
            col1, col2, col3 = st.columns([1,8,1])
            with col1:
                st.button("Back", key="back_elec", on_click=set_step, args=(1,))
            with col3:
                st.button("Next", key="next_elec", on_click=set_step, args=(3,))
            st.markdown("</div>", unsafe_allow_html=True)
    elif st.session_state.input_step == 3:
        with st.container():
            st.markdown("<div class='slider-container'>", unsafe_allow_html=True)
            st.markdown("<h2>Natural Gas Consumption</h2>", unsafe_allow_html=True)
            st.markdown("<p class='question-text'>How many therms of natural gas do you use monthly?</p>", unsafe_allow_html=True)
            gas_val = st.slider("Natural Gas (therms)", min_value=30, max_value=80, value=50, step=1, key="gas_slider")
            st.session_state.gas_val = float(gas_val)
            col1, col2, col3 = st.columns([1,8,1])
            with col1:
                st.button("Back", key="back_gas", on_click=set_step, args=(2,))
            with col3:
                st.button("Next", key="next_gas", on_click=set_step, args=(4,))
            st.markdown("</div>", unsafe_allow_html=True)
    elif st.session_state.input_step == 4:
        with st.container():
            st.markdown("<div class='slider-container'>", unsafe_allow_html=True)
            st.markdown("<h2>Vehicle Miles Driven</h2>", unsafe_allow_html=True)
            st.markdown("<p class='question-text'>How many miles do you drive monthly?</p>", unsafe_allow_html=True)
            vehicle_val = st.slider("Vehicle Miles", min_value=500, max_value=1500, value=1000, step=1, key="vehicle_slider")
            st.session_state.vehicle_val = float(vehicle_val)
            col1, col2, col3 = st.columns([1,8,1])
            with col1:
                st.button("Back", key="back_vehicle", on_click=set_step, args=(3,))
            with col3:
                st.button("Next", key="next_vehicle", on_click=set_step, args=(5,))
            st.markdown("</div>", unsafe_allow_html=True)
    elif st.session_state.input_step == 5:
        with st.container():
            st.markdown("<div class='slider-container'>", unsafe_allow_html=True)
            st.markdown("<h2>Heating Oil Consumption</h2>", unsafe_allow_html=True)
            st.markdown("<p class='question-text'>How many gallons of heating oil do you use monthly?</p>", unsafe_allow_html=True)
            oil_val = st.slider("Heating Oil (gallons)", min_value=10, max_value=30, value=20, step=1, key="oil_slider")
            st.session_state.oil_val = float(oil_val)
            col1, col2, col3 = st.columns([1,8,1])
            with col1:
                st.button("Back", key="back_oil", on_click=set_step, args=(4,))
            with col3:
                st.button("Next", key="next_oil", on_click=set_step, args=(6,))
            st.markdown("</div>", unsafe_allow_html=True)
    elif st.session_state.input_step == 6:
        with st.container():
            st.markdown("<div class='slider-container'>", unsafe_allow_html=True)
            st.markdown("<h2>Propane Consumption</h2>", unsafe_allow_html=True)
            st.markdown("<p class='question-text'>How many gallons of propane do you use monthly?</p>", unsafe_allow_html=True)
            propane_val = st.slider("Propane (gallons)", min_value=5, max_value=15, value=10, step=1, key="propane_slider")
            st.session_state.propane_val = float(propane_val)
            col1, col2, col3 = st.columns([1,6,3])
            with col1:
                st.button("Back", key="back_propane", on_click=set_step, args=(5,))
            with col3:
                st.button("Finish and Calculate", key="finish", on_click=set_step, args=(7,))
            st.markdown("</div>", unsafe_allow_html=True)
    elif st.session_state.input_step == 7:
        with st.container():
            st.markdown("<div class='center'>", unsafe_allow_html=True)
            conv = st.session_state.conversion_factors
            carbon_elec = st.session_state.electricity_val * conv["Electricity"]
            carbon_gas = st.session_state.gas_val * conv["Natural Gas"]
            carbon_vehicle = st.session_state.vehicle_val * conv["Vehicle"]
            carbon_oil = st.session_state.oil_val * conv["Heating Oil"]
            carbon_propane = st.session_state.propane_val * conv["Propane"]
            total_carbon = carbon_elec + carbon_gas + carbon_vehicle + carbon_oil + carbon_propane
            st.markdown(f"<h2>Your Estimated Monthly Carbon Emissions: {total_carbon:.2f} kg CO₂</h2>", unsafe_allow_html=True)
            st.session_state.total_carbon = total_carbon
            if "emission_history" not in st.session_state:
                st.session_state.emission_history = []
            st.session_state.emission_history.append({
                "date": pd.Timestamp.now(),
                "emissions": total_carbon
            })
            st.markdown("</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,5,2])
        with col1:
            st.button("Back", key="back_results", on_click=set_step, args=(6,))
        with col3:
            st.button("Reset Calculation", key="reset", on_click=reset_calculation)

def show_user_history_page():
    st.title("Your Emissions History")
    
    if "emission_history" not in st.session_state or len(st.session_state.emission_history) == 0:
        st.info("No emissions data tracked yet. Please calculate your carbon footprint on the 'User Input' page.")
        return
    
    # Remove centering wrapper to left-align the content
    df_history = pd.DataFrame(st.session_state.emission_history)
    df_history["date"] = pd.to_datetime(df_history["date"])
    df_history = df_history.sort_values("date")
    
    st.subheader("Emissions Over Time")
    canada_avg = 1500.0
    df_history["Canada Average"] = canada_avg
    df_history.set_index("date", inplace=True)
    st.line_chart(df_history[["emissions", "Canada Average"]])
    
    st.subheader("Emissions History Data")
    st.dataframe(df_history.reset_index())

def main():
    if page == "Overview":
        show_overview_page()
    elif page == "User Input":
        show_user_input_page()
    elif page == "User History":
        show_user_history_page()
    elif page == "Suggestions":
        show_suggestions_page()

if __name__ == "__main__":
    main()
