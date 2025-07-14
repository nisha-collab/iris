import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# --- TEMPORARY PATH FIX FOR LOCAL DEVELOPMENT ---
# Get the absolute path to the directory containing app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path to the 'iris' root directory (parent of 'app' directory)
project_root = os.path.join(current_dir, '..')
# Add the project root to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END TEMPORARY PATH FIX ---


# Import functions from our custom modules, now with relative paths
from data.data_loader import load_iris_data
from model.model_trainer import train_iris_model
from utils.visualizer import create_sepal_plot, create_petal_plot
from model.predictor import predict_species # predictor is also in 'model'

# --- Streamlit App Layout ---
st.set_page_config(page_title="Modular Iris Species Predictor", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1 {
        color: #2e8b57;
        text-align: center;
    }
    h2 {
        color: #367c39;
    }
    .prediction-box {
        background-color: #e6ffe6;
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 15px;
        margin-top: 20px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: #2e8b57;
    }
    .stSlider > div > div > div > div {
        background-color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌸 Modular Iris Species Predictor")
st.write("Enter the measurements of an Iris flower to predict its species.")

# --- Load Data and Train Model (cached) ---
# Load data using the function from data_loader.py
df, species_names, feature_names = load_iris_data()

# Train model using the function from model_trainer.py
model = train_iris_model(df, feature_names)

# --- User Input Section ---
st.header("Input Features")

# Create sliders for each feature using min/max/mean from the loaded data
sepal_length = st.slider(
    "Sepal Length (cm)",
    min_value=float(df['sepal length (cm)'].min()),
    max_value=float(df['sepal length (cm)'].max()),
    value=float(df['sepal length (cm)'].mean()),
    step=0.1
)
sepal_width = st.slider(
    "Sepal Width (cm)",
    min_value=float(df['sepal width (cm)'].min()),
    max_value=float(df['sepal width (cm)'].max()),
    value=float(df['sepal width (cm)'].mean()),
    step=0.1
)
petal_length = st.slider(
    "Petal Length (cm)",
    min_value=float(df['petal length (cm)'].min()),
    max_value=float(df['petal length (cm)'].max()),
    value=float(df['petal length (cm)'].mean()),
    step=0.1
)
petal_width = st.slider(
    "Petal Width (cm)",
    min_value=float(df['petal width (cm)'].min()),
    max_value=float(df['petal width (cm)'].max()),
    value=float(df['petal width (cm)'].mean()),
    step=0.1
)

# --- Prediction ---
if st.button("Predict Species"):
    input_features = [sepal_length, sepal_width, petal_length, petal_width]
    # Use the predict_species function from predictor.py
    predicted_species_name = predict_species(model, input_features, feature_names, species_names)

    st.markdown(
        f"<div class='prediction-box'>Predicted Species: **{predicted_species_name}**</div>",
        unsafe_allow_html=True
    )

st.markdown("---")

# --- Data Visualization ---
st.header("Iris Dataset Overview")
st.write("Scatter plots of the Iris dataset, colored by species.")

# Use visualization functions from visualizer.py
st.plotly_chart(create_sepal_plot(df), use_container_width=True)
st.plotly_chart(create_petal_plot(df), use_container_width=True)

st.markdown("---")
st.write("This app uses a RandomForestClassifier trained on the Iris dataset.")

# --- Attribution ---
st.markdown("<p style='text-align: center; color: gray; font-size: 0.9em;'>Made by Nisha Pal</p>", unsafe_allow_html=True)
