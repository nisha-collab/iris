import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# --- Load and Prepare Data ---
@st.cache_data
def load_data():
    """Loads the Iris dataset."""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map(dict(zip(range(len(iris.target_names)), iris.target_names)))
    return df, iris.target_names, iris.feature_names

df, species_names, feature_names = load_data()

# --- Train Model ---
@st.cache_resource
def train_model(data_df):
    """Trains a RandomForestClassifier model."""
    X = data_df[feature_names]
    y = data_df['species']
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

model = train_model(df)

# --- Streamlit App Layout ---
st.set_page_config(page_title="Iris Species Predictor", layout="centered")

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

st.title("🌸 Iris Species Predictor")
st.write("Enter the measurements of an Iris flower to predict its species.")

# --- User Input Section ---
st.header("Input Features")

# Create sliders for each feature
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
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=feature_names)
    prediction = model.predict(input_data)
    predicted_species_index = prediction[0]
    predicted_species_name = species_names[predicted_species_index]

    st.markdown(
        f"<div class='prediction-box'>Predicted Species: **{predicted_species_name}**</div>",
        unsafe_allow_html=True
    )

st.markdown("---")

# --- Optional: Data Visualization ---
st.header("Iris Dataset Overview")
st.write("A scatter plot of the Iris dataset, colored by species.")

# Create a scatter plot using Plotly Express
fig = px.scatter(
    df,
    x="sepal length (cm)",
    y="sepal width (cm)",
    color="species_name",
    hover_data=["petal length (cm)", "petal width (cm)"],
    title="Sepal Length vs. Sepal Width by Species",
    labels={
        "sepal length (cm)": "Sepal Length",
        "sepal width (cm)": "Sepal Width",
        "species_name": "Species"
    }
)
st.plotly_chart(fig, use_container_width=True)

fig_petal = px.scatter(
    df,
    x="petal length (cm)",
    y="petal width (cm)",
    color="species_name",
    hover_data=["sepal length (cm)", "sepal width (cm)"],
    title="Petal Length vs. Petal Width by Species",
    labels={
        "petal length (cm)": "Petal Length",
        "petal width (cm)": "Petal Width",
        "species_name": "Species"
    }
)
st.plotly_chart(fig_petal, use_container_width=True)

st.markdown("---")
st.write("This app uses a RandomForestClassifier trained on the Iris dataset.")
