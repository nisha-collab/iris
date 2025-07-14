import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris

@st.cache_data
def load_iris_data():
    """
    Loads the Iris dataset, converts it to a DataFrame,
    and adds species names.
    Caches the data to avoid reloading on every rerun.
    """
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map(dict(zip(range(len(iris.target_names)), iris.target_names)))
    return df, iris.target_names, iris.feature_names
