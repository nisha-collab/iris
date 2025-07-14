import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

@st.cache_resource
def train_iris_model(data_df, feature_names):
    """
    Trains a RandomForestClassifier model on the Iris dataset.
    Caches the trained model to avoid retraining on every rerun.
    """
    X = data_df[feature_names]
    y = data_df['species']
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model
