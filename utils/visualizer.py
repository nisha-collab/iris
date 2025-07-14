import plotly.express as px
import pandas as pd

def create_sepal_plot(df: pd.DataFrame):
    """
    Creates a scatter plot of Sepal Length vs. Sepal Width,
    colored by species.
    """
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
    return fig

def create_petal_plot(df: pd.DataFrame):
    """
    Creates a scatter plot of Petal Length vs. Petal Width,
    colored by species.
    """
    fig = px.scatter(
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
    return fig
