import pandas as pd
from sklearn.ensemble import RandomForestClassifier # Import for type hinting if desired

def predict_species(model: RandomForestClassifier, input_features: list, feature_names: list, species_names: list):
    """
    Makes a prediction on the input features using the trained model.

    Args:
        model (RandomForestClassifier): The trained machine learning model.
        input_features (list): A list of numerical input features
                                [sepal_length, sepal_width, petal_length, petal_width].
        feature_names (list): List of feature names corresponding to the model's training.
        species_names (list): List of species names for mapping the prediction.

    Returns:
        str: The predicted species name.
    """
    input_data = pd.DataFrame([input_features], columns=feature_names)
    prediction = model.predict(input_data)
    predicted_species_index = prediction[0]
    predicted_species_name = species_names[predicted_species_index]
    return predicted_species_name
