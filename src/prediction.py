import joblib
import numpy as np


def load_model(file_path="models/iris_model.pkl"):
    """
    Loads trained model from disk.
    """
    return joblib.load(file_path)


def predict(model, scaler, input_data):
    """
    Predict iris class from raw input.

    input_data format:
    [sepal_length, sepal_width, petal_length, petal_width]
    """
    input_array = np.array(input_data).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)
    return prediction[0]