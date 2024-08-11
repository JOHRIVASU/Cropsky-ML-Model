from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load your Keras model
model = load_model("C:/Users/VASU/OneDrive/Desktop/plant-disease-prediction-cnn-deep-leanring-project-main/app/trained_model/plant_disease_model.h5")

# Route for handling the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.json['data']

    # Preprocess the data as required by your model
    # For example, if your model expects a numpy array:
    input_data = np.array(data).reshape(1, -1)  # Adjust shape as per your model's requirement

    # Make prediction using the loaded model
    prediction = model.predict(input_data)

    # Return the result as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    # Run the app on the local development server
    app.run(debug=True)
