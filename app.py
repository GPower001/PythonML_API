from flask import Flask, request, jsonify
from flask_cors import CORS 
import model 
import os

app = Flask(__name__)
CORS(app) 

# Load and train the model and supplementary data when the Flask app starts
# This ensures all data and the model are ready before any requests come in
with app.app_context():
    model.load_and_train_model()

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict disease based on symptoms.
    Expects a JSON payload with symptom data.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    patient_symptoms = data.get('symptoms')

    if not patient_symptoms or not isinstance(patient_symptoms, dict):
        return jsonify({"error": "Invalid or missing 'symptoms' in request body. Expected a dictionary."}), 400

    try:
        predicted_disease = model.get_predicted_value(patient_symptoms)
        return jsonify({"predicted_disease": predicted_disease}), 200
    except Exception as e:
        # Log the full error for debugging
        app.logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during prediction.", "details": str(e)}), 500

@app.route('/recommendations', methods=['POST'])
def get_recommendations_api():
    """
    API endpoint to get recommendations for a given disease.
    Expects a JSON payload with the disease name.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    disease_name = data.get('disease')

    if not disease_name or not isinstance(disease_name, str):
        return jsonify({"error": "Invalid or missing 'disease' in request body. Expected a string."}), 400

    try:
        recommendations = model.get_recommendations(disease_name)
        return jsonify(recommendations), 200
    except Exception as e:
        # Log the full error for debugging
        app.logger.error(f"Error during recommendation lookup: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during recommendation lookup.", "details": str(e)}), 500

@app.route('/get_symptom_list', methods=['GET'])
def get_symptom_list():
    """
    API endpoint to return the full list of symptom names.
    This is used by the Node.js backend to format symptom input.
    """
    if not model.all_symptoms_columns:
        # If model not loaded yet, try to load it
        try:
            model.load_and_train_model()
        except Exception as e:
            app.logger.error(f"Error loading model for symptom list: {e}", exc_info=True)
            return jsonify({"error": "Model not loaded, cannot provide symptom list."}), 500
    return jsonify({"symptoms": model.all_symptoms_columns}), 200

@app.route('/')
def health_check():
    """Basic health check endpoint."""
    return "Disease Prediction and Recommendation API is running!"

if __name__ == '__main__':
    # Run the Flask app
    # In a production environment, use a production-ready WSGI server like Gunicorn
    app.run(debug=True, host='0.0.0.0', port=5000)
