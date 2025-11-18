from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the Random Forest model
MODELS_PATH = 'models/'
models = {}

try:
    # Load only Random Forest model
    models['Random Forest'] = pickle.load(open(os.path.join(MODELS_PATH, 'RandomForest.pkl'), 'rb'))
    print("Random Forest model loaded successfully!")
    
    # Print model info for debugging
    print(f"Model type: {type(models['Random Forest'])}")
    if hasattr(models['Random Forest'], 'classes_'):
        print(f"Model classes: {models['Random Forest'].classes_}")
    
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.get_json()
        print(f"Received data: {data}")  # Debug log
        
        # Get selected model
        selected_model = data.get('model', 'Random Forest')
        
        if selected_model not in models:
            return jsonify({'error': f'Model {selected_model} not found'}), 400
        
        model = models[selected_model]
        
        # Create a dictionary with all features in the correct order
        features_dict = {}
        
        # Add Time first
        features_dict['Time'] = float(data.get('Time', 0))
        
        # Add V1 to V28
        for i in range(1, 29):
            features_dict[f'V{i}'] = float(data.get(f'V{i}', 0))
        
        # Add Amount last
        features_dict['Amount'] = float(data.get('Amount', 0))
        
        # Create DataFrame with the exact same column structure as training data
        features_df = pd.DataFrame([features_dict])
        
        # Ensure column order matches the training data
        expected_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        features_df = features_df[expected_columns]
        
        print(f"Features DataFrame shape: {features_df.shape}")  # Debug log
        print(f"Features DataFrame columns: {features_df.columns.tolist()}")  # Debug log
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        
        # Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(features_df)[0]
            fraud_probability = float(prediction_proba[1])
            legitimate_probability = float(prediction_proba[0])
        else:
            # If model doesn't have predict_proba, use decision function or default values
            fraud_probability = 1.0 if prediction == 1 else 0.0
            legitimate_probability = 1.0 if prediction == 0 else 0.0
        
        # Determine confidence level (higher of the two probabilities)
        confidence = max(fraud_probability, legitimate_probability) * 100
        
        # Prepare response
        result = {
            'prediction': int(prediction),
            'probability_fraud': fraud_probability,
            'probability_legitimate': legitimate_probability,
            'model_used': selected_model,
            'status': 'Fraudulent' if prediction == 1 else 'Legitimate',
            'confidence': round(confidence, 2)
        }
        
        print(f"Prediction result: {result}")  # Debug log
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")  # Debug log
        return jsonify({'error': str(e)}), 500

@app.route('/models')
def get_models():
    return jsonify({'models': list(models.keys())})

@app.route('/sample_data')
def sample_data():
    """Provide sample data for testing"""
    sample = {
        'Time': 0.0,
        'Amount': 100.0,
    }
    # Add sample V values (typically these would be PCA components)
    for i in range(1, 29):
        sample[f'V{i}'] = 0.0
    
    return jsonify(sample)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)