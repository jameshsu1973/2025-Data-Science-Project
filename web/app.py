import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import sys

# add src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from utils.const import PRODUCT_NAMES, FORM_FIELDS
from preprocessor import SantanderPreprocessor

# --- Configuration ---
app = Flask(__name__)
# IMPORTANT: Set a secret key for session management
app.secret_key = 'super_secret_key_for_santander_app_123'

# --- Model Loading ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'preprocessor.pkl')

# Load model and preprocessor (will be None if files don't exist yet)
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
    print("Model and preprocessor loaded successfully!")
except FileNotFoundError:
    model = None
    preprocessor = None
    print("Warning: Model files not found. Prediction will not be available.") 

NUM_STEPS = len(FORM_FIELDS)

# --- Routes ---

@app.route('/')
def index():
    """Redirects to the first step of the form and initializes the session."""
    session.clear() # Clear session data on start
    session.permanent = True
    session['form_data'] = {}
    return redirect(url_for('form_step', step=1))

@app.route('/form/<int:step>', methods=['GET', 'POST'])
def form_step(step):
    """Handles the display and submission of a single form page."""
    
    # 1. Validation and Navigation
    if not (1 <= step <= NUM_STEPS):
        flash('Invalid step number.', 'error')
        return redirect(url_for('index'))

    current_field = FORM_FIELDS[step - 1]
    print("entering form step:", step, session)
    if 'form_data' not in session:
        session['form_data'] = {}

    if request.method == 'POST':
        # 2. Process submission
        
        # Get data from the submitted form
        submitted_data = request.form.to_dict()
        
        # Determine the step that was just submitted
        index = step - 1
        field_id = FORM_FIELDS[index]['id']
        # Store the data for the current step's field
        session['form_data'][field_id] = submitted_data.get(field_id)

        print("Current session data:", session)
        # 3. Handle Navigation
        action = submitted_data.get('action')
        
        if action == 'next' and step < NUM_STEPS:
            # Move to the next page
            return redirect(url_for('form_step', step=step + 1))
        
        elif action == 'previous' and step > 1:
            # Move to the previous page
            return redirect(url_for('form_step', step=step - 1))
        
        elif action == 'submit' and step == NUM_STEPS:
            # Handle final submission (store the last field's data)
            # The last field's data is already stored above when processing the POST
            return redirect(url_for('submit_form'))

    # 4. Render the current step
    
    # Get the value previously saved for the current field, if any
    current_value = session['form_data'].get(current_field['id'])

    return render_template(
        'form_page.html',
        field=current_field,
        current_step=step,
        total_steps=NUM_STEPS,
        saved_value=current_value,
        is_last_step=(step == NUM_STEPS)
    )

@app.route('/submit', methods=['GET'])
def submit_form():
    """Displays the final collected data and makes product recommendations."""
    if 'form_data' not in session or not session['form_data']:
        return redirect(url_for('index'))
    
    # Prepare data for display, mapping IDs back to human-readable labels
    final_data = []
    
    for field in FORM_FIELDS:
        field_id = field['id']
        label = field['label']
        value = session['form_data'].get(field_id, 'N/A')
        
        # Optional: convert select/radio codes to descriptive text for display
        if 'options' in field:
            display_value = next((opt['text'] for opt in field['options'] if opt['value'] == value), value)
        else:
            display_value = value

        final_data.append({'label': label, 'value': display_value})
    
    # --- Model Prediction ---
    predictions = None
    recommendations = []
    
    if model is not None and preprocessor is not None:
        try:
            # Prepare input data for prediction
            input_data = session['form_data'].copy()
            
            # Convert to DataFrame
            df = pd.DataFrame([input_data])

            columns = [field['id'] for field in FORM_FIELDS]
            df = df[columns]
            
            # Data preprocessing (apply the same preprocessing as training)
            X = preprocessor.transform(df)
            
            # Make prediction
            # The model should output probabilities for each of the 24 products
            prediction_probs_raw = model.predict_proba(X)
            prediction_probs = [x[0][1] if x.shape[1] > 1 else x[0][0] for x in prediction_probs_raw]
            
            # Get top 7 product recommendations
            # Assuming prediction_probs is a 2D array where each column represents a product
            top_indices = np.argsort(prediction_probs)[::-1][:7]
            
            product_list = list(PRODUCT_NAMES.keys())
            for idx in top_indices:
                product_id = product_list[idx]
                product_name = PRODUCT_NAMES[product_id]
                probability = prediction_probs[idx]
                
                recommendations.append({
                    'product_name': product_name,
                    'probability': f"{probability * 100:.2f}%",
                    'probability_value': probability
                })
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            recommendations = None
    
    # Render template first before clearing session
    response = render_template('submit.html', 
                         final_data=final_data,
                         recommendations=recommendations,
                         model_available=(model is not None))
    
    # Clear the session data only if recommendations were made
    if recommendations is not None:
        session.pop('form_data', None)
    
    return response

if __name__ == '__main__':
    # Run the application
    app.run(debug=True)
