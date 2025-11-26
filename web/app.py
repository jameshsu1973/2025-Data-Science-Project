import json
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash
from datetime import datetime
import os

# --- Configuration ---
app = Flask(__name__)
# IMPORTANT: Set a secret key for session management
app.secret_key = 'super_secret_key_for_santander_app_123'

# --- Model Loading ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'preprocessor.pkl')

# Product names mapping
PRODUCT_NAMES = {
    'ind_ahor_fin_ult1': 'Saving Account',
    'ind_aval_fin_ult1': 'Guarantees',
    'ind_cco_fin_ult1': 'Current Accounts',
    'ind_cder_fin_ult1': 'Derivada Account',
    'ind_cno_fin_ult1': 'Payroll Account',
    'ind_ctju_fin_ult1': 'Junior Account',
    'ind_ctma_fin_ult1': 'Más particular Account',
    'ind_ctop_fin_ult1': 'particular Account',
    'ind_ctpp_fin_ult1': 'particular Plus Account',
    'ind_deco_fin_ult1': 'Short-term deposits',
    'ind_deme_fin_ult1': 'Medium-term deposits',
    'ind_dela_fin_ult1': 'Long-term deposits',
    'ind_ecue_fin_ult1': 'e-account',
    'ind_fond_fin_ult1': 'Funds',
    'ind_hip_fin_ult1': 'Mortgage',
    'ind_plan_fin_ult1': 'Pensions',
    'ind_pres_fin_ult1': 'Loans',
    'ind_reca_fin_ult1': 'Taxes',
    'ind_tjcr_fin_ult1': 'Credit Card',
    'ind_valo_fin_ult1': 'Securities',
    'ind_viv_fin_ult1': 'Home Account',
    'ind_nomina_ult1': 'Payroll',
    'ind_nom_pens_ult1': 'Pensions',
    'ind_recibo_ult1': 'Direct Debit'
}

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

# Define the steps/fields of the form
FORM_FIELDS = [
    {
        'id': 'fecha_dato',
        'label': 'Date of Data Snapshot',
        'type': 'date',
        'help': 'The snapshot date for this record (YYYY-MM-DD).',
        'required': True
    },
    {
        'id': 'ncodpers',
        'label': 'Customer Code (ID)',
        'type': 'number',
        'help': 'Unique identifier for the customer.',
        'min': 1,
        'required': True
    },
    {
        'id': 'ind_empleado',
        'label': 'Employee Index',
        'type': 'select',
        'help': 'The type of employee relationship with the bank.',
        'options': [
            {'value': 'A', 'text': 'Active Employee'},
            {'value': 'B', 'text': 'Ex-Employee'},
            {'value': 'F', 'text': 'Subsidiary/Affiliate'},
            {'value': 'N', 'text': 'Not Employee'},
            {'value': 'P', 'text': 'Passive Employee'}
        ],
        'required': True
    },
    {
        'id': 'pais_residencia',
        'label': 'Country of Residence',
        'type': 'text',
        'help': 'The customer\'s two-letter country code (e.g., ES for Spain, US for United States).',
        'required': True
    },
    {
        'id': 'sexo',
        'label': 'Gender',
        'type': 'radio',
        'help': 'The customer\'s gender.',
        'options': [
            {'value': 'H', 'text': 'Male (Hombre)'},
            {'value': 'V', 'text': 'Female (Mujer)'}
        ],
        'required': True
    },
    {
        'id': 'age',
        'label': 'Age',
        'type': 'number',
        'help': 'The customer\'s age in years.',
        'min': 0,
        'required': True
    },
    {
        'id': 'fecha_alta',
        'label': 'Date of Joining the Bank',
        'type': 'date',
        'help': 'The date when the customer joined the bank (YYYY-MM-DD).',
        'required': True
    },
    {
        'id': 'ind_nuevo',
        'label': 'New Customer Indicator',
        'type': 'radio',
        'help': 'Whether the customer is registered in the last 6 months.',
        'options': [
            {'value': '1', 'text': 'Yes'},
            {'value': '0', 'text': 'No'}
        ],
        'required': True
    },
    {
        'id': 'antiguedad',
        'label': 'Customer Seniority (Months)',
        'type': 'number',
        'help': 'Number of months the customer has been with the bank.',
        'min': 0,
        'required': True
    },
    {
        'id': 'indrel',
        'label': 'Relationship Status with Bank',
        'type': 'radio',
        'help': 'Primary customer if you are currently active, Former customer otherwise.',
        'options': [
            {'value': '1', 'text': 'Primary Customer'},
            {'value': '99', 'text': 'Former Customer'}
        ],
        'required': True
    },
    {
        'id': 'ult_fec_cli_1t',
        'label': 'Last Date as Primary Customer',
        'type': 'date',
        'help': 'The last date the customer was classified as a primary customer (only applies if Relationship Status is "Former").',
        'required': False
    },
    {
        'id': 'indrel_1mes',
        'label': 'Relationship Status Last Month',
        'type': 'radio',
        'help': 'Relationship status of the customer in the previous month.',
        'options': [
            {'value': '1', 'text': 'Primary Customer'},
            {'value': '2', 'text': 'Co-owner'},
            {'value': 'P', 'text': 'Potential Customer'},
            {'value': '3', 'text': 'Former Primary Customer'},
            {'value': '4', 'text': 'Former Co-owner'}
        ],
        'required': True
    },
    {
        'id': 'tiprel_1mes',
        'label': 'Type of Relationship Last Month',
        'type': 'radio',
        'help': 'Type of relationship the customer had with the bank in the previous month.',
        'options': [
            {'value': 'A', 'text': 'Active'},
            {'value': 'I', 'text': 'Inactive'},
            {'value': 'P', 'text': 'Former Customer'},
            {'value': 'R', 'text': 'Potential Customer'}
        ],
        'required': True
    },
    {
        'id': 'indresi',
        'label': 'Resident Indicator',
        'type': 'radio',
        'help': 'Yes if the residence country is the same as the bank country, No otherwise.',
        'options': [
            {'value': 'S', 'text': 'Resident'},
            {'value': 'N', 'text': 'Non-Resident'}
        ],
        'required': True
    },
    {
        'id': 'indext',
        'label': 'Foreigner Indicator',
        'type': 'radio',
        'help': 'Yes if the customer is a foreigner, No otherwise.',
        'options': [
            {'value': 'S', 'text': 'Foreigner'},
            {'value': 'N', 'text': 'Not Foreigner'}
        ],
        'required': True
    },
    {
        'id': 'conyuemp',
        'label': 'Spouse Employee Indicator',
        'type': 'radio',
        'help': 'Yes if the customer\'s spouse is an employee of the bank, No otherwise.',
        'options': [
            {'value': '1', 'text': 'Spouse is Employee'},
            {'value': '', 'text': 'Spouse is Not Employee'}
        ],
        'required': True
    },
    {
        'id': 'canal_entrada',
        'label': 'Channel of Entry',
        'type': 'select',
        'help': 'The channel through which the customer entered the bank.',
        'options': [
            {'value': 'KHE', 'text': 'KHE (Branch)'},
            {'value': 'KFA', 'text': 'KFA (Phone)'},
            {'value': 'KFC', 'text': 'KFC (Internet)'},
            {'value': 'KFB', 'text': 'KFB (Mobile)'},
            {'value': 'KED', 'text': 'KED (Other)'}
        ],
        'required': True
    },
    {
        'id': 'indfall',
        'label': 'Deceased Indicator',
        'type': 'radio',
        'help': 'Whether the customer is deceased or not.',
        'options': [
            {'value': 'N', 'text': 'Deceased'},
            {'value': 'S', 'text': 'Not Deceased'}
        ],
        'required': True
    },
    {
        'id': 'tipodom',
        'label': 'Type of Address',
        'type': 'radio',
        'help': 'Whether the address is primary or not.',
        'options': [
            {'value': '1', 'text': 'Primary'},
            {'value': '0', 'text': 'Not Primary'}
        ],
        'required': True
    },
    {
        'id': 'cod_prov',
        'label': 'Province/State Code',
        'type': 'number',
        'help': 'The code of the province/state where the customer resides.',
        'min': 0,
        'required': True
    },
    {
        'id': 'nomprov',
        'label': 'Province/State Name',
        'type': 'text',
        'help': 'The name of the province/state where the customer resides.',
        'required': True
    },
    {
        'id': 'ind_actividad_cliente',
        'label': 'Customer Activity Index',
        'type': 'radio',
        'help': 'Whether the customer is active or inactive.',
        'options': [
            {'value': '1', 'text': 'Active Customer'},
            {'value': '0', 'text': 'Inactive Customer'}
        ],
        'required': True
    },
    {
        'id': 'renta',
        'label': 'Gross Income (Annual)',
        'type': 'number',
        'help': 'The customer\'s gross income, used for segmentation.',
        'min': 0,
        'required': True
    },
    {
        'id': 'segmento',
        'label': 'Customer Segment',
        'type': 'select',
        'help': 'The customer\'s segment within the bank.',
        'options': [
            {'value': '01 - TOP', 'text': 'VIP'},
            {'value': '02 - PARTICULARES', 'text': 'Individuals'},
            {'value': '03 - UNIVERSITARIO', 'text': 'College Graduated'}
        ],
        'required': True
    }
]

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
            
            # Data preprocessing (apply the same preprocessing as training)
            X = preprocessor.transform(df)
            
            # Make prediction
            # The model should output probabilities for each of the 24 products
            prediction_probs = model.predict_proba(X)
            
            # Get top 7 product recommendations
            # Assuming prediction_probs is a 2D array where each column represents a product
            top_indices = np.argsort(prediction_probs[0])[::-1][:7]
            
            product_list = list(PRODUCT_NAMES.keys())
            for idx in top_indices:
                product_id = product_list[idx]
                product_name = PRODUCT_NAMES[product_id]
                probability = prediction_probs[0][idx]
                
                recommendations.append({
                    'product_name': product_name,
                    'probability': f"{probability * 100:.2f}%",
                    'probability_value': probability
                })
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            recommendations = None
    
    # Save form data to session for potential re-use
    form_data_backup = session['form_data'].copy()
    
    # Clear the session data after successful submission
    session.pop('form_data', None)

    return render_template('submit.html', 
                         final_data=final_data,
                         recommendations=recommendations,
                         model_available=(model is not None))

if __name__ == '__main__':
    # Run the application
    app.run(debug=True)
