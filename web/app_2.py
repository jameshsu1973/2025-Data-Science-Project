import json
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash
from datetime import datetime
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Preprocessor Class ---
class SantanderPreprocessor:
    """Custom preprocessor for Santander data"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_cols = []
        self.numerical_cols = []
        
    def fit(self, df):
        """Fit the preprocessor on training data"""
        df = df.copy()
        
        # Identify categorical and numerical columns
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Fit label encoders for categorical columns
        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = df[col].fillna('Unknown')
            le.fit(df[col])
            self.label_encoders[col] = le
        
        # Prepare numerical data for scaling
        df_encoded = df.copy()
        for col in self.categorical_cols:
            df_encoded[col] = self.label_encoders[col].transform(df[col])
        
        # Fill missing values in numerical columns
        for col in self.numerical_cols:
            df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())
        
        # Fit scaler
        self.scaler.fit(df_encoded)
        
        return self
    
    def transform(self, df):
        """Transform new data"""
        df = df.copy()
        
        # Handle categorical columns
        for col in self.categorical_cols:
            df[col] = df[col].fillna('Unknown')
            # Handle unknown categories
            le = self.label_encoders[col]
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
            df[col] = le.transform(df[col])
        
        # Handle numerical columns
        for col in self.numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)
        
        # Scale the data
        X_scaled = self.scaler.transform(df)
        
        return X_scaled
    
    def fit_transform(self, df):
        """Fit and transform in one step"""
        self.fit(df)
        return self.transform(df)

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
        'help': 'The snapshot date for this record. Format: YYYY-MM-DD. Example: 2016-06-28',
        'required': True
    },
    {
        'id': 'ncodpers',
        'label': 'Customer Code (ID)',
        'type': 'number',
        'help': 'Unique identifier for the customer. Range: 1-999999999. Example: 15889, 1234567',
        'min': 1,
        'max': 999999999,
        'required': True
    },
    {
        'id': 'ind_empleado',
        'label': 'Employee Index',
        'type': 'select',
        'help': 'The type of employee relationship with the bank. (Values seen in training data)',
        'options': [
            {'value': 'N', 'text': 'Not Employee'},
            {'value': 'A', 'text': 'Active Employee'},
            {'value': 'B', 'text': 'Ex-Employee'},
            {'value': 'F', 'text': 'Subsidiary/Affiliate'},
            {'value': 'P', 'text': 'Passive Employee'}
        ],
        'required': True
    },
    {
        'id': 'pais_residencia',
        'label': 'Country of Residence',
        'type': 'select',
        'help': 'Country of residence code. Only countries seen in training data are available.',
        'options': [
            {'value': 'ES', 'text': 'Spain'},
            {'value': 'CA', 'text': 'Canada'},
            {'value': 'CH', 'text': 'Switzerland'},
            {'value': 'CL', 'text': 'Chile'},
            {'value': 'IE', 'text': 'Ireland'}
        ],
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
        'help': 'The customer\'s age in years. Range: 18-100. Example: 35, 42',
        'min': 18,
        'max': 100,
        'required': True
    },
    {
        'id': 'fecha_alta',
        'label': 'Date of Joining the Bank',
        'type': 'date',
        'help': 'The date when the customer joined the bank. Format: YYYY-MM-DD. Example: 2015-01-12',
        'required': True
    },
    {
        'id': 'ind_nuevo',
        'label': 'New Customer Indicator',
        'type': 'radio',
        'help': 'Whether the customer is registered in the last 6 months. (Values seen in training data)',
        'options': [
            {'value': '0', 'text': 'No'},
            {'value': '1', 'text': 'Yes'}
        ],
        'required': True
    },
    {
        'id': 'antiguedad',
        'label': 'Customer Seniority (Months)',
        'type': 'number',
        'help': 'Number of months the customer has been with the bank. Range: 0-600. Example: 18 (1.5 years), 60 (5 years)',
        'min': 0,
        'max': 600,
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
        'help': 'Relationship status of the customer in the previous month. (Values seen in training data)',
        'options': [
            {'value': '1', 'text': 'Primary Customer'},
            {'value': '3', 'text': 'Former Primary Customer'}
        ],
        'required': True
    },
    {
        'id': 'tiprel_1mes',
        'label': 'Type of Relationship Last Month',
        'type': 'radio',
        'help': 'Type of relationship the customer had with the bank in the previous month. (Values seen in training data)',
        'options': [
            {'value': 'A', 'text': 'Active'},
            {'value': 'I', 'text': 'Inactive'}
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
        'help': 'Whether the customer\'s spouse is an employee of the bank.',
        'options': [
            {'value': 'N', 'text': 'No'},
            {'value': 'S', 'text': 'Yes'}
        ],
        'required': True
    },
    {
        'id': 'canal_entrada',
        'label': 'Channel of Entry',
        'type': 'select',
        'help': 'The channel through which the customer entered the bank.',
        'options': [
            {'value': 'KHE', 'text': 'Branch Office'},
            {'value': 'KFA', 'text': 'Phone Banking'},
            {'value': 'KFC', 'text': 'Internet Banking'},
            {'value': 'KHM', 'text': 'ATM'},
            {'value': 'RED', 'text': 'Other Channels'}
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
        'help': 'Type of address in bank records.',
        'options': [
            {'value': '1', 'text': 'Primary Address'},
            {'value': '0', 'text': 'Secondary Address'}
        ],
        'required': True
    },
    {
        'id': 'cod_prov',
        'label': 'Province/State Code',
        'type': 'select',
        'help': 'Province code for Spanish provinces. (Values seen in training data)',
        'options': [
            {'value': '1', 'text': '1 - ALAVA'},
            {'value': '2', 'text': '2 - ALBACETE'},
            {'value': '3', 'text': '3 - ALICANTE'},
            {'value': '4', 'text': '4 - ALMERIA'},
            {'value': '5', 'text': '5 - AVILA'},
            {'value': '6', 'text': '6 - BADAJOZ'},
            {'value': '7', 'text': '7 - BALEARS, ILLES'},
            {'value': '8', 'text': '8 - BARCELONA'},
            {'value': '9', 'text': '9 - BURGOS'},
            {'value': '10', 'text': '10 - CACERES'},
            {'value': '11', 'text': '11 - CADIZ'},
            {'value': '12', 'text': '12 - CASTELLON'},
            {'value': '13', 'text': '13 - CIUDAD REAL'},
            {'value': '14', 'text': '14 - CORDOBA'},
            {'value': '15', 'text': '15 - CORUÑA, A'},
            {'value': '16', 'text': '16 - CUENCA'},
            {'value': '17', 'text': '17 - GIRONA'},
            {'value': '18', 'text': '18 - GRANADA'},
            {'value': '19', 'text': '19 - GUADALAJARA'},
            {'value': '20', 'text': '20 - GIPUZKOA'},
            {'value': '21', 'text': '21 - HUELVA'},
            {'value': '22', 'text': '22 - HUESCA'},
            {'value': '23', 'text': '23 - JAEN'},
            {'value': '24', 'text': '24 - LEON'},
            {'value': '25', 'text': '25 - LERIDA'},
            {'value': '26', 'text': '26 - RIOJA, LA'},
            {'value': '27', 'text': '27 - LUGO'},
            {'value': '28', 'text': '28 - MADRID'},
            {'value': '29', 'text': '29 - MALAGA'},
            {'value': '30', 'text': '30 - MURCIA'},
            {'value': '31', 'text': '31 - NAVARRA'},
            {'value': '32', 'text': '32 - OURENSE'},
            {'value': '33', 'text': '33 - ASTURIAS'},
            {'value': '34', 'text': '34 - PALENCIA'},
            {'value': '35', 'text': '35 - PALMAS, LAS'},
            {'value': '36', 'text': '36 - PONTEVEDRA'},
            {'value': '37', 'text': '37 - SALAMANCA'},
            {'value': '38', 'text': '38 - SANTA CRUZ DE TENERIFE'},
            {'value': '39', 'text': '39 - CANTABRIA'},
            {'value': '40', 'text': '40 - SEGOVIA'},
            {'value': '41', 'text': '41 - SEVILLA'},
            {'value': '42', 'text': '42 - SORIA'},
            {'value': '43', 'text': '43 - TARRAGONA'},
            {'value': '44', 'text': '44 - TERUEL'},
            {'value': '45', 'text': '45 - TOLEDO'},
            {'value': '46', 'text': '46 - VALENCIA'},
            {'value': '47', 'text': '47 - VALLADOLID'},
            {'value': '48', 'text': '48 - BIZKAIA'},
            {'value': '49', 'text': '49 - ZAMORA'},
            {'value': '50', 'text': '50 - ZARAGOZA'},
            {'value': '51', 'text': '51 - CEUTA'},
            {'value': '52', 'text': '52 - MELILLA'}
        ],
        'required': True
    },
    {
        'id': 'nomprov',
        'label': 'Province/State Name',
        'type': 'select',
        'help': 'Province name. Must match the province code. (Values seen in training data)',
        'options': [
            {'value': 'ALAVA', 'text': 'ALAVA'},
            {'value': 'ALBACETE', 'text': 'ALBACETE'},
            {'value': 'ALICANTE', 'text': 'ALICANTE'},
            {'value': 'ALMERIA', 'text': 'ALMERIA'},
            {'value': 'ASTURIAS', 'text': 'ASTURIAS'},
            {'value': 'AVILA', 'text': 'AVILA'},
            {'value': 'BADAJOZ', 'text': 'BADAJOZ'},
            {'value': 'BALEARS, ILLES', 'text': 'BALEARS, ILLES'},
            {'value': 'BARCELONA', 'text': 'BARCELONA'},
            {'value': 'BIZKAIA', 'text': 'BIZKAIA'},
            {'value': 'BURGOS', 'text': 'BURGOS'},
            {'value': 'CACERES', 'text': 'CACERES'},
            {'value': 'CADIZ', 'text': 'CADIZ'},
            {'value': 'CANTABRIA', 'text': 'CANTABRIA'},
            {'value': 'CASTELLON', 'text': 'CASTELLON'},
            {'value': 'CEUTA', 'text': 'CEUTA'},
            {'value': 'CIUDAD REAL', 'text': 'CIUDAD REAL'},
            {'value': 'CORDOBA', 'text': 'CORDOBA'},
            {'value': 'CORUÑA, A', 'text': 'CORUÑA, A'},
            {'value': 'CUENCA', 'text': 'CUENCA'},
            {'value': 'GIPUZKOA', 'text': 'GIPUZKOA'},
            {'value': 'GIRONA', 'text': 'GIRONA'},
            {'value': 'GRANADA', 'text': 'GRANADA'},
            {'value': 'GUADALAJARA', 'text': 'GUADALAJARA'},
            {'value': 'HUELVA', 'text': 'HUELVA'},
            {'value': 'HUESCA', 'text': 'HUESCA'},
            {'value': 'JAEN', 'text': 'JAEN'},
            {'value': 'LEON', 'text': 'LEON'},
            {'value': 'LERIDA', 'text': 'LERIDA'},
            {'value': 'LUGO', 'text': 'LUGO'},
            {'value': 'MADRID', 'text': 'MADRID'},
            {'value': 'MALAGA', 'text': 'MALAGA'},
            {'value': 'MELILLA', 'text': 'MELILLA'},
            {'value': 'MURCIA', 'text': 'MURCIA'},
            {'value': 'NAVARRA', 'text': 'NAVARRA'},
            {'value': 'OURENSE', 'text': 'OURENSE'},
            {'value': 'PALENCIA', 'text': 'PALENCIA'},
            {'value': 'PALMAS, LAS', 'text': 'PALMAS, LAS'},
            {'value': 'PONTEVEDRA', 'text': 'PONTEVEDRA'},
            {'value': 'RIOJA, LA', 'text': 'RIOJA, LA'},
            {'value': 'SALAMANCA', 'text': 'SALAMANCA'},
            {'value': 'SANTA CRUZ DE TENERIFE', 'text': 'SANTA CRUZ DE TENERIFE'},
            {'value': 'SEGOVIA', 'text': 'SEGOVIA'},
            {'value': 'SEVILLA', 'text': 'SEVILLA'},
            {'value': 'SORIA', 'text': 'SORIA'},
            {'value': 'TARRAGONA', 'text': 'TARRAGONA'},
            {'value': 'TERUEL', 'text': 'TERUEL'},
            {'value': 'TOLEDO', 'text': 'TOLEDO'},
            {'value': 'VALENCIA', 'text': 'VALENCIA'},
            {'value': 'VALLADOLID', 'text': 'VALLADOLID'},
            {'value': 'ZAMORA', 'text': 'ZAMORA'},
            {'value': 'ZARAGOZA', 'text': 'ZARAGOZA'}
        ],
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
        'help': 'Annual gross income in euros. Range: 0-500,000. Example: 50000 (€50k), 101850 (€101.85k)',
        'min': 0,
        'max': 500000,
        'step': 100,
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
            import traceback
            traceback.print_exc()
            recommendations = None
    
    # Render template first before clearing session
    response = render_template('submit.html', 
                         final_data=final_data,
                         recommendations=recommendations,
                         model_available=(model is not None))
    
    # Clear the session data after rendering the template
    session.pop('form_data', None)
    
    return response

if __name__ == '__main__':
    # Run the application
    app.run(debug=True)
