"""
Test the prediction functionality by simulating a customer profile
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Preprocessor Class (必須定義才能載入 pickle)
from src.preprocessor import SantanderPreprocessor
from src.utils.const import PRODUCT_NAMES

import pickle
import pandas as pd
import numpy as np

# Sample customer data - 使用者填寫的資料
customer_data = {
    'fecha_dato': '2020-02-26',
    'ncodpers': '15889',
    'ind_empleado': 'A',
    'pais_residencia': 'US',
    'sexo': 'H',
    'age': '35',
    'fecha_alta': '2021-02-26',
    'ind_nuevo': '1',
    'antiguedad': '35',
    'indrel': '1',
    'ult_fec_cli_1t': '2025-11-19',
    'indrel_1mes': '1',
    'tiprel_1mes': 'A',
    'indresi': 'S',
    'indext': 'N',
    'conyuemp': '1',
    'canal_entrada': 'KFA',
    'indfall': 'N',
    'tipodom': '1',
    'cod_prov': '8',
    'nomprov': 'MADRID',
    'ind_actividad_cliente': '1',
    'renta': '50000',
    'segmento': '02 - PARTICULARES'
}

print("="*70)
print("Testing Santander Product Recommendation System")
print("="*70)
print("\nCustomer Profile:")
print("-"*70)
for key, value in customer_data.items():
    print(f"{key:25s}: {value}")

print("\n" + "="*70)
print("Loading model...")

# Load model and preprocessor
try:
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    print("✓ Model loaded successfully!")
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    exit(1)

print("\nMaking prediction...")

# Convert to DataFrame
df = pd.DataFrame([customer_data])

# Preprocess
X = preprocessor.transform(df)

# Predict
try:
    prediction_probs = model.predict_proba(X)
    
    # Get top 7 recommendations
    import numpy as np
    
    # For MultiOutputClassifier, prediction_probs is a list of arrays
    # We need to extract the probability of class 1 (having the product) for each output
    product_probs = []
    for i, pred in enumerate(prediction_probs):
        # pred is array of shape (1, 2) for binary classification
        # We want probability of class 1 (having the product)
        prob = pred[0][1] if pred.shape[1] > 1 else pred[0][0]
        product_probs.append(prob)
    
    product_probs = np.array(product_probs)
    top_indices = np.argsort(product_probs)[::-1][:7]
    
    print("\n" + "="*70)
    print("TOP 7 RECOMMENDED PRODUCTS")
    print("="*70)
    
    product_list = list(PRODUCT_NAMES.keys())
    for rank, idx in enumerate(top_indices, 1):
        product_id = product_list[idx]
        product_name = PRODUCT_NAMES[product_id]
        probability = product_probs[idx]
        
        # Visual bar
        bar_length = int(probability * 40)
        bar = '█' * bar_length + '░' * (40 - bar_length)
        
        print(f"\n{rank}. {product_name}")
        print(f"   {bar} {probability*100:.2f}%")
    
    print("\n" + "="*70)
    print("Prediction complete!")
    print("="*70)

except Exception as e:
    print(f"✗ Error during prediction: {e}")
    import traceback
    traceback.print_exc()
