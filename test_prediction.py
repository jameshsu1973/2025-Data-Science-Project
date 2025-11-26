"""
Test the prediction functionality by simulating a customer profile
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Preprocessor Class (必須定義才能載入 pickle)
class SantanderPreprocessor:
    """Custom preprocessor for Santander data"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_cols = []
        self.numerical_cols = []
        
    def fit(self, df):
        df = df.copy()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = df[col].fillna('Unknown')
            le.fit(df[col])
            self.label_encoders[col] = le
        
        df_encoded = df.copy()
        for col in self.categorical_cols:
            df_encoded[col] = self.label_encoders[col].transform(df[col])
        
        for col in self.numerical_cols:
            df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())
        
        self.scaler.fit(df_encoded)
        return self
    
    def transform(self, df):
        df = df.copy()
        
        for col in self.categorical_cols:
            df[col] = df[col].fillna('Unknown')
            le = self.label_encoders[col]
            # 對於未見過的類別，使用第一個類別的編碼
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
        
        for col in self.numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)
        
        X_scaled = self.scaler.transform(df)
        return X_scaled
    
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

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
