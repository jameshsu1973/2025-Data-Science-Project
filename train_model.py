"""
Santander Product Recommendation - Model Training Script
This script trains a machine learning model to predict product recommendations.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = 'dataset/train_ver2.csv'
MODEL_SAVE_PATH = 'models/model.pkl'
PREPROCESSOR_SAVE_PATH = 'models/preprocessor.pkl'

# Feature columns (what user inputs in the form)
FEATURE_COLS = [
    'fecha_dato', 'ncodpers', 'ind_empleado', 'pais_residencia', 'sexo',
    'age', 'fecha_alta', 'ind_nuevo', 'antiguedad', 'indrel', 'ult_fec_cli_1t',
    'indrel_1mes', 'tiprel_1mes', 'indresi', 'indext', 'conyuemp',
    'canal_entrada', 'indfall', 'tipodom', 'cod_prov', 'nomprov',
    'ind_actividad_cliente', 'renta', 'segmento'
]

# Target columns (products to predict)
TARGET_COLS = [
    'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
    'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
    'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
    'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
    'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
    'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
    'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
    'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1'
]


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


def load_and_prepare_data(sample_size=10000):
    """Load and prepare the dataset"""
    print("Loading data...")
    
    # Load data (use a sample for faster training)
    df = pd.read_csv(DATA_PATH, nrows=sample_size)
    
    print(f"Data loaded: {df.shape}")
    
    # Separate features and targets
    X = df[FEATURE_COLS]
    y = df[TARGET_COLS]
    
    # Fill missing target values with 0
    y = y.fillna(0).astype(int)
    
    return X, y


def train_model(X, y):
    """Train the prediction model"""
    print("\nPreprocessing data...")
    
    # Initialize and fit preprocessor
    preprocessor = SantanderPreprocessor()
    X_processed = preprocessor.fit_transform(X)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train model
    print("\nTraining model...")
    # Using a simple Random Forest for demonstration
    # You can replace this with more sophisticated models
    base_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Multi-output classification
    model = MultiOutputClassifier(base_model)
    model.fit(X_train, y_train)
    
    # Evaluate
    score = model.score(X_test, y_test)
    print(f"\nModel accuracy on test set: {score:.4f}")
    
    return model, preprocessor


def save_model(model, preprocessor):
    """Save the trained model and preprocessor"""
    print("\nSaving model...")
    
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    with open(PREPROCESSOR_SAVE_PATH, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Preprocessor saved to: {PREPROCESSOR_SAVE_PATH}")


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("Santander Product Recommendation - Model Training")
    print("=" * 60)
    
    # Load data
    X, y = load_and_prepare_data(sample_size=10000)  # Adjust sample_size as needed
    
    # Train model
    model, preprocessor = train_model(X, y)
    
    # Save model
    save_model(model, preprocessor)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print("\nYou can now run the web application:")
    print("  python web/app.py")
    print("\nThe model will be automatically loaded and used for predictions.")


if __name__ == '__main__':
    main()
