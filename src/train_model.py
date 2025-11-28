"""
Santander Product Recommendation - Model Training Script
This script trains a machine learning model to predict product recommendations.
"""
import os
from preprocessor import SantanderPreprocessor
from utils.const import FEATURE_COLS, TARGET_COLS

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import warnings

warnings.filterwarnings('ignore')

# Configuration
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(REPO_PATH, 'dataset/train_ver2.csv')
MODEL_SAVE_PATH = os.path.join(REPO_PATH, 'models/model.pkl')
PREPROCESSOR_SAVE_PATH = os.path.join(REPO_PATH, 'models/preprocessor.pkl') 

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

    with open(PREPROCESSOR_SAVE_PATH, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print("\nYou can now run the web application:")
    print("  python web/app.py")
    print("\nThe model will be automatically loaded and used for predictions.")


if __name__ == '__main__':
    main()
