"""
Santander Product Recommendation - Model Training Script
This script trains a machine learning model to predict product recommendations.
"""
import os
from preprocessor import SantanderPreprocessor
from utils.const import FEATURE_COLS, TARGET_COLS

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import hamming_loss, f1_score, classification_report
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
    
    # Evaluate with multiple metrics
    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    y_pred = model.predict(X_test)
    
    # 1. Subset Accuracy (all 24 labels must be correct)
    score = model.score(X_test, y_test)
    print(f"\n1. Subset Accuracy: {score:.4f}")
    print("   (Strict: all 24 products must be predicted correctly)")
    
    # 2. Hamming Loss (average error per label)
    h_loss = hamming_loss(y_test, y_pred)
    print(f"\n2. Hamming Loss: {h_loss:.4f}")
    print(f"   (Average error rate per product: {h_loss*100:.2f}%)")
    
    # 3. F1 Score
    f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"\n3. F1 Score:")
    print(f"   - Micro F1: {f1_micro:.4f} (weighted by sample count)")
    print(f"   - Macro F1: {f1_macro:.4f} (average across all products)")
    
    # 4. Feature Importance
    print("\n" + "=" * 60)
    print("Feature Importance (Top 10)")
    print("=" * 60)
    
    # Average feature importance across all 24 product classifiers
    feature_importances = np.mean([
        estimator.feature_importances_ 
        for estimator in model.estimators_
    ], axis=0)
    
    # Create DataFrame for display
    importance_df = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    print("\n" + importance_df.head(10).to_string(index=False))
    
    # Save feature importance to CSV
    importance_path = os.path.join(REPO_PATH, 'models', 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"\nFull feature importance saved to: {importance_path}")
    
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
