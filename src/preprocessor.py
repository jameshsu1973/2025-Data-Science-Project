from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

class SantanderPreprocessor:
    """Custom preprocessor for Santander data"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_cols = []
        self.numerical_cols = []
        self.numerical_medians = {}  # 儲存每個數值欄位的中位數
        
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
            le.fit(list(le.classes_) + ['Unknown'])  # Ensure 'Unknown' is included
            self.label_encoders[col] = le
        
        # Prepare numerical data for scaling
        df_encoded = df.copy()
        for col in self.categorical_cols:
            df_encoded[col] = self.label_encoders[col].transform(df[col])
        
        # Fill missing values in numerical columns and save medians
        for col in self.numerical_cols:
            median_val = df_encoded[col].median()
            self.numerical_medians[col] = median_val  # 儲存中位數
            df_encoded[col] = df_encoded[col].fillna(median_val)
        
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
        
        # Handle numerical columns - 使用訓練時保存的中位數
        for col in self.numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # 使用訓練時計算的中位數，而非填 0
                median_val = self.numerical_medians.get(col, 0)
                df[col] = df[col].fillna(median_val)
        
        # Scale the data
        X_scaled = self.scaler.transform(df)
        
        return X_scaled
    
    def fit_transform(self, df):
        """Fit and transform in one step"""
        self.fit(df)
        return self.transform(df)