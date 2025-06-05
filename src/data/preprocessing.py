"""
Data preprocessing pipeline for fertilizer name prediction
Handles feature engineering, encoding, and scaling for multi-class classification
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for fertilizer name prediction
    Handles multi-class classification with string target labels
    """
    
    def __init__(self, target_column: str = 'Fertilizer Name'):
        self.target_column = target_column
        self.numeric_features = []
        self.categorical_features = []
        self.preprocessor = None
        self.feature_names = []
        self.label_encoder = LabelEncoder()
        self.target_classes = []
        
    def identify_feature_types(self, df: pd.DataFrame) -> None:
        """
        Automatically identify numeric and categorical features
        """
        # Exclude ID and target columns
        feature_columns = [col for col in df.columns 
                          if col not in ['id', self.target_column]]
        
        self.numeric_features = []
        self.categorical_features = []
        
        for col in feature_columns:
            if df[col].dtype in ['int64', 'float64']:
                # Check if it's actually categorical (small number of unique values)
                if df[col].nunique() <= 10 and df[col].dtype == 'int64':
                    self.categorical_features.append(col)
                else:
                    self.numeric_features.append(col)
            else:
                self.categorical_features.append(col)
        
        logger.info(f"Identified {len(self.numeric_features)} numeric features")
        logger.info(f"Identified {len(self.categorical_features)} categorical features")
    
    def create_preprocessing_pipeline(self) -> None:
        """
        Create sklearn preprocessing pipeline
        """
        # Numeric features pipeline
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical features pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine pipelines
        self.preprocessor = ColumnTransformer([
            ('numeric', numeric_pipeline, self.numeric_features),
            ('categorical', categorical_pipeline, self.categorical_features)
        ])
        
        logger.info("Preprocessing pipeline created successfully")
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the preprocessing pipeline and transform the data
        
        Args:
            df: Training dataframe
            
        Returns:
            Tuple of (X_processed, y_encoded)
        """
        # Identify feature types
        self.identify_feature_types(df)
        
        # Create preprocessing pipeline
        self.create_preprocessing_pipeline()
        
        # Separate features and target
        X = df.drop(columns=['id', self.target_column] if 'id' in df.columns 
                   else [self.target_column])
        y_str = df[self.target_column].values
        
        # Encode target labels to integers
        y_encoded = self.label_encoder.fit_transform(y_str)
        self.target_classes = self.label_encoder.classes_
        
        # Fit and transform features
        X_processed = self.preprocessor.fit_transform(X)
        
        # Store feature names for later use
        self._store_feature_names()
        
        logger.info(f"Training data processed: {X_processed.shape}")
        logger.info(f"Target classes: {len(self.target_classes)} unique fertilizers")
        logger.info(f"Fertilizer types: {list(self.target_classes)}")
        return X_processed, y_encoded
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor
        
        Args:
            df: Dataframe to transform
            
        Returns:
            Processed feature matrix
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        # Remove ID column if present, but keep other columns for prediction
        columns_to_drop = ['id'] if 'id' in df.columns else []
        if self.target_column in df.columns:
            columns_to_drop.append(self.target_column)
            
        X = df.drop(columns=columns_to_drop)
        X_processed = self.preprocessor.transform(X)
        
        logger.info(f"Test data processed: {X_processed.shape}")
        return X_processed
    
    def _store_feature_names(self) -> None:
        """
        Store feature names after preprocessing for interpretability
        """
        feature_names = []
        
        # Numeric feature names
        feature_names.extend(self.numeric_features)
        
        # Categorical feature names (after one-hot encoding)
        if self.categorical_features:
            cat_feature_names = self.preprocessor.named_transformers_['categorical'][
                'onehot'].get_feature_names_out(self.categorical_features)
            feature_names.extend(cat_feature_names)
        
        self.feature_names = feature_names
        logger.info(f"Stored {len(self.feature_names)} feature names")
    
    def decode_predictions(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Convert encoded predictions back to fertilizer names
        
        Args:
            y_encoded: Encoded predictions (integers)
            
        Returns:
            Array of fertilizer names (strings)
        """
        return self.label_encoder.inverse_transform(y_encoded)
    
    def save_preprocessor(self, filepath: str) -> None:
        """
        Save the fitted preprocessor to disk
        """
        joblib.dump({
            'preprocessor': self.preprocessor,
            'label_encoder': self.label_encoder,
            'target_classes': self.target_classes,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'feature_names': self.feature_names,
            'target_column': self.target_column
        }, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str) -> None:
        """
        Load a fitted preprocessor from disk
        """
        data = joblib.load(filepath)
        self.preprocessor = data['preprocessor']
        self.label_encoder = data['label_encoder']
        self.target_classes = data['target_classes']
        self.numeric_features = data['numeric_features']
        self.categorical_features = data['categorical_features']
        self.feature_names = data['feature_names']
        self.target_column = data['target_column']
        logger.info(f"Preprocessor loaded from {filepath}")


def create_sample_data(n_samples: int = 1000, save_path: str = "data/") -> None:
    """
    Create sample fertilizer prediction data for development and testing
    This simulates agricultural data with fertilizer recommendations
    """
    np.random.seed(42)
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Define fertilizer types
    fertilizer_types = [
        'NPK 10-10-10', 'NPK 15-15-15', 'NPK 20-20-20',
        'Urea (46-0-0)', 'DAP (18-46-0)', 'MOP (0-0-60)',
        'Organic Compost', 'Bone Meal', 'Fish Emulsion',
        'Calcium Nitrate', 'Magnesium Sulfate', 'Potassium Sulfate'
    ]
    
    # Generate features relevant to fertilizer recommendation
    data = {}
    data['id'] = range(n_samples)
    
    # Soil properties
    data['Soil_pH'] = np.random.normal(6.5, 1.0, n_samples)
    data['Nitrogen_ppm'] = np.random.lognormal(3, 0.5, n_samples)
    data['Phosphorus_ppm'] = np.random.lognormal(2.5, 0.6, n_samples)
    data['Potassium_ppm'] = np.random.lognormal(4, 0.4, n_samples)
    data['Organic_Matter_percent'] = np.random.gamma(2, 1.5, n_samples)
    
    # Environmental conditions
    data['Temperature_C'] = np.random.normal(22, 8, n_samples)
    data['Rainfall_mm'] = np.random.exponential(50, n_samples)
    data['Humidity_percent'] = np.random.normal(65, 15, n_samples)
    
    # Crop and management factors
    data['Crop_Type'] = np.random.choice(['Wheat', 'Corn', 'Rice', 'Soybean', 'Tomato', 'Potato'], n_samples)
    data['Growth_Stage'] = np.random.choice(['Seedling', 'Vegetative', 'Flowering', 'Fruiting'], n_samples)
    data['Field_Size_acres'] = np.random.lognormal(1, 0.8, n_samples)
    data['Previous_Fertilizer'] = np.random.choice(['None', 'Organic', 'Chemical', 'Mixed'], n_samples)
    
    # Create fertilizer recommendations based on features
    fertilizer_recommendations = []
    for i in range(n_samples):
        # Simple rule-based logic for fertilizer recommendation
        ph = data['Soil_pH'][i]
        n = data['Nitrogen_ppm'][i]
        p = data['Phosphorus_ppm'][i]
        k = data['Potassium_ppm'][i]
        crop = data['Crop_Type'][i]
        stage = data['Growth_Stage'][i]
        
        if n < 20:  # Low nitrogen
            if stage == 'Vegetative':
                fertilizer = 'Urea (46-0-0)'
            else:
                fertilizer = 'NPK 20-20-20'
        elif p < 15:  # Low phosphorus
            fertilizer = 'DAP (18-46-0)'
        elif k < 50:  # Low potassium
            fertilizer = 'MOP (0-0-60)'
        elif crop in ['Tomato', 'Potato'] and stage == 'Flowering':
            fertilizer = 'Calcium Nitrate'
        elif ph < 6.0:  # Acidic soil
            fertilizer = 'Organic Compost'
        elif data['Previous_Fertilizer'][i] == 'None':
            fertilizer = 'NPK 15-15-15'
        else:
            # Random selection for remaining cases
            fertilizer = np.random.choice(fertilizer_types)
        
        fertilizer_recommendations.append(fertilizer)
    
    data['Fertilizer Name'] = fertilizer_recommendations
    
    # Create train and test splits
    train_size = int(0.8 * n_samples)
    
    train_data = pd.DataFrame({k: v[:train_size] for k, v in data.items()})
    test_data = pd.DataFrame({k: v[train_size:] for k, v in data.items() if k != 'Fertilizer Name'})
    
    # Save files
    train_data.to_csv(os.path.join(save_path, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(save_path, 'test.csv'), index=False)
    
    # Create sample submission with fertilizer names
    sample_submission = pd.DataFrame({
        'id': test_data['id'],
        'Fertilizer Name': np.random.choice(fertilizer_types, len(test_data))
    })
    sample_submission.to_csv(os.path.join(save_path, 'sample_submission.csv'), index=False)
    
    logger.info(f"Sample fertilizer data created in {save_path}")
    logger.info(f"Train shape: {train_data.shape}")
    logger.info(f"Test shape: {test_data.shape}")
    logger.info(f"Unique fertilizers: {len(fertilizer_types)}")
    logger.info(f"Fertilizer distribution in training data:")
    logger.info(train_data['Fertilizer Name'].value_counts())


if __name__ == "__main__":
    # Create sample data for development
    create_sample_data()