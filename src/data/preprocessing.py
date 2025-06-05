"""
Data preprocessing pipeline for Playground Series S5E6
Handles feature engineering, encoding, and scaling
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
    Comprehensive data preprocessing pipeline for the competition
    """
    
    def __init__(self, target_column: str = 'target'):
        self.target_column = target_column
        self.numeric_features = []
        self.categorical_features = []
        self.preprocessor = None
        self.feature_names = []
        
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
            Tuple of (X_processed, y)
        """
        # Identify feature types
        self.identify_feature_types(df)
        
        # Create preprocessing pipeline
        self.create_preprocessing_pipeline()
        
        # Separate features and target
        X = df.drop(columns=['id', self.target_column] if 'id' in df.columns 
                   else [self.target_column])
        y = df[self.target_column].values
        
        # Fit and transform
        X_processed = self.preprocessor.fit_transform(X)
        
        # Store feature names for later use
        self._store_feature_names()
        
        logger.info(f"Training data processed: {X_processed.shape}")
        return X_processed, y
    
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
    
    def save_preprocessor(self, filepath: str) -> None:
        """
        Save the fitted preprocessor to disk
        """
        joblib.dump({
            'preprocessor': self.preprocessor,
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
        self.numeric_features = data['numeric_features']
        self.categorical_features = data['categorical_features']
        self.feature_names = data['feature_names']
        self.target_column = data['target_column']
        logger.info(f"Preprocessor loaded from {filepath}")


def create_sample_data(n_samples: int = 1000, n_features: int = 10, 
                      save_path: str = "data/") -> None:
    """
    Create sample data for development and testing
    This simulates a typical Playground Series dataset
    """
    np.random.seed(42)
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Generate features
    data = {}
    data['id'] = range(n_samples)
    
    # Numeric features
    for i in range(n_features):
        if i < 3:  # Some features with normal distribution
            data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
        elif i < 6:  # Some features with uniform distribution
            data[f'feature_{i}'] = np.random.uniform(-2, 2, n_samples)
        else:  # Some categorical features
            data[f'feature_{i}'] = np.random.choice(['A', 'B', 'C', 'D'], n_samples)
    
    # Create target with some correlation to features
    target_prob = (
        0.3 * data['feature_0'] + 
        0.2 * data['feature_1'] + 
        0.1 * data['feature_2'] +
        np.random.normal(0, 0.5, n_samples)
    )
    target_prob = 1 / (1 + np.exp(-target_prob))  # Sigmoid
    data['target'] = np.random.binomial(1, target_prob)
    
    # Create train and test splits
    train_size = int(0.8 * n_samples)
    
    train_data = pd.DataFrame({k: v[:train_size] for k, v in data.items()})
    test_data = pd.DataFrame({k: v[train_size:] for k, v in data.items() if k != 'target'})
    
    # Save files
    train_data.to_csv(os.path.join(save_path, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(save_path, 'test.csv'), index=False)
    
    # Create sample submission
    sample_submission = pd.DataFrame({
        'id': test_data['id'],
        'target': np.random.uniform(0, 1, len(test_data))
    })
    sample_submission.to_csv(os.path.join(save_path, 'sample_submission.csv'), index=False)
    
    logger.info(f"Sample data created in {save_path}")
    logger.info(f"Train shape: {train_data.shape}")
    logger.info(f"Test shape: {test_data.shape}")


if __name__ == "__main__":
    # Create sample data for development
    create_sample_data()