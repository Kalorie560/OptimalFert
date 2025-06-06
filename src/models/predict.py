"""
Prediction and submission generation module for fertilizer name prediction
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FertilizerPredictor:
    """
    Handles model loading and prediction generation for fertilizer name prediction
    """
    
    def __init__(self, model_path: str, preprocessor_path: str):
        self.model = None
        self.preprocessor = None
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        
        self.load_model_and_preprocessor()
    
    def load_model_and_preprocessor(self):
        """Load trained model and preprocessor"""
        try:
            # Load model
            model_data = joblib.load(self.model_path)
            if isinstance(model_data, dict):
                self.model = model_data['model']
                logger.info(f"Model loaded with score: {model_data.get('score', 'N/A')}")
            else:
                self.model = model_data
            
            logger.info(f"Model loaded from {self.model_path}")
            
            # Load preprocessor
            preprocessor_data = joblib.load(self.preprocessor_path)
            if isinstance(preprocessor_data, dict):
                # Custom preprocessor object
                from src.data.preprocessing import DataPreprocessor
                self.preprocessor = DataPreprocessor()
                self.preprocessor.preprocessor = preprocessor_data['preprocessor']
                self.preprocessor.label_encoder = preprocessor_data['label_encoder']
                self.preprocessor.target_classes = preprocessor_data['target_classes']
                self.preprocessor.numeric_features = preprocessor_data['numeric_features']
                self.preprocessor.categorical_features = preprocessor_data['categorical_features']
                self.preprocessor.feature_names = preprocessor_data['feature_names']
                self.preprocessor.target_column = preprocessor_data['target_column']
            else:
                self.preprocessor = preprocessor_data
            
            logger.info(f"Preprocessor loaded from {self.preprocessor_path}")
            
        except Exception as e:
            logger.error(f"Error loading model or preprocessor: {e}")
            raise
    
    def predict_test_data(self, test_data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate fertilizer name predictions for test data
        
        Args:
            test_data_path: Path to test CSV file
            
        Returns:
            Tuple of (test_ids, fertilizer_names)
        """
        # Load test data
        test_df = pd.read_csv(test_data_path)
        logger.info(f"Loaded test data with shape: {test_df.shape}")
        
        # Extract IDs
        test_ids = test_df['id'].values
        
        # Preprocess test data
        X_test = self.preprocessor.transform(test_df)
        logger.info(f"Test data preprocessed to shape: {X_test.shape}")
        
        # Generate predictions (class labels)
        predicted_labels = self.model.predict(X_test)
        
        # Convert predictions back to fertilizer names
        fertilizer_names = self.preprocessor.decode_predictions(predicted_labels)
        
        logger.info(f"Generated {len(fertilizer_names)} fertilizer predictions")
        logger.info(f"Unique fertilizers predicted: {len(np.unique(fertilizer_names))}")
        logger.info(f"Fertilizer prediction counts:")
        unique, counts = np.unique(fertilizer_names, return_counts=True)
        for fert, count in zip(unique, counts):
            logger.info(f"  {fert}: {count}")
        
        return test_ids, fertilizer_names
    
    def create_submission_file(self, test_data_path: str, 
                             submission_path: str = "submission.csv",
                             sample_submission_path: Optional[str] = None) -> pd.DataFrame:
        """
        Create submission file in correct format
        
        Args:
            test_data_path: Path to test CSV file
            submission_path: Path where to save submission file
            sample_submission_path: Path to sample submission file for format validation
            
        Returns:
            Submission dataframe
        """
        # Generate predictions
        test_ids, fertilizer_names = self.predict_test_data(test_data_path)
        
        # Create submission dataframe
        submission_df = pd.DataFrame({
            'id': test_ids,
            'Fertilizer Name': fertilizer_names
        })
        
        # Validate format if sample submission is provided
        if sample_submission_path and os.path.exists(sample_submission_path):
            sample_df = pd.read_csv(sample_submission_path)
            
            # Check column names
            if list(submission_df.columns) != list(sample_df.columns):
                logger.warning(f"Column mismatch! Expected: {list(sample_df.columns)}, "
                             f"Got: {list(submission_df.columns)}")
            
            # Check ID order
            if not submission_df['id'].equals(sample_df['id']):
                logger.warning("ID order doesn't match sample submission. Reordering...")
                submission_df = submission_df.set_index('id').loc[sample_df['id']].reset_index()
            
            # Check shape
            if submission_df.shape != sample_df.shape:
                logger.warning(f"Shape mismatch! Expected: {sample_df.shape}, "
                             f"Got: {submission_df.shape}")
            
            logger.info("Submission format validated against sample submission")
        
        # Validate fertilizer names are from known set
        known_fertilizers = set(self.preprocessor.target_classes)
        predicted_fertilizers = set(submission_df['Fertilizer Name'])
        unknown_fertilizers = predicted_fertilizers - known_fertilizers
        if unknown_fertilizers:
            logger.warning(f"Unknown fertilizers in predictions: {unknown_fertilizers}")
            # Replace unknown fertilizers with most common one
            most_common_fertilizer = submission_df['Fertilizer Name'].mode()[0]
            submission_df.loc[
                submission_df['Fertilizer Name'].isin(unknown_fertilizers), 
                'Fertilizer Name'
            ] = most_common_fertilizer
        
        # Save submission
        os.makedirs(os.path.dirname(submission_path) if os.path.dirname(submission_path) else '.', 
                   exist_ok=True)
        submission_df.to_csv(submission_path, index=False)
        
        logger.info(f"Submission file saved to {submission_path}")
        logger.info(f"Submission shape: {submission_df.shape}")
        logger.info(f"Fertilizer prediction distribution:")
        fertilizer_counts = submission_df['Fertilizer Name'].value_counts()
        for fertilizer, count in fertilizer_counts.items():
            logger.info(f"  {fertilizer}: {count} ({count/len(submission_df)*100:.1f}%)")
        
        return submission_df
    
    def predict_single_sample(self, feature_dict: dict) -> Tuple[str, dict]:
        """
        Predict fertilizer for a single sample (for web app)
        
        Args:
            feature_dict: Dictionary with feature names as keys and values
            
        Returns:
            Tuple of (predicted_fertilizer_name, probability_dict)
        """
        # Create dataframe from feature dict
        sample_df = pd.DataFrame([feature_dict])
        
        # Add dummy ID if not present
        if 'id' not in sample_df.columns:
            sample_df['id'] = 0
        
        # Preprocess
        X_sample = self.preprocessor.transform(sample_df)
        
        # Predict class probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_sample)[0]
            predicted_class = np.argmax(probabilities)
        else:
            predicted_class = self.model.predict(X_sample)[0]
            # Create dummy probabilities
            n_classes = len(self.preprocessor.target_classes)
            probabilities = np.zeros(n_classes)
            probabilities[predicted_class] = 1.0
        
        # Convert to fertilizer name
        fertilizer_name = self.preprocessor.decode_predictions([predicted_class])[0]
        
        # Create probability dictionary
        prob_dict = {
            fertilizer: float(prob) 
            for fertilizer, prob in zip(self.preprocessor.target_classes, probabilities)
        }
        
        return fertilizer_name, prob_dict


def generate_submission(model_path: str = "models/best_model.pkl",
                       preprocessor_path: str = "models/preprocessor.pkl",
                       test_data_path: str = "data/test.csv",
                       sample_submission_path: str = "data/sample_submission.csv",
                       output_path: str = "submission.csv") -> pd.DataFrame:
    """
    Generate fertilizer prediction submission file
    
    Args:
        model_path: Path to trained model
        preprocessor_path: Path to fitted preprocessor
        test_data_path: Path to test data
        sample_submission_path: Path to sample submission file
        output_path: Path for output submission file
        
    Returns:
        Submission dataframe
    """
    logger.info("Starting fertilizer prediction submission generation...")
    
    # Initialize predictor
    predictor = FertilizerPredictor(model_path, preprocessor_path)
    
    # Create submission
    submission_df = predictor.create_submission_file(
        test_data_path=test_data_path,
        submission_path=output_path,
        sample_submission_path=sample_submission_path
    )
    
    logger.info("Submission generation completed successfully!")
    
    return submission_df


def validate_submission(submission_path: str, 
                       sample_submission_path: str) -> bool:
    """
    Validate submission file format
    
    Args:
        submission_path: Path to submission file
        sample_submission_path: Path to sample submission file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        submission_df = pd.read_csv(submission_path)
        sample_df = pd.read_csv(sample_submission_path)
        
        # Check shape
        if submission_df.shape != sample_df.shape:
            logger.error(f"Shape mismatch: expected {sample_df.shape}, got {submission_df.shape}")
            return False
        
        # Check columns
        if list(submission_df.columns) != list(sample_df.columns):
            logger.error(f"Column mismatch: expected {list(sample_df.columns)}, "
                        f"got {list(submission_df.columns)}")
            return False
        
        # Check IDs
        if not submission_df['id'].equals(sample_df['id']):
            logger.error("ID mismatch with sample submission")
            return False
        
        # Check target values based on column type
        target_col = submission_df.columns[1]  # Assuming second column is target
        
        if target_col == 'Fertilizer Name':
            # For fertilizer names, check if all values are strings
            if not submission_df[target_col].dtype == object:
                logger.error("Fertilizer Name column should contain string values")
                return False
            # Check for empty values
            if submission_df[target_col].str.strip().eq('').any():
                logger.error("Fertilizer Name column contains empty values")
                return False
        else:
            # For numeric targets
            if not pd.api.types.is_numeric_dtype(submission_df[target_col]):
                logger.error("Target column is not numeric")
                return False
            if submission_df[target_col].min() < 0 or submission_df[target_col].max() > 1:
                logger.warning("Target values outside [0, 1] range - may be valid depending on competition")
        
        # Check for missing values
        if submission_df.isnull().any().any():
            logger.error("Submission contains missing values")
            return False
        
        logger.info("Submission validation passed!")
        return True
        
    except Exception as e:
        logger.error(f"Error validating submission: {e}")
        return False


if __name__ == "__main__":
    # Generate submission
    submission = generate_submission()
    
    # Validate submission
    if os.path.exists("data/sample_submission.csv"):
        validate_submission("submission.csv", "data/sample_submission.csv")