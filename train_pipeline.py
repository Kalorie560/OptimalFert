"""
Complete training pipeline for Playground Series S5E6
Orchestrates data loading, preprocessing, EDA, model training, and submission generation
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.preprocessing import DataPreprocessor, create_sample_data
from src.visualization.eda import run_eda
from src.models.train_model import ModelTrainer
from src.models.predict import generate_submission

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories"""
    directories = [
        'data', 'models', 'outputs', 'logs', 'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory '{directory}' ready")


def run_complete_pipeline(use_sample_data: bool = True):
    """
    Run the complete ML pipeline
    
    Args:
        use_sample_data: If True, creates and uses sample data for demonstration
    """
    logger.info("="*60)
    logger.info("STARTING PLAYGROUND SERIES S5E6 TRAINING PIPELINE")
    logger.info("="*60)
    
    # Setup
    setup_directories()
    
    # Step 1: Data Preparation
    logger.info("\n" + "="*40)
    logger.info("STEP 1: DATA PREPARATION")
    logger.info("="*40)
    
    if use_sample_data:
        logger.info("Creating sample data for demonstration...")
        create_sample_data(n_samples=2000, n_features=15, save_path="data/")
        train_path = "data/train.csv"
        test_path = "data/test.csv"
    else:
        train_path = "data/train.csv"
        test_path = "data/test.csv"
        
        if not os.path.exists(train_path):
            logger.error(f"Training data not found at {train_path}")
            logger.info("Please place your competition data files in the 'data/' directory")
            logger.info("Expected files: train.csv, test.csv, sample_submission.csv")
            return False
    
    # Step 2: Exploratory Data Analysis
    logger.info("\n" + "="*40)
    logger.info("STEP 2: EXPLORATORY DATA ANALYSIS")
    logger.info("="*40)
    
    try:
        eda_results = run_eda(train_path)
        logger.info("EDA completed successfully")
    except Exception as e:
        logger.error(f"EDA failed: {e}")
        return False
    
    # Step 3: Data Preprocessing
    logger.info("\n" + "="*40)
    logger.info("STEP 3: DATA PREPROCESSING")
    logger.info("="*40)
    
    try:
        # Load training data
        train_df = pd.read_csv(train_path)
        logger.info(f"Loaded training data: {train_df.shape}")
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(target_column='target')
        
        # Fit and transform training data
        X_train, y_train = preprocessor.fit_transform(train_df)
        
        # Save preprocessor
        preprocessor.save_preprocessor("models/preprocessor.pkl")
        
        # Save processed data
        np.savez("data/processed_train.npz", X=X_train, y=y_train)
        
        logger.info(f"Training data preprocessed: {X_train.shape}")
        logger.info(f"Target distribution: {np.bincount(y_train)}")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return False
    
    # Step 4: Model Training
    logger.info("\n" + "="*40)
    logger.info("STEP 4: MODEL TRAINING")
    logger.info("="*40)
    
    try:
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Train base models
        logger.info("Training base models...")
        base_results = trainer.train_base_models(X_train, y_train)
        
        # Get best performing base model for optimization
        best_base_model = max(
            [name for name, result in base_results.items() if result['fitted']],
            key=lambda name: base_results[name]['cv_results']['mean_score']
        )
        
        logger.info(f"Best base model: {best_base_model}")
        
        # Hyperparameter optimization
        logger.info("Starting hyperparameter optimization...")
        optimization_results = trainer.optimize_hyperparameters(X_train, y_train, best_base_model)
        
        # Create ensemble
        logger.info("Creating ensemble model...")
        ensemble_results = trainer.create_ensemble_model(X_train, y_train)
        
        # Save best model
        trainer.save_best_model("models/best_model.pkl")
        
        logger.info(f"Training completed! Best CV AUC: {trainer.best_score:.4f}")
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return False
    
    # Step 5: Test Prediction and Submission
    logger.info("\n" + "="*40)
    logger.info("STEP 5: PREDICTION AND SUBMISSION")
    logger.info("="*40)
    
    try:
        # Generate submission
        submission_df = generate_submission(
            model_path="models/best_model.pkl",
            preprocessor_path="models/preprocessor.pkl",
            test_data_path=test_path,
            sample_submission_path="data/sample_submission.csv",
            output_path="submission.csv"
        )
        
        logger.info("Submission file generated successfully!")
        logger.info(f"Submission shape: {submission_df.shape}")
        
    except Exception as e:
        logger.error(f"Submission generation failed: {e}")
        return False
    
    # Step 6: Summary
    logger.info("\n" + "="*40)
    logger.info("PIPELINE COMPLETION SUMMARY")
    logger.info("="*40)
    
    logger.info("✅ Pipeline completed successfully!")
    logger.info("\nGenerated files:")
    logger.info("- models/best_model.pkl (trained model)")
    logger.info("- models/preprocessor.pkl (data preprocessor)")
    logger.info("- submission.csv (competition submission)")
    logger.info("- outputs/ (EDA visualizations)")
    logger.info("- training.log (detailed logs)")
    
    logger.info(f"\n🎯 Best model CV AUC: {trainer.best_score:.4f}")
    
    logger.info("\nNext steps:")
    logger.info("1. Review EDA results in outputs/ directory")
    logger.info("2. Submit submission.csv to Kaggle")
    logger.info("3. Run web app: streamlit run src/web_app/streamlit_app.py")
    
    return True


def check_environment():
    """Check if all required packages are available"""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'lightgbm', 'xgboost', 
        'catboost', 'optuna', 'matplotlib', 'seaborn', 'plotly',
        'streamlit', 'joblib', 'scipy', 'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error("Missing required packages:")
        for package in missing_packages:
            logger.error(f"  - {package}")
        logger.error("\nInstall missing packages with:")
        logger.error("pip install " + " ".join(missing_packages))
        return False
    
    logger.info("All required packages are available ✅")
    return True


if __name__ == "__main__":
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Run pipeline
    success = run_complete_pipeline(use_sample_data=True)
    
    if success:
        logger.info("\n🎉 Training pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("\n❌ Training pipeline failed!")
        sys.exit(1)