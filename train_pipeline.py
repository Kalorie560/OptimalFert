"""
Complete training pipeline for Playground Series S5E6
Orchestrates data loading, preprocessing, EDA, model training, and submission generation
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import yaml
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


def setup_clearml():
    """Initialize ClearML using config.yaml if available"""
    config_path = "config.yaml"
    
    if not os.path.exists(config_path):
        logger.warning("config.yaml not found. ClearML tracking will be disabled.")
        logger.info("To enable ClearML tracking:")
        logger.info("1. Copy config.yaml.template to config.yaml")
        logger.info("2. Add your ClearML API credentials")
        return False
    
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        clearml_config = config.get('clearml', {})
        project_config = config.get('project', {})
        
        # Validate required fields
        required_fields = ['api_host', 'web_host', 'files_host', 'api_key', 'api_secret_key']
        placeholder_patterns = ['YOUR_API_KEY_HERE', 'YOUR_SECRET_KEY_HERE', 'YOUR_API_HOST_HERE', 'YOUR_WEB_HOST_HERE', 'YOUR_FILES_HOST_HERE']
        missing_fields = [field for field in required_fields if not clearml_config.get(field) or any(placeholder in str(clearml_config.get(field, '')) for placeholder in placeholder_patterns)]
        
        if missing_fields:
            logger.warning(f"ClearML configuration incomplete. Missing or placeholder values: {missing_fields}")
            logger.info("Please update config.yaml with your actual ClearML credentials")
            return False
        
        # Initialize ClearML
        try:
            from clearml import Task
            
            # Create Task with project configuration
            task = Task.init(
                project_name=project_config.get('name', 'Playground-Series-S5E6'),
                task_name=f"Training Pipeline - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
                auto_connect_frameworks=True
            )
            
            # Set task description
            if project_config.get('description'):
                task.set_description(project_config['description'])
            
            # Connect configuration
            task.connect(config)
            
            logger.info("✅ ClearML experiment tracking initialized successfully")
            logger.info(f"Project: {project_config.get('name', 'Playground-Series-S5E6')}")
            logger.info(f"Task URL: {task.get_output_log_web_page()}")
            
            return True
            
        except ImportError:
            logger.warning("ClearML package not installed. Install with: pip install clearml")
            return False
        except Exception as e:
            logger.warning(f"Failed to initialize ClearML: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to load config.yaml: {e}")
        return False


def setup_directories():
    """Create necessary directories"""
    directories = [
        'data', 'models', 'outputs', 'logs', 'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory '{directory}' ready")


def load_config():
    """Load configuration from config.yaml"""
    config_path = "config.yaml"
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config.yaml: {e}")
    return {}


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
    
    # Load configuration
    config = load_config()
    
    # Initialize ClearML experiment tracking
    logger.info("\n" + "="*40)
    logger.info("STEP 0: EXPERIMENT TRACKING SETUP")
    logger.info("="*40)
    clearml_enabled = setup_clearml()
    
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
        # Initialize trainer (it will load configuration from config.yaml internally)
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
        'pandas', 'numpy', 'sklearn', 'lightgbm', 'xgboost', 
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
        logger.error("")
        logger.error("🔧 SOLUTION:")
        logger.error("Install all dependencies from requirements.txt:")
        logger.error("pip install -r requirements.txt")
        logger.error("")
        logger.error("Or install missing packages individually:")
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