"""
Script to launch the Streamlit web application
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Launch the Streamlit web application"""
    
    # Path to the Streamlit app
    app_path = os.path.join("src", "web_app", "streamlit_app.py")
    
    # Check if the app file exists
    if not os.path.exists(app_path):
        logger.error(f"Streamlit app not found at {app_path}")
        sys.exit(1)
    
    # Check if required model files exist
    model_path = os.path.join("models", "best_model.pkl")
    preprocessor_path = os.path.join("models", "preprocessor.pkl")
    
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found at {model_path}")
        logger.info("Train a model first using: python train_pipeline.py")
    
    if not os.path.exists(preprocessor_path):
        logger.warning(f"Preprocessor file not found at {preprocessor_path}")
        logger.info("Train a model first using: python train_pipeline.py")
    
    # Launch Streamlit
    logger.info("Launching Streamlit web application...")
    logger.info("Access the app at: http://localhost:8501")
    
    try:
        subprocess.run(["streamlit", "run", app_path], check=True)
    except FileNotFoundError:
        logger.error("Streamlit not found. Install with: pip install streamlit")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running Streamlit: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()