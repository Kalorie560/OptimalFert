"""
Generate submission file for Playground Series S5E6
"""

import os
import sys
import logging
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.predict import generate_submission, validate_submission

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Generate submission for Playground Series S5E6')
    parser.add_argument('--model', default='models/best_model.pkl', 
                       help='Path to trained model file')
    parser.add_argument('--preprocessor', default='models/preprocessor.pkl',
                       help='Path to preprocessor file')
    parser.add_argument('--test_data', default='data/test.csv',
                       help='Path to test data file')
    parser.add_argument('--sample_submission', default='data/sample_submission.csv',
                       help='Path to sample submission file')
    parser.add_argument('--output', default='submission.csv',
                       help='Output submission file path')
    parser.add_argument('--validate', action='store_true',
                       help='Validate submission format')
    
    args = parser.parse_args()
    
    # Check if required files exist
    required_files = [args.model, args.preprocessor, args.test_data]
    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.error(f"Required file not found: {file_path}")
            return False
    
    try:
        # Generate submission
        logger.info("Generating submission file...")
        submission_df = generate_submission(
            model_path=args.model,
            preprocessor_path=args.preprocessor,
            test_data_path=args.test_data,
            sample_submission_path=args.sample_submission,
            output_path=args.output
        )
        
        logger.info(f"Submission generated: {args.output}")
        logger.info(f"Shape: {submission_df.shape}")
        
        # Validate if requested
        if args.validate and os.path.exists(args.sample_submission):
            logger.info("Validating submission format...")
            is_valid = validate_submission(args.output, args.sample_submission)
            if is_valid:
                logger.info("✅ Submission validation passed!")
            else:
                logger.error("❌ Submission validation failed!")
                return False
        
        logger.info("✅ Submission generation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error generating submission: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)