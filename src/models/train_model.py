"""
Model training module for Playground Series S5E6
Implements multiple ML models with hyperparameter optimization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
import joblib
import yaml
import os
from typing import Dict, Any, Tuple, List
import logging

# ClearML integration
try:
    from clearml import Task, Logger
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    print("ClearML not available. Install with: pip install clearml")

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Comprehensive model training class with ClearML integration
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self.load_config(config_path)
        self.models = {}
        self.best_model = None
        self.best_score = 0.0
        self.clearml_task = None
        self.clearml_logger = None
        
        self.setup_clearml()
    
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default config.")
            return self.get_default_config()
    
    def get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            'project': {
                'name': 'Playground-Series-S5E6',
                'description': 'Binary classification with ROC AUC optimization'
            },
            'models': {
                'random_state': 42,
                'cv_folds': 5
            },
            'optimization': {
                'n_trials': 50,
                'study_name': 'playground_s5e6_optimization'
            }
        }
    
    def setup_clearml(self):
        """Setup ClearML task and logger"""
        if not CLEARML_AVAILABLE:
            logger.warning("ClearML not available. Proceeding without experiment tracking.")
            return
        
        try:
            # Initialize ClearML task
            self.clearml_task = Task.init(
                project_name=self.config.get('project', {}).get('name', 'Playground-Series-S5E6'),
                task_name=f"Model_Training_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
                task_type=Task.TaskTypes.training
            )
            
            # Connect configuration
            self.clearml_task.connect(self.config)
            
            # Setup logger
            self.clearml_logger = Logger.current_logger()
            
            logger.info("ClearML task initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize ClearML: {e}")
            self.clearml_task = None
            self.clearml_logger = None
    
    def get_base_models(self) -> Dict[str, Any]:
        """Get base models for training"""
        random_state = self.config.get('models', {}).get('random_state', 42)
        
        models = {
            'lightgbm': lgb.LGBMClassifier(
                objective='binary',
                metric='auc',
                boosting_type='gbdt',
                random_state=random_state,
                verbose=-1
            ),
            'xgboost': xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                random_state=random_state,
                verbosity=0
            ),
            'catboost': CatBoostClassifier(
                objective='Logloss',
                eval_metric='AUC',
                random_state=random_state,
                verbose=False
            ),
            'random_forest': RandomForestClassifier(
                random_state=random_state,
                n_jobs=-1
            ),
            'logistic_regression': LogisticRegression(
                random_state=random_state,
                max_iter=1000
            )
        }
        
        return models
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Perform cross-validation for a single model"""
        cv_folds = self.config.get('models', {}).get('cv_folds', 5)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                            random_state=self.config.get('models', {}).get('random_state', 42))
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        
        results = {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scores': cv_scores.tolist()
        }
        
        return results
    
    def train_base_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        """Train all base models and compare performance"""
        models = self.get_base_models()
        results = {}
        
        logger.info("Training base models...")
        
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Cross-validation
                cv_results = self.cross_validate_model(model, X, y)
                
                # Fit on full data
                model.fit(X, y)
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'cv_results': cv_results,
                    'fitted': True
                }
                
                # Log to ClearML
                if self.clearml_logger:
                    self.clearml_logger.report_scalar(
                        title="Model Performance",
                        series=f"{model_name}_cv_auc",
                        value=cv_results['mean_score'],
                        iteration=0
                    )
                
                logger.info(f"{model_name} - CV AUC: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results[model_name] = {
                    'model': None,
                    'cv_results': None,
                    'fitted': False,
                    'error': str(e)
                }
        
        self.models = results
        
        # Find best model
        best_model_name = max(
            [name for name, result in results.items() if result['fitted']],
            key=lambda name: results[name]['cv_results']['mean_score']
        )
        
        self.best_model = results[best_model_name]['model']
        self.best_score = results[best_model_name]['cv_results']['mean_score']
        
        logger.info(f"Best base model: {best_model_name} with AUC: {self.best_score:.4f}")
        
        return results
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                                model_name: str = 'lightgbm') -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            # Define hyperparameter search space based on model
            if model_name == 'lightgbm':
                params = {
                    'objective': 'binary',
                    'metric': 'auc',
                    'boosting_type': 'gbdt',
                    'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'random_state': self.config.get('models', {}).get('random_state', 42),
                    'verbose': -1
                }
                model = lgb.LGBMClassifier(**params)
                
            elif model_name == 'xgboost':
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'random_state': self.config.get('models', {}).get('random_state', 42),
                    'verbosity': 0
                }
                model = xgb.XGBClassifier(**params)
                
            elif model_name == 'catboost':
                params = {
                    'objective': 'Logloss',
                    'eval_metric': 'AUC',
                    'iterations': trial.suggest_int('iterations', 100, 1000),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'random_state': self.config.get('models', {}).get('random_state', 42),
                    'verbose': False
                }
                model = CatBoostClassifier(**params)
                
            elif model_name == 'logistic_regression':
                params = {
                    'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
                    'solver': 'saga',  # saga supports all penalties
                    'max_iter': trial.suggest_int('max_iter', 1000, 5000),
                    'random_state': self.config.get('models', {}).get('random_state', 42)
                }
                # Add l1_ratio for elasticnet penalty
                if params['penalty'] == 'elasticnet':
                    params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
                model = LogisticRegression(**params)
            
            else:
                raise ValueError(f"Unsupported model for optimization: {model_name}")
            
            # Cross-validation
            cv_results = self.cross_validate_model(model, X, y)
            return cv_results['mean_score']
        
        # Create study
        study_name = f"{self.config.get('optimization', {}).get('study_name', 'optimization')}_{model_name}"
        study = optuna.create_study(direction='maximize', study_name=study_name)
        
        # Optimize
        n_trials = self.config.get('optimization', {}).get('n_trials', 50)
        logger.info(f"Starting hyperparameter optimization for {model_name} with {n_trials} trials...")
        
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"Best parameters for {model_name}: {best_params}")
        logger.info(f"Best CV AUC: {best_score:.4f}")
        
        # Train final model with best parameters
        if model_name == 'lightgbm':
            best_params.update({
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'random_state': self.config.get('models', {}).get('random_state', 42),
                'verbose': -1
            })
            final_model = lgb.LGBMClassifier(**best_params)
        elif model_name == 'xgboost':
            best_params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': self.config.get('models', {}).get('random_state', 42),
                'verbosity': 0
            })
            final_model = xgb.XGBClassifier(**best_params)
        elif model_name == 'catboost':
            best_params.update({
                'objective': 'Logloss',
                'eval_metric': 'AUC',
                'random_state': self.config.get('models', {}).get('random_state', 42),
                'verbose': False
            })
            final_model = CatBoostClassifier(**best_params)
        elif model_name == 'logistic_regression':
            best_params.update({
                'random_state': self.config.get('models', {}).get('random_state', 42)
            })
            final_model = LogisticRegression(**best_params)
        
        final_model.fit(X, y)
        
        # Log to ClearML
        if self.clearml_logger:
            self.clearml_logger.report_scalar(
                title="Optimization Results",
                series=f"{model_name}_optimized_auc",
                value=best_score,
                iteration=0
            )
        
        # Update best model if this is better
        if best_score > self.best_score:
            self.best_model = final_model
            self.best_score = best_score
            logger.info(f"New best model: {model_name} with AUC: {best_score:.4f}")
        
        return {
            'model': final_model,
            'best_params': best_params,
            'best_score': best_score,
            'study': study
        }
    
    def create_ensemble_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Create ensemble model from best performing models"""
        if not self.models:
            raise ValueError("No models trained. Run train_base_models first.")
        
        # Get top 3 models
        fitted_models = {name: result for name, result in self.models.items() 
                        if result['fitted']}
        
        if len(fitted_models) < 2:
            logger.warning("Not enough models for ensemble. Returning best single model.")
            return {'model': self.best_model, 'type': 'single'}
        
        top_models = sorted(fitted_models.items(), 
                          key=lambda x: x[1]['cv_results']['mean_score'], 
                          reverse=True)[:3]
        
        logger.info(f"Creating ensemble from top {len(top_models)} models")
        
        # Simple averaging ensemble
        class EnsembleModel:
            def __init__(self, models, weights=None):
                self.models = models
                self.weights = weights or [1/len(models)] * len(models)
            
            def predict_proba(self, X):
                predictions = np.array([model.predict_proba(X)[:, 1] 
                                      for model in self.models])
                weighted_pred = np.average(predictions, weights=self.weights, axis=0)
                # Return in sklearn format
                return np.vstack([1-weighted_pred, weighted_pred]).T
            
            def predict(self, X):
                return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
        
        ensemble_models = [result['model'] for _, result in top_models]
        ensemble = EnsembleModel(ensemble_models)
        
        # Evaluate ensemble
        cv_results = self.cross_validate_model(ensemble, X, y)
        
        logger.info(f"Ensemble CV AUC: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        
        # Update best model if ensemble is better
        if cv_results['mean_score'] > self.best_score:
            self.best_model = ensemble
            self.best_score = cv_results['mean_score']
            logger.info(f"Ensemble is new best model with AUC: {cv_results['mean_score']:.4f}")
        
        return {
            'model': ensemble,
            'cv_results': cv_results,
            'component_models': [name for name, _ in top_models],
            'type': 'ensemble'
        }
    
    def save_best_model(self, filepath: str):
        """Save the best model to disk"""
        if self.best_model is None:
            raise ValueError("No model trained yet.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.best_model,
            'score': self.best_score,
            'timestamp': pd.Timestamp.now()
        }, filepath)
        
        logger.info(f"Best model saved to {filepath}")
        
        # Upload to ClearML
        if self.clearml_task:
            self.clearml_task.upload_artifact('best_model', artifact_object=filepath)


def train_competition_model(train_data_path: str, config_path: str = "config.yaml") -> ModelTrainer:
    """
    Complete training pipeline for competition
    
    Args:
        train_data_path: Path to preprocessed training data
        config_path: Path to configuration file
    
    Returns:
        Trained ModelTrainer instance
    """
    # Load preprocessed data
    if train_data_path.endswith('.npz'):
        data = np.load(train_data_path)
        X, y = data['X'], data['y']
    else:
        # Assume CSV format with last column as target
        df = pd.read_csv(train_data_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    
    logger.info(f"Loaded training data: X shape {X.shape}, y shape {y.shape}")
    
    # Initialize trainer
    trainer = ModelTrainer(config_path)
    
    # Train base models
    trainer.train_base_models(X, y)
    
    # Optimize best performing model
    best_base_model = max(
        [name for name, result in trainer.models.items() if result['fitted']],
        key=lambda name: trainer.models[name]['cv_results']['mean_score']
    )
    
    logger.info(f"Optimizing hyperparameters for {best_base_model}")
    trainer.optimize_hyperparameters(X, y, best_base_model)
    
    # Create ensemble
    trainer.create_ensemble_model(X, y)
    
    # Save best model
    trainer.save_best_model("models/best_model.pkl")
    
    logger.info(f"Training completed. Best AUC: {trainer.best_score:.4f}")
    
    return trainer


if __name__ == "__main__":
    # Example usage
    trainer = train_competition_model("data/processed_train.npz")