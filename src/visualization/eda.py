"""
Exploratory Data Analysis (EDA) module for Playground Series S5E6
Comprehensive analysis and visualization of the competition dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger(__name__)


class EDAAnalyzer:
    """
    Comprehensive EDA analyzer for competition data
    """
    
    def __init__(self, train_df: pd.DataFrame, target_column: str = 'target'):
        self.train_df = train_df.copy()
        self.target_column = target_column
        self.numeric_features = []
        self.categorical_features = []
        self.analysis_results = {}
        
        self._identify_feature_types()
    
    def _identify_feature_types(self):
        """Identify numeric and categorical features"""
        feature_columns = [col for col in self.train_df.columns 
                          if col not in ['id', self.target_column]]
        
        for col in feature_columns:
            if self.train_df[col].dtype in ['int64', 'float64']:
                if self.train_df[col].nunique() <= 10 and self.train_df[col].dtype == 'int64':
                    self.categorical_features.append(col)
                else:
                    self.numeric_features.append(col)
            else:
                self.categorical_features.append(col)
    
    def basic_info(self):
        """Generate basic dataset information"""
        target_counts = self.train_df[self.target_column].value_counts()
        
        info = {
            'shape': self.train_df.shape,
            'missing_values': self.train_df.isnull().sum().sum(),
            'target_distribution': target_counts.to_dict(),
            'target_classes': len(target_counts),
            'numeric_features_count': len(self.numeric_features),
            'categorical_features_count': len(self.categorical_features),
            'memory_usage_mb': self.train_df.memory_usage(deep=True).sum() / 1024**2
        }
        
        self.analysis_results['basic_info'] = info
        logger.info(f"Dataset shape: {info['shape']}")
        logger.info(f"Target distribution: {info['target_distribution']}")
        logger.info(f"Number of target classes: {info['target_classes']}")
        
        return info
    
    def missing_values_analysis(self):
        """Analyze missing values pattern"""
        missing_stats = pd.DataFrame({
            'feature': self.train_df.columns,
            'missing_count': self.train_df.isnull().sum(),
            'missing_percentage': (self.train_df.isnull().sum() / len(self.train_df)) * 100
        })
        missing_stats = missing_stats[missing_stats['missing_count'] > 0].sort_values(
            'missing_percentage', ascending=False)
        
        self.analysis_results['missing_values'] = missing_stats
        
        if len(missing_stats) > 0:
            logger.warning(f"Found missing values in {len(missing_stats)} features")
        else:
            logger.info("No missing values found")
        
        return missing_stats
    
    def target_analysis(self):
        """Detailed target variable analysis"""
        value_counts = self.train_df[self.target_column].value_counts()
        percentages = self.train_df[self.target_column].value_counts(normalize=True) * 100
        
        target_stats = {
            'value_counts': value_counts,
            'percentage': percentages,
            'unique_classes': self.train_df[self.target_column].nunique(),
            'most_frequent_class': value_counts.index[0],
            'least_frequent_class': value_counts.index[-1],
            'class_imbalance_ratio': value_counts.iloc[0] / value_counts.iloc[-1]
        }
        
        self.analysis_results['target_analysis'] = target_stats
        return target_stats
    
    def numeric_features_analysis(self):
        """Analyze numeric features"""
        if not self.numeric_features:
            return {}
        
        numeric_stats = self.train_df[self.numeric_features].describe()
        
        # For categorical target, calculate ANOVA F-statistic instead of correlation
        from scipy.stats import f_oneway
        target_relationships = {}
        for feature in self.numeric_features:
            # Group feature values by target class
            groups = [self.train_df[self.train_df[self.target_column] == class_][feature].dropna() 
                     for class_ in self.train_df[self.target_column].unique()]
            # Filter out empty groups
            groups = [group for group in groups if len(group) > 0]
            
            if len(groups) > 1:
                try:
                    f_stat, p_value = f_oneway(*groups)
                    target_relationships[feature] = {'f_statistic': f_stat, 'p_value': p_value}
                except:
                    target_relationships[feature] = {'f_statistic': 0, 'p_value': 1}
        
        # Skewness and kurtosis
        skewness = self.train_df[self.numeric_features].skew()
        kurtosis = self.train_df[self.numeric_features].kurtosis()
        
        numeric_analysis = {
            'descriptive_stats': numeric_stats,
            'target_relationships': target_relationships,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
        
        self.analysis_results['numeric_analysis'] = numeric_analysis
        logger.info(f"Analyzed {len(self.numeric_features)} numeric features")
        
        return numeric_analysis
    
    def categorical_features_analysis(self):
        """Analyze categorical features"""
        if not self.categorical_features:
            return {}
        
        categorical_stats = {}
        
        for feature in self.categorical_features:
            stats_dict = {
                'unique_values': self.train_df[feature].nunique(),
                'value_counts': self.train_df[feature].value_counts(),
                'missing_count': self.train_df[feature].isnull().sum()
            }
            
            # For categorical target, calculate mode and count by category
            if self.train_df[self.target_column].dtype == 'object':
                # Group by feature and get the most common target for each category
                target_mode_by_category = self.train_df.groupby(feature)[self.target_column].agg(['count', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None])
                target_mode_by_category.columns = ['count', 'most_common_target']
                stats_dict['target_distribution_by_category'] = target_mode_by_category
            else:
                # For numeric target, calculate mean and count
                target_rate = self.train_df.groupby(feature)[self.target_column].agg(['mean', 'count'])
                stats_dict['target_rate_by_category'] = target_rate
            
            categorical_stats[feature] = stats_dict
        
        self.analysis_results['categorical_analysis'] = categorical_stats
        logger.info(f"Analyzed {len(self.categorical_features)} categorical features")
        
        return categorical_stats
    
    def feature_importance_analysis(self):
        """Calculate feature importance using mutual information"""
        if not self.numeric_features and not self.categorical_features:
            return {}
        
        # Prepare data for mutual information
        X = self.train_df[self.numeric_features + self.categorical_features].copy()
        y = self.train_df[self.target_column]
        
        # Handle categorical features
        for cat_feature in self.categorical_features:
            X[cat_feature] = X[cat_feature].astype('category').cat.codes
        
        # Handle categorical target - encode to numeric for mutual information
        if y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = y
        
        # Fill missing values
        X = X.fillna(X.median())
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'mutual_info_score': mi_scores
        }).sort_values('mutual_info_score', ascending=False)
        
        self.analysis_results['feature_importance'] = feature_importance
        logger.info("Feature importance analysis completed")
        
        return feature_importance
    
    def create_visualizations(self, save_path: str = "outputs/"):
        """Create comprehensive visualizations"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Target distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar plot
        target_counts = self.train_df[self.target_column].value_counts()
        axes[0].bar(target_counts.index, target_counts.values)
        axes[0].set_title('Target Distribution')
        axes[0].set_xlabel('Target Value')
        axes[0].set_ylabel('Count')
        
        # Pie plot
        axes[1].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%')
        axes[1].set_title('Target Distribution (Percentage)')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/target_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Numeric features distribution
        if self.numeric_features:
            n_features = len(self.numeric_features)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, feature in enumerate(self.numeric_features):
                if i < len(axes):
                    axes[i].hist(self.train_df[feature], bins=30, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'Distribution of {feature}')
                    axes[i].set_xlabel(feature)
                    axes[i].set_ylabel('Frequency')
            
            # Hide extra subplots
            for i in range(len(self.numeric_features), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/numeric_features_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Correlation heatmap (numeric features only for categorical target)
        if len(self.numeric_features) > 1:
            corr_matrix = self.train_df[self.numeric_features].corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5)
            plt.title('Numeric Features Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(f"{save_path}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Feature importance plot
        if 'feature_importance' in self.analysis_results:
            feature_importance = self.analysis_results['feature_importance']
            
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(15)
            
            plt.barh(range(len(top_features)), top_features['mutual_info_score'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Mutual Information Score')
            plt.title('Top 15 Features by Mutual Information')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f"{save_path}/feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to {save_path}")
    
    def generate_report(self) -> dict:
        """Generate comprehensive EDA report"""
        logger.info("Starting comprehensive EDA analysis...")
        
        # Run all analyses
        self.basic_info()
        self.missing_values_analysis()
        self.target_analysis()
        self.numeric_features_analysis()
        self.categorical_features_analysis()
        self.feature_importance_analysis()
        
        # Create visualizations
        self.create_visualizations()
        
        logger.info("EDA analysis completed")
        return self.analysis_results


def run_eda(train_path: str, target_column: str = 'target') -> dict:
    """
    Run complete EDA on training data
    
    Args:
        train_path: Path to training CSV file
        target_column: Name of target column
        
    Returns:
        Dictionary containing all analysis results
    """
    # Load data
    train_df = pd.read_csv(train_path)
    logger.info(f"Loaded training data with shape: {train_df.shape}")
    
    # Initialize analyzer
    analyzer = EDAAnalyzer(train_df, target_column)
    
    # Generate report
    results = analyzer.generate_report()
    
    return results


if __name__ == "__main__":
    # Example usage
    results = run_eda("data/train.csv")