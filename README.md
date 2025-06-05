# 🎯 Playground Series S5E6 - OptimalFert ML Pipeline

A comprehensive machine learning solution for Kaggle's Playground Series S5E6 competition, featuring advanced model training, ClearML experiment tracking, and an interactive web application.

## 📊 Competition Overview

**Competition**: [Playground Series S5E6](https://www.kaggle.com/competitions/playground-series-s5e6/)
- **Type**: Binary Classification
- **Evaluation Metric**: ROC AUC (Area Under the Receiver Operating Characteristic Curve)
- **Goal**: Predict binary target variable from multiple features

## 🏗️ Project Structure

```
OptimalFert/
├── 📁 data/                    # Dataset files
│   ├── train.csv              # Training data
│   ├── test.csv               # Test data
│   └── sample_submission.csv  # Submission format
├── 📁 src/                    # Source code
│   ├── 📁 data/               # Data processing modules
│   │   └── preprocessing.py   # Data preprocessing pipeline
│   ├── 📁 models/             # ML model modules
│   │   ├── train_model.py     # Model training with ClearML
│   │   └── predict.py         # Prediction and submission
│   ├── 📁 visualization/      # EDA and plotting
│   │   └── eda.py            # Exploratory Data Analysis
│   └── 📁 web_app/           # Web application
│       └── streamlit_app.py  # Streamlit prediction app
├── 📁 models/                 # Saved models and preprocessors
├── 📁 outputs/                # EDA visualizations and results
├── 📁 notebooks/              # Jupyter notebooks (optional)
├── train_pipeline.py          # Complete training pipeline
├── generate_submission.py     # Submission generation script
├── requirements.txt           # Python dependencies
├── config.yaml.template       # ClearML configuration template
└── .gitignore                # Git ignore rules
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd OptimalFert

# Install dependencies
pip install -r requirements.txt
```

### 2. ClearML Configuration

```bash
# Copy configuration template
cp config.yaml.template config.yaml

# Edit config.yaml with your ClearML credentials
# Get your API keys from: https://app.clear.ml/settings/workspace-configuration
```

Example `config.yaml`:
```yaml
clearml:
  api_host: "https://api.clear.ml"
  web_host: "https://app.clear.ml"
  files_host: "https://files.clear.ml"
  api_key: "YOUR_API_KEY_HERE"
  api_secret_key: "YOUR_SECRET_KEY_HERE"
```

### 3. Data Preparation

Place your competition data files in the `data/` directory:
- `train.csv` - Training dataset
- `test.csv` - Test dataset  
- `sample_submission.csv` - Submission format

Or use the built-in sample data generator for testing:
```python
from src.data.preprocessing import create_sample_data
create_sample_data(n_samples=2000, n_features=15, save_path="data/")
```

## 🔄 Training Pipeline

### Complete Automated Pipeline

Run the entire ML pipeline with one command:

```bash
python train_pipeline.py
```

This will execute:
1. **Data Loading & Validation**
2. **Exploratory Data Analysis** (saves visualizations to `outputs/`)
3. **Data Preprocessing** (feature engineering, encoding, scaling)
4. **Model Training** (5 different algorithms with cross-validation)
5. **Hyperparameter Optimization** (using Optuna)
6. **Ensemble Creation** (combines best models)
7. **Submission Generation** (`submission.csv`)

### Individual Components

You can also run components separately:

```bash
# Generate submission only (after training)
python generate_submission.py --model models/best_model.pkl --test_data data/test.csv

# With custom parameters
python generate_submission.py \
  --model models/best_model.pkl \
  --preprocessor models/preprocessor.pkl \
  --test_data data/test.csv \
  --output my_submission.csv \
  --validate
```

## 🧠 Machine Learning Pipeline

### Data Preprocessing
- **Automatic Feature Type Detection**: Distinguishes numeric vs categorical features
- **Missing Value Handling**: Median for numeric, mode for categorical
- **Feature Encoding**: One-hot encoding for categorical features
- **Scaling**: StandardScaler for numeric features
- **Pipeline Persistence**: Reusable preprocessing for test data

### Model Architecture
- **LightGBM**: Gradient boosting with hyperparameter optimization
- **XGBoost**: Extreme gradient boosting
- **CatBoost**: Categorical boosting algorithm
- **Random Forest**: Ensemble of decision trees
- **Logistic Regression**: Linear baseline model
- **Ensemble**: Weighted average of top-performing models

### Optimization Strategy
- **Cross-Validation**: 5-fold StratifiedKFold for robust evaluation
- **Hyperparameter Tuning**: Optuna-based optimization (50+ trials)
- **Metric Focus**: ROC AUC maximization throughout
- **Model Selection**: Best individual model vs ensemble comparison

### Experiment Tracking (ClearML)
- **Hyperparameters**: All model configurations logged
- **Metrics**: Cross-validation scores, ROC AUC tracking
- **Artifacts**: Models, preprocessors, visualizations
- **Reproducibility**: Random seed management and versioning

## 🌐 Web Application

Launch the interactive prediction interface:

```bash
streamlit run src/web_app/streamlit_app.py
```

### Features:
- **Single Predictions**: Input features manually for real-time predictions
- **Batch Predictions**: Upload CSV files for multiple predictions
- **Interactive Interface**: User-friendly feature input with validation
- **Visualization**: Prediction gauge and probability distributions
- **Export Capability**: Download batch prediction results

### Usage:
1. Navigate to the web interface (typically `http://localhost:8501`)
2. Enter feature values in the sidebar
3. View real-time prediction probability
4. Use batch mode for multiple predictions

## 📈 Exploratory Data Analysis

The EDA module automatically generates:

### Statistical Analysis
- **Dataset Overview**: Shape, missing values, memory usage
- **Target Distribution**: Class balance analysis
- **Feature Statistics**: Descriptive statistics for all features
- **Correlation Analysis**: Feature relationships and target correlations

### Visualizations (saved to `outputs/`)
- `target_distribution.png` - Target class distribution
- `numeric_features_distribution.png` - Feature distributions
- `correlation_heatmap.png` - Feature correlation matrix
- `feature_importance.png` - Mutual information scores

### Feature Insights
- **Mutual Information**: Feature importance ranking
- **Categorical Analysis**: Target rates by category
- **Outlier Detection**: Statistical outlier identification
- **Distribution Analysis**: Skewness and kurtosis metrics

## 🎯 Model Performance

### Evaluation Metrics
- **Primary**: ROC AUC (competition metric)
- **Cross-Validation**: 5-fold stratified validation
- **Ensemble**: Weighted averaging of top models
- **Baseline**: Multiple algorithm comparison

### Expected Performance
- **Individual Models**: ROC AUC 0.75-0.85 (depends on data)
- **Optimized Models**: ROC AUC 0.80-0.90 with hyperparameter tuning
- **Ensemble**: Typically 1-3% improvement over best individual model

## 📝 Submission Generation

### Automated Process
```bash
python generate_submission.py
```

### Manual Process
```python
from src.models.predict import generate_submission

submission = generate_submission(
    model_path="models/best_model.pkl",
    preprocessor_path="models/preprocessor.pkl", 
    test_data_path="data/test.csv",
    output_path="submission.csv"
)
```

### Validation Features
- **Format Compliance**: Strict adherence to `sample_submission.csv`
- **ID Matching**: Ensures correct test sample ordering
- **Value Range**: Validates probability ranges [0, 1]
- **Missing Values**: Checks for and prevents null predictions

## 🔧 Configuration

### Model Parameters
Key hyperparameters (optimized via Optuna):
- **LightGBM**: `num_leaves`, `learning_rate`, `feature_fraction`
- **XGBoost**: `max_depth`, `learning_rate`, `subsample`
- **CatBoost**: `depth`, `learning_rate`, `l2_leaf_reg`

### Training Configuration
- **Cross-Validation**: 5 folds (configurable)
- **Random State**: 42 (reproducibility)
- **Optimization Trials**: 50+ per model
- **Ensemble**: Top 3 models weighted averaging

## 🚨 Troubleshooting

### Common Issues

**ClearML Connection Error**:
```bash
# Check API credentials in config.yaml
# Verify internet connection to clear.ml
```

**Memory Issues**:
```bash
# Reduce dataset size for testing
# Use feature selection for large datasets
```

**Model Training Fails**:
```bash
# Check data format and preprocessing
# Verify sufficient samples for cross-validation
```

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Kaggle**: For providing the Playground Series competition platform
- **ClearML**: For experiment tracking and model management
- **Streamlit**: For the web application framework
- **Optuna**: For hyperparameter optimization
- **Open Source Community**: For the amazing ML libraries (scikit-learn, LightGBM, XGBoost, etc.)

## 📞 Support

For questions and support:
- 📧 Create an issue in this repository
- 💬 Check the troubleshooting section above
- 📖 Refer to the detailed code documentation

---

**Happy Machine Learning! 🚀**