# 🌱 OptimalFert - Intelligent Fertilizer Recommendation System

A comprehensive machine learning solution for agricultural fertilizer recommendation, featuring advanced multi-class classification models, ClearML experiment tracking, and an interactive web application for precision agriculture.

## 📊 System Overview

**Application**: Intelligent Fertilizer Recommendation System
- **Type**: Multi-class Classification
- **Evaluation Metrics**: Accuracy, F1-Macro Score, Log-Loss
- **Goal**: Predict optimal fertilizer type based on soil conditions, crop information, and environmental data
- **Fertilizer Types**: 12 different fertilizer formulations (NPK combinations, organic options, specialty nutrients)

## 🏗️ Project Structure

```
OptimalFert/
├── 📁 data/                    # Agricultural dataset files
│   ├── train.csv              # Training data with soil, crop, and fertilizer information
│   ├── test.csv               # Test data for fertilizer recommendation
│   └── sample_submission.csv  # Submission format (id, Fertilizer Name)
├── 📁 src/                    # Source code
│   ├── 📁 data/               # Data processing modules
│   │   └── preprocessing.py   # Agricultural data preprocessing pipeline
│   ├── 📁 models/             # ML model modules
│   │   ├── train_model.py     # Multi-class model training with ClearML
│   │   └── predict.py         # Fertilizer prediction and recommendation
│   ├── 📁 visualization/      # EDA and plotting
│   │   └── eda.py            # Agricultural data analysis
│   └── 📁 web_app/           # Web application
│       └── streamlit_app.py  # Streamlit fertilizer recommendation app
├── 📁 models/                 # Saved models and preprocessors
├── 📁 outputs/                # EDA visualizations and results
├── 📁 notebooks/              # Jupyter notebooks (optional)
├── train_pipeline.py          # Complete fertilizer prediction pipeline
├── generate_submission.py     # Fertilizer recommendation generation
├── requirements.txt           # Python dependencies
├── config.yaml               # ClearML configuration file
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

The `data/` directory is ready for your agricultural data files. Place the following files:
- `train.csv` - Training dataset with soil, crop, and fertilizer information
- `test.csv` - Test dataset for prediction
- `sample_submission.csv` - Submission format with fertilizer names

Or use the built-in agricultural sample data generator for testing:
```python
from src.data.preprocessing import create_sample_data
create_sample_data(n_samples=2000, save_path="data/")
```

**Sample Features**:
- **Soil Properties**: pH, Nitrogen/Phosphorus/Potassium levels, Organic Matter
- **Environmental Conditions**: Temperature, Rainfall, Humidity
- **Crop Information**: Crop type, Growth stage, Field size, Previous fertilizer usage

## 🔄 Training Pipeline

### Complete Automated Pipeline

Run the entire fertilizer prediction pipeline with one command:

```bash
python train_pipeline.py
```

This will execute:
1. **Agricultural Data Loading & Validation**
2. **Exploratory Data Analysis** (saves visualizations to `outputs/`)
3. **Agricultural Data Preprocessing** (soil/crop feature engineering, fertilizer encoding)
4. **Multi-class Model Training** (5 different algorithms with cross-validation)
5. **Hyperparameter Optimization** (using Optuna for agricultural data)
6. **Ensemble Creation** (combines best fertilizer prediction models)
7. **Fertilizer Recommendation Generation** (`submission.csv` with fertilizer names)

### Individual Components

You can also run components separately:

```bash
# Generate fertilizer recommendations only (after training)
python generate_submission.py --model models/best_model.pkl --test_data data/test.csv

# With custom parameters
python generate_submission.py \
  --model models/best_model.pkl \
  --preprocessor models/preprocessor.pkl \
  --test_data data/test.csv \
  --output fertilizer_recommendations.csv \
  --validate
```

## 🧠 Machine Learning Pipeline

### Data Preprocessing
- **Agricultural Feature Detection**: Automatically identifies soil, environmental, and crop features
- **Missing Value Handling**: Median for numeric (pH, NPK levels), mode for categorical (crop types)
- **Feature Encoding**: One-hot encoding for categorical agricultural features
- **Scaling**: StandardScaler for soil and environmental measurements
- **Label Encoding**: Converts fertilizer names to numeric classes for training
- **Pipeline Persistence**: Reusable preprocessing for new agricultural data

### Model Architecture
- **LightGBM**: Multi-class gradient boosting optimized for agricultural feature patterns
- **XGBoost**: Multi-class extreme gradient boosting for fertilizer classification
- **CatBoost**: Categorical boosting ideal for mixed agricultural data types
- **Random Forest**: Ensemble of decision trees for robust fertilizer prediction
- **Logistic Regression**: Multi-class linear baseline model
- **Ensemble**: Weighted average of top-performing fertilizer prediction models

### Optimization Strategy
- **Cross-Validation**: 5-fold StratifiedKFold ensuring balanced fertilizer representation
- **Hyperparameter Tuning**: Optuna-based optimization for agricultural prediction (50+ trials)
- **Metric Focus**: Accuracy and F1-Macro score maximization for multi-class fertilizer prediction
- **Model Selection**: Best individual fertilizer predictor vs ensemble comparison

### Experiment Tracking (ClearML)
- **Hyperparameters**: All fertilizer prediction model configurations logged
- **Metrics**: Cross-validation scores, Accuracy, F1-Macro tracking
- **Artifacts**: Fertilizer prediction models, agricultural preprocessors, visualizations
- **Reproducibility**: Random seed management and versioning for agricultural experiments

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