# Notebooks Directory

This directory is intended for Jupyter notebooks that provide interactive analysis and experimentation.

## Suggested Notebooks

1. **01_EDA.ipynb** - Interactive exploratory data analysis
2. **02_Feature_Engineering.ipynb** - Feature engineering experiments
3. **03_Model_Experiments.ipynb** - Individual model training and evaluation
4. **04_Ensemble_Analysis.ipynb** - Ensemble model development
5. **05_Error_Analysis.ipynb** - Model error analysis and insights

## Usage

Create notebooks here to explore data, experiment with features, and test model approaches before integrating them into the main pipeline.

```python
# Common imports for notebooks
import sys
import os
sys.path.append('../src')

from src.data.preprocessing import DataPreprocessor
from src.visualization.eda import EDAAnalyzer
from src.models.train_model import ModelTrainer
```