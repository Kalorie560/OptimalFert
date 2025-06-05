"""
Streamlit web application for Playground Series S5E6 predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.models.predict import CompetitionPredictor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title="Playground Series S5E6 Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_predictor():
    """Load the trained model and preprocessor"""
    try:
        model_path = os.path.join(project_root, "models", "best_model.pkl")
        preprocessor_path = os.path.join(project_root, "models", "preprocessor.pkl")
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            return None
        
        if not os.path.exists(preprocessor_path):
            st.error(f"Preprocessor file not found at {preprocessor_path}")
            return None
        
        predictor = CompetitionPredictor(model_path, preprocessor_path)
        return predictor
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_sample_data():
    """Load sample data to understand feature ranges"""
    try:
        train_path = os.path.join(project_root, "data", "train.csv")
        if os.path.exists(train_path):
            return pd.read_csv(train_path)
        else:
            return None
    except Exception as e:
        logger.warning(f"Could not load sample data: {e}")
        return None


def create_feature_inputs(predictor, sample_data):
    """Create input widgets for all features"""
    feature_values = {}
    
    if predictor is None:
        st.error("Model not loaded. Please check the model files.")
        return {}
    
    # Get feature information from preprocessor
    numeric_features = predictor.preprocessor.numeric_features
    categorical_features = predictor.preprocessor.categorical_features
    
    st.sidebar.header("📊 Feature Input")
    
    # Numeric features
    if numeric_features:
        st.sidebar.subheader("Numeric Features")
        
        for feature in numeric_features:
            # Get feature statistics from sample data
            if sample_data is not None and feature in sample_data.columns:
                feature_stats = sample_data[feature].describe()
                min_val = float(feature_stats['min'])
                max_val = float(feature_stats['max'])
                mean_val = float(feature_stats['mean'])
                std_val = float(feature_stats['std'])
                
                # Create input with reasonable bounds
                feature_values[feature] = st.sidebar.number_input(
                    f"{feature}",
                    min_value=min_val - 2*std_val,
                    max_value=max_val + 2*std_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100,
                    help=f"Range in data: [{min_val:.2f}, {max_val:.2f}], Mean: {mean_val:.2f}"
                )
            else:
                # Default input without sample data
                feature_values[feature] = st.sidebar.number_input(
                    f"{feature}",
                    value=0.0,
                    help="Enter numeric value"
                )
    
    # Categorical features
    if categorical_features:
        st.sidebar.subheader("Categorical Features")
        
        for feature in categorical_features:
            if sample_data is not None and feature in sample_data.columns:
                # Get unique values from sample data
                unique_values = sorted(sample_data[feature].dropna().unique())
                
                if len(unique_values) <= 20:  # Use selectbox for small number of categories
                    feature_values[feature] = st.sidebar.selectbox(
                        f"{feature}",
                        options=unique_values,
                        help=f"Available options: {unique_values}"
                    )
                else:  # Use text input for many categories
                    feature_values[feature] = st.sidebar.text_input(
                        f"{feature}",
                        value=str(unique_values[0]) if unique_values else "",
                        help=f"Sample values: {unique_values[:5]}..."
                    )
            else:
                # Default input without sample data
                feature_values[feature] = st.sidebar.text_input(
                    f"{feature}",
                    value="",
                    help="Enter categorical value"
                )
    
    return feature_values


def main():
    """Main application"""
    
    # Title and description
    st.title("🎯 Playground Series S5E6 Predictor")
    st.markdown("""
    This web application provides predictions for the Kaggle Playground Series S5E6 competition.
    Enter feature values in the sidebar to get real-time predictions.
    """)
    
    # Load model and data
    predictor = load_predictor()
    sample_data = load_sample_data()
    
    if predictor is None:
        st.error("⚠️ Model not available. Please ensure model files exist.")
        st.info("""
        To use this application:
        1. Train a model using the training pipeline
        2. Ensure model files are saved in the `models/` directory
        3. Restart the application
        """)
        return
    
    # Create main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📈 Prediction Results")
        
        # Get feature inputs
        feature_values = create_feature_inputs(predictor, sample_data)
        
        if feature_values:
            try:
                # Make prediction
                prediction = predictor.predict_single_sample(feature_values)
                
                # Display prediction
                st.metric(
                    label="Prediction Probability",
                    value=f"{prediction:.4f}",
                    help="Probability of positive class (target=1)"
                )
                
                # Prediction interpretation
                if prediction > 0.7:
                    st.success("🟢 High probability of positive class")
                elif prediction > 0.3:
                    st.warning("🟡 Moderate probability")
                else:
                    st.info("🔵 Low probability of positive class")
                
                # Prediction gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prediction,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Prediction Score"},
                    delta = {'reference': 0.5},
                    gauge = {'axis': {'range': [None, 1]},
                             'bar': {'color': "darkblue"},
                             'steps' : [
                                 {'range': [0, 0.3], 'color': "lightgray"},
                                 {'range': [0.3, 0.7], 'color': "gray"},
                                 {'range': [0.7, 1], 'color': "lightgreen"}],
                             'threshold' : {'line': {'color': "red", 'width': 4},
                                          'thickness': 0.75, 'value': 0.5}}))
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    
    with col2:
        st.header("ℹ️ Information")
        
        # Model information
        st.subheader("Model Details")
        st.info(f"""
        **Features**: {len(predictor.preprocessor.numeric_features) + len(predictor.preprocessor.categorical_features)}
        - Numeric: {len(predictor.preprocessor.numeric_features)}
        - Categorical: {len(predictor.preprocessor.categorical_features)}
        
        **Target**: Binary classification (0/1)
        **Metric**: ROC AUC
        """)
        
        # Feature importance (if available)
        if sample_data is not None:
            st.subheader("Dataset Overview")
            st.info(f"""
            **Training samples**: {len(sample_data):,}
            **Target distribution**:
            - Class 0: {(sample_data['target'] == 0).sum():,} ({(sample_data['target'] == 0).mean()*100:.1f}%)
            - Class 1: {(sample_data['target'] == 1).sum():,} ({(sample_data['target'] == 1).mean()*100:.1f}%)
            """)
        
        # Instructions
        st.subheader("How to Use")
        st.markdown("""
        1. 📝 Enter feature values in the sidebar
        2. 🔄 Prediction updates automatically
        3. 📊 View prediction probability and gauge
        4. 🎯 Values closer to 1.0 indicate higher likelihood of positive class
        """)


def batch_prediction_page():
    """Page for batch predictions from CSV file"""
    st.title("📊 Batch Predictions")
    st.markdown("Upload a CSV file to get predictions for multiple samples.")
    
    predictor = load_predictor()
    
    if predictor is None:
        st.error("Model not available.")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with the same features as training data"
    )
    
    if uploaded_file is not None:
        try:
            # Read file
            df = pd.read_csv(uploaded_file)
            st.write("**Uploaded data preview:**")
            st.dataframe(df.head())
            
            # Make predictions
            if st.button("Generate Predictions"):
                with st.spinner("Generating predictions..."):
                    predictions = []
                    
                    for idx, row in df.iterrows():
                        pred = predictor.predict_single_sample(row.to_dict())
                        predictions.append(pred)
                    
                    # Add predictions to dataframe
                    result_df = df.copy()
                    result_df['prediction'] = predictions
                    result_df['predicted_class'] = (np.array(predictions) > 0.5).astype(int)
                    
                    # Display results
                    st.success("Predictions generated successfully!")
                    st.dataframe(result_df)
                    
                    # Download button
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="Download predictions as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    st.subheader("Prediction Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Samples", len(predictions))
                    with col2:
                        st.metric("Mean Probability", f"{np.mean(predictions):.4f}")
                    with col3:
                        st.metric("Predicted Positive", f"{sum(np.array(predictions) > 0.5)}")
                    
                    # Histogram of predictions
                    fig = px.histogram(
                        x=predictions,
                        bins=20,
                        title="Distribution of Prediction Probabilities"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {e}")


if __name__ == "__main__":
    # Create page navigation
    page = st.sidebar.selectbox(
        "Navigate",
        ["Single Prediction", "Batch Predictions"]
    )
    
    if page == "Single Prediction":
        main()
    elif page == "Batch Predictions":
        batch_prediction_page()