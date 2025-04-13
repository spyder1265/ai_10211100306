import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Append parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions
from utils.data_processor import preprocess_data
from utils.model_trainer import train_regression_model
from utils.visualizer import plot_regression_results

st.set_page_config(
    page_title="Regression Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Regression Analysis")
st.markdown("""
This section allows you to build a regression model to predict a continuous variable based on input data.
Upload your dataset, select features and target variable, and train a model to make predictions.
""")

# File upload
st.header("1. Upload Dataset")
uploaded_file = st.file_uploader("Upload a CSV file for regression analysis", type=["csv"])

if uploaded_file is not None:
    # Load and preview data
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset successfully loaded!")
        
        # Display dataset preview
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Display dataset info
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Number of rows: {df.shape[0]}")
            st.write(f"Number of columns: {df.shape[1]}")
        with col2:
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
        
        # Display basic statistics
        st.subheader("Statistical Summary")
        st.write(df.describe())
        
        # Data preprocessing
        st.header("2. Data Preprocessing")
        
        # Select features and target
        st.subheader("Select Features and Target")
        target_column = st.selectbox("Select the target column (variable to predict)", df.columns)
        
        # Remove target from feature selection
        feature_columns = [col for col in df.columns if col != target_column]
        selected_features = st.multiselect("Select features to use for prediction", feature_columns, feature_columns[:3])
        
        # Handle missing values
        st.subheader("Handle Missing Values")
        missing_method = st.radio("Select method to handle missing values", 
                                ["Drop rows with missing values", 
                                 "Fill missing values with mean", 
                                 "Fill missing values with median"])
        
        # Apply preprocessing
        if st.button("Preprocess Data"):
            # Check if features and target are selected
            if not selected_features:
                st.error("Please select at least one feature.")
            elif not target_column:
                st.error("Please select a target column.")
            else:
                # Get categorical features
                categorical_features = []
                for feature in selected_features:
                    if df[feature].dtype == 'object':
                        categorical_features.append(feature)
                
                # Handle categorical features if any exist
                df_copy = df.copy()
                if categorical_features:
                    st.info(f"Encoding categorical features: {', '.join(categorical_features)}")
                    from utils.data_processor import encode_categorical_features
                    df_encoded, encoders = encode_categorical_features(df_copy, categorical_features)
                    df_copy = df_encoded
                    st.session_state['encoders'] = encoders
                
                # Preprocess data
                X, y, df_processed = preprocess_data(df_copy, selected_features, target_column, missing_method)
                
                st.session_state['X'] = X
                st.session_state['y'] = y
                st.session_state['df_processed'] = df_processed
                st.session_state['features'] = selected_features
                st.session_state['target'] = target_column
                st.session_state['categorical_features'] = categorical_features
                
                st.success("Data preprocessing completed!")
                
                # Display processed data
                st.subheader("Processed Dataset")
                st.dataframe(df_processed.head())
                
                # Correlation matrix
                st.subheader("Correlation Matrix")
                corr = df_processed.corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                cax = ax.matshow(corr, cmap='coolwarm')
                plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
                plt.yticks(range(len(corr.columns)), corr.columns)
                fig.colorbar(cax)
                st.pyplot(fig)
        
        # Model training section (only show if data is preprocessed)
        if 'X' in st.session_state and 'y' in st.session_state:
            st.header("3. Model Training")
            
            # Model parameters
            st.subheader("Model Configuration")
            test_size = st.slider("Test set size (%)", 10, 50, 20) / 100
            random_state = st.number_input("Random state (for reproducibility)", 0, 100, 42)
            
            # Train model
            if st.button("Train Model"):
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    st.session_state['X'], 
                    st.session_state['y'], 
                    test_size=test_size, 
                    random_state=random_state
                )
                
                # Standardize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model, y_pred, metrics = train_regression_model(X_train_scaled, y_train, X_test_scaled, y_test)
                
                # Store model and related info in session state
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['metrics'] = metrics
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = y_pred
                
                st.success("Model training completed!")
                
                # Display model metrics
                st.subheader("Model Performance")
                metrics_df = pd.DataFrame({
                    'Metric': ['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'R² Score'],
                    'Value': [
                        metrics['mae'],
                        metrics['mse'],
                        metrics['rmse'],
                        metrics['r2']
                    ]
                })
                st.table(metrics_df)
                
                # Plot results
                st.subheader("Prediction Results")
                fig = plot_regression_results(st.session_state['y_test'], st.session_state['y_pred'], st.session_state['target'])
                st.pyplot(fig)
                
                # Feature importance
                st.subheader("Feature Importance")
                coef_df = pd.DataFrame({
                    'Feature': st.session_state['features'],
                    'Coefficient': model.coef_
                })
                coef_df = coef_df.sort_values('Coefficient', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(coef_df['Feature'], coef_df['Coefficient'])
                ax.set_xlabel('Coefficient')
                ax.set_ylabel('Feature')
                ax.set_title('Feature Importance')
                st.pyplot(fig)
                
            # Model prediction (only show if model is trained)
            if 'model' in st.session_state:
                st.header("4. Make Predictions")
                st.subheader("Enter values for prediction")
                
                # Create input fields for each feature
                input_data = {}
                for feature in st.session_state['features']:
                    # Check if feature is categorical
                    if feature in st.session_state.get('categorical_features', []):
                        # For categorical features, use a selectbox with original categories
                        unique_values = list(df[feature].unique())
                        selected_value = st.selectbox(f"Select value for {feature}", unique_values)
                        
                        # Get the encoded value if using encoders
                        if 'encoders' in st.session_state and feature in st.session_state['encoders']:
                            encoder = st.session_state['encoders'][feature]
                            encoded_value = encoder.transform([str(selected_value)])[0]
                            input_data[feature] = encoded_value
                        else:
                            input_data[feature] = selected_value
                    else:
                        # For numerical features, use a number input
                        try:
                            default_value = float(df[feature].mean())
                            input_data[feature] = st.number_input(f"Enter value for {feature}", value=default_value)
                        except:
                            # Fallback for any issues
                            default_value = 0.0
                            input_data[feature] = st.number_input(f"Enter value for {feature}", value=default_value)
                
                if st.button("Predict"):
                    # Convert input data to dataframe
                    input_df = pd.DataFrame([input_data])
                    
                    # Scale input data
                    input_scaled = st.session_state['scaler'].transform(input_df)
                    
                    # Make prediction
                    prediction = st.session_state['model'].predict(input_scaled)[0]
                    
                    # Display prediction
                    st.subheader("Prediction Result")
                    st.markdown(f"Predicted {st.session_state['target']}: **{prediction:.4f}**")
                    
                    # Add some context
                    mean_target = df[st.session_state['target']].mean()
                    if prediction > mean_target:
                        st.info(f"This prediction is above the average {st.session_state['target']} ({mean_target:.4f})")
                    else:
                        st.info(f"This prediction is below the average {st.session_state['target']} ({mean_target:.4f})")
        
    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")
else:
    st.info("Please upload a CSV file to begin.")

# Add info about using sample dataset
st.sidebar.header("Instructions")
st.sidebar.markdown("""
### How to use:
1. Upload a CSV file with numeric data
2. Select the target variable to predict
3. Choose features for the prediction
4. Preprocess the data
5. Train the model
6. Make predictions with custom inputs

### Example datasets:
- Housing price datasets
- Stock price datasets
- Sales prediction datasets
""")

# Add explanation of regression
st.sidebar.header("What is Regression?")
st.sidebar.markdown("""
Regression is a statistical method used to determine the relationship between a dependent variable (target) and one or more independent variables (features).

Linear regression attempts to model the relationship by fitting a linear equation to the observed data.
""")
