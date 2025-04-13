import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import io

# TensorFlow import error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import Callback
    TF_AVAILABLE = True
except (ImportError, TypeError):
    TF_AVAILABLE = False
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Append parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions
from utils.data_processor import preprocess_data
from utils.visualizer import plot_training_history

# Custom callback to store training progress
if TF_AVAILABLE:
    class TrainingCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            self.losses = []
            self.val_losses = []
            self.accuracies = []
            self.val_accuracies = []
            
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.accuracies.append(logs.get('accuracy') if 'accuracy' in logs else logs.get('acc'))
            self.val_accuracies.append(logs.get('val_accuracy') if 'val_accuracy' in logs else logs.get('val_acc'))
else:
    # Dummy class when TensorFlow is not available
    class TrainingCallback:
        def __init__(self):
            self.losses = []
            self.val_losses = []
            self.accuracies = []
            self.val_accuracies = []

st.set_page_config(
    page_title="Neural Network Training",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Neural Network Training")
st.markdown("""
This section allows you to design and train a neural network on your dataset.
Upload your data, configure the network architecture, and visualize the training process.
""")

# File upload
st.header("1. Upload Dataset")
uploaded_file = st.file_uploader("Upload a CSV file for neural network training", type=["csv"])

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
        target_column = st.selectbox("Select the target column (classification target)", df.columns)
        
        # Remove target from feature selection
        feature_columns = [col for col in df.columns if col != target_column]
        selected_features = st.multiselect("Select features for training", feature_columns, feature_columns[:3])
        
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
                # Preprocess features
                X, _, df_processed = preprocess_data(df, selected_features, None, missing_method)
                
                # Process target
                if df[target_column].dtype == 'object' or df[target_column].nunique() < 10:
                    # Classification task - encode target
                    le = LabelEncoder()
                    y = le.fit_transform(df[target_column].astype(str))
                    n_classes = len(le.classes_)
                    class_names = list(le.classes_)
                    task_type = 'classification'
                    st.write(f"Detected **classification** task with {n_classes} classes: {class_names}")
                else:
                    # Regression task
                    y = df[target_column].values
                    n_classes = 1
                    class_names = None
                    task_type = 'regression'
                    st.write(f"Detected **regression** task for target: {target_column}")
                
                # Store processed data
                st.session_state['X'] = X
                st.session_state['y'] = y
                st.session_state['df_processed'] = df_processed
                st.session_state['features'] = selected_features
                st.session_state['target'] = target_column
                st.session_state['n_classes'] = n_classes
                st.session_state['class_names'] = class_names
                st.session_state['task_type'] = task_type
                
                if task_type == 'classification':
                    st.session_state['label_encoder'] = le
                
                st.success("Data preprocessing completed!")
                
                # Display class distribution for classification
                if task_type == 'classification':
                    st.subheader("Class Distribution")
                    class_counts = pd.Series(y).value_counts().sort_index()
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(class_names, class_counts.values)
                    ax.set_xlabel('Class')
                    ax.set_ylabel('Count')
                    ax.set_title('Class Distribution')
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig)
        
        # Neural Network configuration (only show if data is preprocessed)
        if 'X' in st.session_state and 'y' in st.session_state:
            st.header("3. Neural Network Configuration")
            
            # Data splitting
            st.subheader("Data Splitting")
            test_size = st.slider("Test set size (%)", 10, 50, 20) / 100
            validation_size = st.slider("Validation set size (%)", 10, 50, 20) / 100
            random_state = st.number_input("Random state (for reproducibility)", 0, 100, 42)
            
            # Network architecture
            st.subheader("Network Architecture")
            col1, col2 = st.columns(2)
            
            with col1:
                n_layers = st.slider("Number of hidden layers", 1, 5, 2)
                units_per_layer = [st.slider(f"Units in layer {i+1}", 4, 256, 64, step=4) for i in range(n_layers)]
                activation = st.selectbox("Activation function", ["relu", "sigmoid", "tanh"], 0)
            
            with col2:
                dropout_rate = st.slider("Dropout rate", 0.0, 0.5, 0.2, 0.05)
                batch_size = st.slider("Batch size", 8, 128, 32, step=8)
                epochs = st.slider("Number of epochs", 10, 100, 30)
                learning_rate = st.select_slider(
                    "Learning rate",
                    options=[0.001, 0.01, 0.05, 0.1],
                    value=0.01
                )
            
            # Check if TensorFlow is available
            if not TF_AVAILABLE:
                st.error("TensorFlow is not available or encountering errors. Neural Network functionality is disabled.")
                st.info("You can still use the other modules: Regression, Clustering, and Document Analyzer.")
            
            # Train model
            if TF_AVAILABLE and st.button("Train Model"):
                # Split data
                X_train, X_temp, y_train, y_temp = train_test_split(
                    st.session_state['X'], 
                    st.session_state['y'], 
                    test_size=(test_size + validation_size), 
                    random_state=random_state
                )
                
                # Further split temp data into validation and test
                split_ratio = validation_size / (test_size + validation_size)
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=split_ratio, random_state=random_state
                )
                
                # Standardize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)
                
                # Build model
                model = Sequential()
                
                # Input layer
                model.add(Dense(units_per_layer[0], activation=activation, input_shape=(X_train.shape[1],)))
                model.add(Dropout(dropout_rate))
                
                # Hidden layers
                for i in range(1, n_layers):
                    model.add(Dense(units_per_layer[i], activation=activation))
                    model.add(Dropout(dropout_rate))
                
                # Output layer
                if st.session_state['task_type'] == 'classification':
                    n_classes = st.session_state['n_classes']
                    if n_classes == 2:  # Binary classification
                        model.add(Dense(1, activation='sigmoid'))
                        loss = 'binary_crossentropy'
                    else:  # Multi-class classification
                        model.add(Dense(n_classes, activation='softmax'))
                        loss = 'sparse_categorical_crossentropy'
                    metrics = ['accuracy']
                else:  # Regression
                    model.add(Dense(1, activation='linear'))
                    loss = 'mse'
                    metrics = ['mae']
                
                # Compile model
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                
                # Display model summary
                st.subheader("Model Summary")
                stringlist = []
                model.summary(print_fn=lambda x: stringlist.append(x))
                model_summary = "\n".join(stringlist)
                st.text(model_summary)
                
                # Custom callback for progress tracking
                training_callback = TrainingCallback()
                
                # Progress bar placeholder
                progress_bar = st.progress(0)
                epoch_status = st.empty()
                metrics_container = st.container()
                plot_container = st.container()
                
                # Define a custom keras callback to update the progress bar
                class ProgressCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress = (epoch + 1) / epochs
                        progress_bar.progress(progress)
                        epoch_status.text(f"Epoch {epoch + 1}/{epochs}")
                        
                        with metrics_container:
                            if st.session_state['task_type'] == 'classification':
                                metric_name = 'accuracy'
                                val_metric_name = 'val_accuracy'
                            else:
                                metric_name = 'mae'
                                val_metric_name = 'val_mae'
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Loss", f"{logs.get('loss'):.4f}")
                            col2.metric("Validation Loss", f"{logs.get('val_loss'):.4f}")
                            col3.metric(metric_name.capitalize(), f"{logs.get(metric_name):.4f}")
                            col4.metric(f"Val {metric_name.capitalize()}", f"{logs.get(val_metric_name):.4f}")
                        
                        # Update plots
                        with plot_container:
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                            
                            # Loss plot
                            ax1.plot(training_callback.losses[:epoch+1], label='Training Loss')
                            ax1.plot(training_callback.val_losses[:epoch+1], label='Validation Loss')
                            ax1.set_title('Loss vs. Epochs')
                            ax1.set_xlabel('Epoch')
                            ax1.set_ylabel('Loss')
                            ax1.legend()
                            
                            # Metric plot
                            if st.session_state['task_type'] == 'classification':
                                ax2.plot(training_callback.accuracies[:epoch+1], label='Training Accuracy')
                                ax2.plot(training_callback.val_accuracies[:epoch+1], label='Validation Accuracy')
                                ax2.set_title('Accuracy vs. Epochs')
                                ax2.set_ylabel('Accuracy')
                            else:
                                ax2.plot(training_callback.losses[:epoch+1], label='Training MAE')
                                ax2.plot(training_callback.val_losses[:epoch+1], label='Validation MAE')
                                ax2.set_title('MAE vs. Epochs')
                                ax2.set_ylabel('MAE')
                            
                            ax2.set_xlabel('Epoch')
                            ax2.legend()
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                
                # Train model
                history = model.fit(
                    X_train_scaled, y_train,
                    validation_data=(X_val_scaled, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[training_callback, ProgressCallback()]
                )
                
                # Store model and related info in session state
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                
                # Final evaluation
                st.subheader("Model Evaluation")
                loss, metric = model.evaluate(X_test_scaled, y_test, verbose=0)
                
                if st.session_state['task_type'] == 'classification':
                    st.write(f"Test Loss: {loss:.4f}")
                    st.write(f"Test Accuracy: {metric:.4f}")
                    
                    # For classification, show confusion matrix
                    y_pred = model.predict(X_test_scaled)
                    if st.session_state['n_classes'] == 2:
                        y_pred_classes = (y_pred > 0.5).astype(int).flatten()
                    else:
                        y_pred_classes = np.argmax(y_pred, axis=1)
                    
                    from sklearn.metrics import confusion_matrix, classification_report
                    cm = confusion_matrix(y_test, y_pred_classes)
                    
                    st.subheader("Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    cax = ax.matshow(cm, cmap='Blues')
                    fig.colorbar(cax)
                    
                    # Set labels
                    if st.session_state['class_names']:
                        classes = st.session_state['class_names']
                        ax.set_xticklabels([''] + classes)
                        ax.set_yticklabels([''] + classes)
                    
                    # Add text annotations
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
                    
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('True')
                    st.pyplot(fig)
                    
                    # Classification report
                    if st.session_state['class_names']:
                        report = classification_report(y_test, y_pred_classes, target_names=st.session_state['class_names'])
                    else:
                        report = classification_report(y_test, y_pred_classes)
                    
                    st.text("Classification Report:")
                    st.text(report)
                    
                else:  # Regression
                    st.write(f"Test Loss (MSE): {loss:.4f}")
                    st.write(f"Test MAE: {metric:.4f}")
                    
                    # For regression, show predictions vs actual
                    y_pred = model.predict(X_test_scaled).flatten()
                    
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.scatter(y_test, y_pred, alpha=0.5)
                    
                    # Add perfect prediction line
                    min_val = min(np.min(y_test), np.min(y_pred))
                    max_val = max(np.max(y_test), np.max(y_pred))
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                    
                    ax.set_xlabel('Actual')
                    ax.set_ylabel('Predicted')
                    ax.set_title('Predicted vs Actual Values')
                    st.pyplot(fig)
                
                # Make model available for predictions
                st.session_state['model_trained'] = True
                
            # Model prediction (only show if model is trained)
            if 'model_trained' in st.session_state and st.session_state['model_trained']:
                st.header("4. Make Predictions")
                st.subheader("Enter values for prediction")
                
                # Create input fields for each feature
                input_data = {}
                for feature in st.session_state['features']:
                    if feature in df.columns:
                        # Get the mean value from the original dataset for default
                        default_value = float(df[feature].mean())
                        input_data[feature] = st.number_input(f"Enter value for {feature}", value=default_value)
                
                if st.button("Predict"):
                    # Convert input data to dataframe
                    input_df = pd.DataFrame([input_data])
                    
                    # Scale input data
                    input_scaled = st.session_state['scaler'].transform(input_df)
                    
                    # Make prediction
                    prediction = st.session_state['model'].predict(input_scaled)
                    
                    # Display prediction
                    st.subheader("Prediction Result")
                    
                    if st.session_state['task_type'] == 'classification':
                        if st.session_state['n_classes'] == 2:
                            pred_class = 1 if prediction[0][0] > 0.5 else 0
                            pred_prob = prediction[0][0] if pred_class == 1 else 1 - prediction[0][0]
                            
                            if st.session_state['class_names']:
                                pred_class_name = st.session_state['class_names'][pred_class]
                                st.markdown(f"Predicted Class: **{pred_class_name}**")
                            else:
                                st.markdown(f"Predicted Class: **{pred_class}**")
                            
                            st.markdown(f"Confidence: **{pred_prob:.4f}**")
                            
                            # Visualize with gauge
                            import plotly.graph_objects as go
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = float(pred_prob),
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Prediction Confidence"},
                                gauge = {
                                    'axis': {'range': [0, 1]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 0.33], 'color': "lightgray"},
                                        {'range': [0.33, 0.66], 'color': "gray"},
                                        {'range': [0.66, 1], 'color': "darkgray"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 0.5
                                    }
                                }
                            ))
                            st.plotly_chart(fig)
                            
                        else:  # Multi-class
                            pred_class = np.argmax(prediction[0])
                            pred_prob = prediction[0][pred_class]
                            
                            if st.session_state['class_names']:
                                pred_class_name = st.session_state['class_names'][pred_class]
                                st.markdown(f"Predicted Class: **{pred_class_name}**")
                            else:
                                st.markdown(f"Predicted Class: **{pred_class}**")
                            
                            st.markdown(f"Confidence: **{pred_prob:.4f}**")
                            
                            # Visualize class probabilities
                            if st.session_state['class_names']:
                                labels = st.session_state['class_names']
                            else:
                                labels = [f"Class {i}" for i in range(st.session_state['n_classes'])]
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.bar(labels, prediction[0])
                            ax.set_xlabel('Class')
                            ax.set_ylabel('Probability')
                            ax.set_title('Class Probabilities')
                            plt.xticks(rotation=45, ha='right')
                            st.pyplot(fig)
                            
                    else:  # Regression
                        st.markdown(f"Predicted {st.session_state['target']}: **{prediction[0][0]:.4f}**")
                        
                        # Add context
                        mean_target = df[st.session_state['target']].mean()
                        std_target = df[st.session_state['target']].std()
                        
                        if prediction[0][0] > mean_target + std_target:
                            st.info(f"This prediction is significantly above average")
                        elif prediction[0][0] < mean_target - std_target:
                            st.info(f"This prediction is significantly below average")
                        else:
                            st.info(f"This prediction is within one standard deviation of the average")
        
    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")
else:
    st.info("Please upload a CSV file to begin.")

# Add info about neural networks
st.sidebar.header("Instructions")
st.sidebar.markdown("""
### How to use:
1. Upload a CSV file with numeric data
2. Select target variable and features
3. Configure your neural network
4. Train the model
5. Make predictions with custom inputs

### Example datasets:
- MNIST digits dataset
- Iris flower dataset
- Wine quality dataset
""")

# Add explanation of neural networks
st.sidebar.header("What is a Neural Network?")
st.sidebar.markdown("""
Neural networks are computing systems inspired by the human brain. They can learn to perform tasks by considering examples, without being explicitly programmed.

Key components:
- Input layer: Receives the input data
- Hidden layers: Process the data
- Output layer: Produces the final result
- Activation functions: Introduce non-linearity
- Weights & biases: Parameters learned during training
""")
