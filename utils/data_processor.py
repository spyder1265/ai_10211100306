import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(df, features, target, missing_method="Drop rows with missing values"):
    """
    Preprocess the dataframe for machine learning.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to preprocess
    features : list
        List of feature columns
    target : str or None
        Target column name, None for clustering
    missing_method : str
        Method to handle missing values
        
    Returns:
    --------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray or None
        Target vector, None if target is None
    df_processed : pandas.DataFrame
        Processed dataframe
    """
    # Make a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Select features
    if target:
        data = df_processed[features + [target]]
    else:
        data = df_processed[features]
    
    # Handle missing values
    if missing_method == "Drop rows with missing values":
        data = data.dropna()
    elif missing_method == "Fill missing values with mean":
        for col in features:
            if data[col].dtype in ['int64', 'float64']:
                data[col] = data[col].fillna(data[col].mean())
            else:
                data[col] = data[col].fillna(data[col].mode()[0])
    elif missing_method == "Fill missing values with median":
        for col in features:
            if data[col].dtype in ['int64', 'float64']:
                data[col] = data[col].fillna(data[col].median())
            else:
                data[col] = data[col].fillna(data[col].mode()[0])
    
    # Create feature matrix
    X = data[features].to_numpy()
    
    # Create target vector if target is provided
    if target:
        y = data[target].to_numpy()
    else:
        y = None
    
    return X, y, data

def encode_categorical_features(df, categorical_features):
    """
    Encode categorical features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with categorical features
    categorical_features : list
        List of categorical column names
        
    Returns:
    --------
    df_encoded : pandas.DataFrame
        Dataframe with encoded categorical features
    encoders : dict
        Dictionary of label encoders for each categorical feature
    """
    df_encoded = df.copy()
    encoders = {}
    
    for col in categorical_features:
        if col in df.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    return df_encoded, encoders

def scale_features(X):
    """
    Scale features using StandardScaler.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
        
    Returns:
    --------
    X_scaled : numpy.ndarray
        Scaled feature matrix
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def split_time_series(X, y, test_size=0.2, val_size=0.2):
    """
    Split time series data into train, validation, and test sets.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    test_size : float
        Fraction of data for testing
    val_size : float
        Fraction of data for validation
        
    Returns:
    --------
    X_train, X_val, X_test : numpy.ndarray
        Feature matrices for train, validation, and test sets
    y_train, y_val, y_test : numpy.ndarray
        Target vectors for train, validation, and test sets
    """
    n = len(X)
    test_idx = int(n * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))
    
    X_train, y_train = X[:val_idx], y[:val_idx]
    X_val, y_val = X[val_idx:test_idx], y[val_idx:test_idx]
    X_test, y_test = X[test_idx:], y[test_idx:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test
