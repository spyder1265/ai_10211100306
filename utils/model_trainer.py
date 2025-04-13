from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def train_regression_model(X_train, y_train, X_test, y_test):
    """
    Train a linear regression model and evaluate it.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training feature matrix
    y_train : numpy.ndarray
        Training target vector
    X_test : numpy.ndarray
        Testing feature matrix
    y_test : numpy.ndarray
        Testing target vector
        
    Returns:
    --------
    model : sklearn.linear_model.LinearRegression
        Trained regression model
    y_pred : numpy.ndarray
        Predicted values for test set
    metrics : dict
        Dictionary of evaluation metrics
    """
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Collect metrics
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }
    
    return model, y_pred, metrics

def train_kmeans_model(X, n_clusters):
    """
    Train a K-means clustering model.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    n_clusters : int
        Number of clusters to form
        
    Returns:
    --------
    model : sklearn.cluster.KMeans
        Trained K-means model
    clusters : numpy.ndarray
        Cluster assignments for each data point
    metrics : dict
        Dictionary of evaluation metrics (inertia, silhouette score)
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    # Initialize and train the model
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = model.fit_predict(X)
    
    # Calculate metrics
    inertia = model.inertia_
    
    # Only calculate silhouette score if there's more than one cluster
    if n_clusters > 1:
        silhouette = silhouette_score(X, clusters)
    else:
        silhouette = np.nan
    
    # Collect metrics
    metrics = {
        'inertia': inertia,
        'silhouette_score': silhouette
    }
    
    return model, clusters, metrics

def train_neural_network(X_train, y_train, X_val, y_val, layer_sizes, 
                         activation='relu', dropout_rate=0.2, 
                         learning_rate=0.01, epochs=50, batch_size=32, 
                         task_type='classification', num_classes=None):
    """
    Train a neural network model using TensorFlow.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training feature matrix
    y_train : numpy.ndarray
        Training target vector
    X_val : numpy.ndarray
        Validation feature matrix
    y_val : numpy.ndarray
        Validation target vector
    layer_sizes : list
        List of integers representing the size of each hidden layer
    activation : str
        Activation function to use in hidden layers
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Learning rate for the optimizer
    epochs : int
        Number of epochs to train for
    batch_size : int
        Batch size for training
    task_type : str
        'classification' or 'regression'
    num_classes : int
        Number of classes for classification tasks
        
    Returns:
    --------
    model : tensorflow.keras.Model
        Trained neural network model
    history : tensorflow.keras.callbacks.History
        Training history
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    import numpy as np
    
    # Build the model
    model = Sequential()
    
    # Input layer
    model.add(Dense(layer_sizes[0], activation=activation, input_shape=(X_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    
    # Hidden layers
    for units in layer_sizes[1:]:
        model.add(Dense(units, activation=activation))
        model.add(Dropout(dropout_rate))
    
    # Output layer
    if task_type == 'classification':
        if num_classes == 2:  # Binary classification
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:  # Multi-class classification
            model.add(Dense(num_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    else:  # Regression
        model.add(Dense(1, activation='linear'))
        loss = 'mse'
        metrics = ['mae']
    
    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    
    return model, history
