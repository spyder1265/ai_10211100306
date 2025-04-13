import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_regression_results(y_true, y_pred, target_name):
    """
    Plot regression results: scatter plot of true vs predicted values.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True target values
    y_pred : numpy.ndarray
        Predicted target values
    target_name : str
        Name of the target variable
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    
    # Add labels and title
    ax.set_xlabel(f'True {target_name}')
    ax.set_ylabel(f'Predicted {target_name}')
    ax.set_title(f'True vs Predicted {target_name}')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig

def plot_clusters(X, clusters, feature_names=None, method='pca'):
    """
    Plot clusters in 2D space using dimensionality reduction.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    clusters : numpy.ndarray
        Cluster assignments
    feature_names : list
        Names of the features
    method : str
        Dimensionality reduction method: 'pca' or 'tsne'
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with the plot
    """
    # Apply dimensionality reduction
    if X.shape[1] > 2:
        if method == 'pca':
            reducer = PCA(n_components=2)
            title_prefix = 'PCA'
        else:  # tsne
            reducer = TSNE(n_components=2, random_state=42)
            title_prefix = 't-SNE'
        
        X_2d = reducer.fit_transform(X)
        
        # Get axis labels
        if feature_names and method == 'pca':
            x_label = f"{title_prefix} 1"
            y_label = f"{title_prefix} 2"
        else:
            x_label = f"{title_prefix} Dimension 1"
            y_label = f"{title_prefix} Dimension 2"
    else:
        X_2d = X
        if feature_names:
            x_label = feature_names[0]
            y_label = feature_names[1]
        else:
            x_label = "Feature 1"
            y_label = "Feature 2"
        title_prefix = "Original"
    
    # Plot clusters
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    
    # Add labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'{title_prefix} Cluster Visualization')
    
    # Add legend
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_elbow(k_range, inertias):
    """
    Plot elbow curve for K-means.
    
    Parameters:
    -----------
    k_range : range
        Range of k values
    inertias : list
        Inertia values for each k
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot elbow curve
    ax.plot(k_range, inertias, 'bo-')
    
    # Add labels and title
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method for Optimal k')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add xticks for each cluster number
    ax.set_xticks(list(k_range))
    
    return fig

def plot_training_history(history):
    """
    Plot training history for neural network.
    
    Parameters:
    -----------
    history : tensorflow.keras.callbacks.History
        Training history
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with the plots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training & validation loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot training & validation accuracy if it exists
    if 'accuracy' in history.history or 'acc' in history.history:
        acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
        val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
        
        ax2.plot(history.history[acc_key], label='Training Accuracy')
        ax2.plot(history.history[val_acc_key], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    elif 'mae' in history.history:
        # For regression tasks with MAE metric
        ax2.plot(history.history['mae'], label='Training MAE')
        ax2.plot(history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_confusion_matrix(cm, class_names=None):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    cm : numpy.ndarray
        Confusion matrix
    class_names : list
        Names of the classes
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    
    # Set labels
    if class_names:
        ax.set_xticklabels([''] + class_names)
        ax.set_yticklabels([''] + class_names)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    return fig
