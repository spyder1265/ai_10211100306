import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import sys
import os
import io

# Append parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions
from utils.data_processor import preprocess_data
from utils.visualizer import plot_clusters, plot_elbow

st.set_page_config(
    page_title="Clustering Analysis",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Clustering Analysis")
st.markdown("""
This section allows you to perform clustering on a dataset using K-Means algorithm.
Upload your dataset, select features for clustering, and visualize the clusters.
""")

# File upload
st.header("1. Upload Dataset")
uploaded_file = st.file_uploader("Upload a CSV file for clustering analysis", type=["csv"])

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
        
        # Select features for clustering
        st.subheader("Select Features for Clustering")
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        selected_features = st.multiselect("Select features to use for clustering", numeric_columns, numeric_columns[:3])
        
        # Handle missing values
        st.subheader("Handle Missing Values")
        missing_method = st.radio("Select method to handle missing values", 
                                ["Drop rows with missing values", 
                                 "Fill missing values with mean", 
                                 "Fill missing values with median"])
        
        # Apply preprocessing
        if st.button("Preprocess Data"):
            # Check if features are selected
            if not selected_features:
                st.error("Please select at least one feature.")
            else:
                # Preprocess data (using target=None for clustering)
                X, _, df_processed = preprocess_data(df, selected_features, None, missing_method)
                
                st.session_state['X'] = X
                st.session_state['df_processed'] = df_processed
                st.session_state['features'] = selected_features
                
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
                
                # Standardize the data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                st.session_state['X_scaled'] = X_scaled
                st.session_state['scaler'] = scaler
        
        # Clustering section (only show if data is preprocessed)
        if 'X_scaled' in st.session_state:
            st.header("3. K-Means Clustering")
            
            # Determine optimal number of clusters using elbow method
            st.subheader("Find Optimal Number of Clusters")
            max_clusters = min(10, len(st.session_state['X']) - 1)  # Limit to 10 or dataset size
            k_range = range(1, max_clusters + 1)
            
            inertias = []
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(st.session_state['X_scaled'])
                inertias.append(kmeans.inertia_)
            
            # Plot elbow curve
            fig = plot_elbow(k_range, inertias)
            st.pyplot(fig)
            
            # Select number of clusters
            st.subheader("Select Number of Clusters")
            n_clusters = st.slider("Number of clusters", 2, max_clusters, 3)
            
            # Run K-Means clustering
            if st.button("Run Clustering"):
                # Fit KMeans
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(st.session_state['X_scaled'])
                
                # Store clustering results
                st.session_state['clusters'] = clusters
                st.session_state['kmeans'] = kmeans
                st.session_state['n_clusters'] = n_clusters
                
                # Add cluster labels to the dataframe
                df_clustered = st.session_state['df_processed'].copy()
                df_clustered['Cluster'] = clusters
                st.session_state['df_clustered'] = df_clustered
                
                st.success(f"K-Means clustering completed with {n_clusters} clusters!")
                
                # Display cluster statistics
                st.subheader("Cluster Statistics")
                cluster_stats = df_clustered.groupby('Cluster').mean()
                st.write(cluster_stats)
                
                # Visualize clusters
                st.subheader("Cluster Visualization")
                
                # Use PCA to reduce to 2D if more than 2 features
                if len(st.session_state['features']) > 2:
                    st.write("Using PCA to visualize clusters in 2D")
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(st.session_state['X_scaled'])
                    
                    # Create a DataFrame for plotting
                    plot_df = pd.DataFrame({
                        'PC1': X_pca[:, 0],
                        'PC2': X_pca[:, 1],
                        'Cluster': clusters
                    })
                    
                    # Plot using plotly
                    fig = px.scatter(
                        plot_df, x='PC1', y='PC2', 
                        color='Cluster', 
                        title='Clusters Visualization (PCA)',
                        color_continuous_scale=px.colors.qualitative.G10
                    )
                    st.plotly_chart(fig)
                    
                    # Show variance explained by PCA
                    explained_variance = pca.explained_variance_ratio_
                    st.write(f"Variance explained by PC1: {explained_variance[0]*100:.2f}%")
                    st.write(f"Variance explained by PC2: {explained_variance[1]*100:.2f}%")
                    
                else:
                    # Direct 2D plot if only 2 features
                    plot_df = pd.DataFrame({
                        st.session_state['features'][0]: st.session_state['X_scaled'][:, 0],
                        st.session_state['features'][1]: st.session_state['X_scaled'][:, 1],
                        'Cluster': clusters
                    })
                    
                    fig = px.scatter(
                        plot_df, 
                        x=st.session_state['features'][0], 
                        y=st.session_state['features'][1], 
                        color='Cluster',
                        title='Clusters Visualization',
                        color_continuous_scale=px.colors.qualitative.G10
                    )
                    st.plotly_chart(fig)
                
                # Plot centroids
                st.subheader("Cluster Centroids")
                centroids = kmeans.cluster_centers_
                
                # Create a table of centroids
                if len(st.session_state['features']) > 2:
                    # For PCA visualization
                    centroids_pca = pca.transform(centroids)
                    centroids_df = pd.DataFrame(centroids_pca, columns=['PC1', 'PC2'])
                else:
                    # For direct 2D visualization
                    centroids_df = pd.DataFrame(
                        centroids, 
                        columns=st.session_state['features']
                    )
                
                centroids_df['Cluster'] = range(n_clusters)
                st.write(centroids_df)
                
                # Download clustered data
                st.subheader("Download Clustered Data")
                csv = df_clustered.to_csv(index=False)
                st.download_button(
                    label="Download clustered data as CSV",
                    data=csv,
                    file_name="clustered_data.csv",
                    mime="text/csv",
                )
                
                # Additional cluster analysis
                st.subheader("Cluster Distribution")
                cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(cluster_counts.index, cluster_counts.values)
                ax.set_xlabel('Cluster')
                ax.set_ylabel('Number of samples')
                ax.set_title('Number of samples in each cluster')
                ax.set_xticks(range(n_clusters))
                st.pyplot(fig)
                
                # Feature distribution by cluster
                st.subheader("Feature Distribution by Cluster")
                
                # Allow selecting a feature to analyze
                feature_to_analyze = st.selectbox(
                    "Select a feature to analyze distribution across clusters", 
                    st.session_state['features']
                )
                
                # Get the index of the selected feature
                feature_idx = st.session_state['features'].index(feature_to_analyze)
                
                # Box plot for the selected feature
                fig, ax = plt.subplots(figsize=(10, 6))
                for i in range(n_clusters):
                    cluster_data = st.session_state['X_scaled'][clusters == i, feature_idx]
                    ax.boxplot(cluster_data, positions=[i], widths=0.6)
                ax.set_xlabel('Cluster')
                ax.set_ylabel(f'Standardized {feature_to_analyze}')
                ax.set_title(f'Distribution of {feature_to_analyze} by Cluster')
                ax.set_xticks(range(n_clusters))
                st.pyplot(fig)
                
    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")
else:
    st.info("Please upload a CSV file to begin.")

# Add info about clustering
st.sidebar.header("Instructions")
st.sidebar.markdown("""
### How to use:
1. Upload a CSV file with numeric data
2. Select features for clustering
3. Preprocess the data
4. Use the elbow method to determine optimal clusters
5. Run K-Means clustering
6. Analyze the clustering results

### Example datasets:
- Customer segmentation data
- Market segmentation data
- Document clustering
""")

# Add explanation of K-Means
st.sidebar.header("What is K-Means Clustering?")
st.sidebar.markdown("""
K-Means is an unsupervised learning algorithm that groups data points into K clusters based on feature similarity.

The algorithm works by:
1. Randomly initializing K cluster centroids
2. Assigning each data point to the nearest centroid
3. Recalculating centroids based on assigned points
4. Repeating steps 2-3 until convergence
""")

