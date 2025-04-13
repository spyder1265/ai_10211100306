# AI Learning Platform

This interactive Streamlit application allows users to explore and solve diverse machine learning and AI problems, including:

- **Regression**: Predict continuous variables using linear regression
- **Clustering**: Group data points using K-means clustering
- **Neural Networks**: Train neural networks for classification tasks
- **Large Language Models**: Ask questions about documents using state-of-the-art LLMs

## Features

### 1. Regression Module
- Upload datasets or use provided sample data
- Select features and target variables
- Configure and train linear regression models
- Visualize results with scatter plots and performance metrics
- Make predictions with custom inputs

### 2. Clustering Module
- Perform K-means clustering on datasets
- Visualize clusters in 2D/3D space
- Determine optimal cluster count using the elbow method
- Analyze feature distributions within clusters
- Download clustered data

### 3. Neural Network Module
- Train neural networks for classification or regression
- Configure network architecture (layers, neurons, activation functions)
- Visualize training progress in real-time
- Evaluate model performance with metrics and confusion matrices
- Make predictions on new data

### 4. LLM Question-Answering Module
- Retrieve information from documents using RAG (Retrieval-Augmented Generation)
- Analyze Ghana Election Results and Ghana 2025 Budget Statement
- Upload custom documents for analysis
- Ask questions in natural language
- Get contextually relevant answers from the system

## Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, TensorFlow
- **Visualization**: Matplotlib, Plotly
- **LLM Integration**: LangChain, HuggingFace Transformers
- **Vector Database**: FAISS

## Getting Started

1. Install the required dependencies:
   ```
   pip install streamlit pandas numpy scikit-learn tensorflow matplotlib plotly langchain faiss-cpu seaborn
   