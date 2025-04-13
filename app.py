import streamlit as st
import os

st.set_page_config(
    page_title="AI Learning Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page content
st.title("🧠 AI Learning Platform")
st.markdown("""
## Welcome to the AI Learning Platform
This interactive application allows you to explore and solve diverse machine learning and AI problems including:

- **Regression**: Predict continuous variables using linear regression
- **Clustering**: Group data points using K-means clustering
- **Neural Networks**: Train neural networks for classification tasks
- **Large Language Models**: Ask questions about documents using state-of-the-art LLMs

### Instructions
- Navigate to different sections using the sidebar
- Follow the prompts in each section to upload data and configure models
- Explore the visualizations and results

### Created by:
This application was developed as part of the CS4241-Introduction to Artificial Intelligence course.
""")

# Display the datasets used
st.header("Datasets")
st.markdown("""
- **Ghana Election Results**: Contains voting data from Ghana's elections
- **Ghana 2025 Budget Statement**: Contains the budget statement and economic policy for Ghana in 2025
""")

# Add images or visualizations that showcase AI concepts
st.header("AI Concepts Demonstrated")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Machine Learning Pipeline")
    st.markdown("""
    1. Data Collection
    2. Data Preprocessing
    3. Model Selection
    4. Model Training
    5. Model Evaluation
    6. Model Deployment
    """)

with col2:
    st.subheader("AI Applications")
    st.markdown("""
    - Predictive Analytics
    - Pattern Recognition
    - Natural Language Processing
    - Computer Vision
    - Decision Making
    """)

# Add footer
st.markdown("---")
st.markdown("*AI Learning Platform © 2025*")
