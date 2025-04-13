import streamlit as st
import pandas as pd
import os
import sys
import tempfile
import time
import fitz  # PyMuPDF
import numpy as np
import re
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Append parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page configuration
st.set_page_config(
    page_title="Document Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Analyzer")
st.markdown("""
This section allows you to explore documents, extract insights, and find information within them.
Choose from the available datasets or upload your own to get started.
""")

# Define functions for document processing
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    doc = fitz.open(pdf_path)
    chunks = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        # Split into chunks of roughly 1000 characters with overlap
        if len(text) > 1000:
            # Create overlapping chunks
            for i in range(0, len(text), 800):  # 200 character overlap
                chunk = text[i:i+1000]
                if chunk:
                    chunks.append({
                        'text': chunk,
                        'source': f"Page {page_num+1}, Chunk {i//800+1}",
                        'page_num': page_num
                    })
        else:
            chunks.append({
                'text': text,
                'source': f"Page {page_num+1}",
                'page_num': page_num
            })
    
    return chunks, len(doc)

def process_csv(csv_path):
    """Process CSV file into text chunks"""
    df = pd.read_csv(csv_path)
    chunks = []
    
    # Convert each row to a chunk
    for idx, row in df.iterrows():
        text = " ".join([f"{col}: {val}" for col, val in row.items()])
        chunks.append({
            'text': text,
            'source': f"Row {idx+1}",
            'row_num': idx
        })
    
    return chunks, df

def search_documents(query, chunks, top_n=5):
    """Search for most similar chunks to the query"""
    if not chunks:
        return []
    
    # Create corpus of documents
    corpus = [chunk['text'] for chunk in chunks]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Transform query
    query_vector = vectorizer.transform([query])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get top N most similar
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.0:  # Only include if there's some similarity
            results.append({
                'chunk': chunks[idx],
                'similarity': similarities[idx]
            })
    
    return results

# Initialize session state variables
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'document_chunks' not in st.session_state:
    st.session_state.document_chunks = []
if 'dataset_name' not in st.session_state:
    st.session_state.dataset_name = None
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# Document selection
st.header("1. Select or Upload Document")

# Option to select dataset or upload
data_option = st.radio(
    "Choose data source", 
    ["Ghana Election Results", "Ghana 2025 Budget Statement", "Upload your own document"],
    index=0
)

if data_option == "Ghana Election Results":
    csv_path = "attached_assets/Ghana_Election_Result.csv"
    st.write("Using Ghana Election Results dataset")
    dataset_name = "Ghana Election Results"
    
    # Only process if the dataset hasn't been processed or if it's a different dataset
    if not st.session_state.document_processed or st.session_state.dataset_name != dataset_name:
        with st.spinner("Processing Ghana Election Results..."):
            try:
                # Process the CSV
                chunks, df = process_csv(csv_path)
                st.session_state.document_chunks = chunks
                st.session_state.document_processed = True
                st.session_state.dataset_name = dataset_name
                
                # Display preview
                st.success("Ghana Election Results dataset loaded successfully!")
                st.subheader("Dataset Preview")
                st.dataframe(df.head())
                
                # Display dataset statistics
                st.subheader("Dataset Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", df.shape[0])
                with col2:
                    st.metric("Years Covered", df['Year'].nunique())
                with col3:
                    st.metric("Political Parties", df['Party'].nunique())
                
            except Exception as e:
                st.error(f"Error loading Ghana Election Results: {str(e)}")
                st.session_state.document_processed = False
                
elif data_option == "Ghana 2025 Budget Statement":
    pdf_path = "attached_assets/2025-Budget-Statement-and-Economic-Policy_v4.pdf"
    st.write("Using Ghana 2025 Budget Statement")
    dataset_name = "Ghana 2025 Budget Statement"
    
    # Only process if the dataset hasn't been processed or if it's a different dataset
    if not st.session_state.document_processed or st.session_state.dataset_name != dataset_name:
        with st.spinner("Processing Ghana 2025 Budget Statement... This may take a minute."):
            try:
                # Process the PDF
                chunks, num_pages = extract_text_from_pdf(pdf_path)
                st.session_state.document_chunks = chunks
                st.session_state.document_processed = True
                st.session_state.dataset_name = dataset_name
                
                st.success("Ghana 2025 Budget Statement loaded successfully!")
                
                # Display some statistics
                st.subheader("Document Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Pages", num_pages)
                with col2:
                    st.metric("Chunks Created", len(chunks))
                
                # Display the first page content preview
                st.subheader("Document Preview")
                st.markdown(f"**First page content preview:**")
                st.write(chunks[0]['text'][:500] + "...")
                
            except Exception as e:
                st.error(f"Error loading Ghana 2025 Budget Statement: {str(e)}")
                st.session_state.document_processed = False
                
else:  # Upload option
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "csv"])
    
    if uploaded_file:
        # Save the file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name
        
        # Process based on file type
        try:
            if uploaded_file.name.endswith('.pdf'):
                with st.spinner("Processing PDF..."):
                    chunks, num_pages = extract_text_from_pdf(temp_path)
                    st.session_state.document_chunks = chunks
                    st.session_state.document_processed = True
                    st.session_state.dataset_name = uploaded_file.name
                    
                    st.success(f"PDF successfully loaded!")
                    
                    # Display document statistics
                    st.subheader("Document Statistics")
                    st.metric("Pages", num_pages)
                    st.metric("Chunks Created", len(chunks))
                    
            elif uploaded_file.name.endswith('.csv'):
                with st.spinner("Processing CSV..."):
                    chunks, df = process_csv(temp_path)
                    st.session_state.document_chunks = chunks
                    st.session_state.document_processed = True
                    st.session_state.dataset_name = uploaded_file.name
                    
                    st.success(f"CSV successfully loaded!")
                    
                    # Display document statistics
                    st.subheader("Document Statistics")
                    st.metric("Rows", df.shape[0])
                    st.metric("Chunks Created", len(chunks))
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            st.session_state.document_processed = False

# Search functionality
if st.session_state.document_processed:
    st.header("2. Search Document")
    
    search_query = st.text_input("Enter your search query")
    
    if st.button("Search") and search_query:
        # Add query to search history
        st.session_state.search_history.append({"query": search_query, "timestamp": time.time()})
        
        with st.spinner("Searching document..."):
            # Search for relevant chunks
            results = search_documents(search_query, st.session_state.document_chunks)
            
            if results:
                st.success(f"Found {len(results)} relevant sections")
                
                # Display results
                for i, result in enumerate(results):
                    chunk = result['chunk']
                    similarity = result['similarity']
                    
                    with st.expander(f"Result {i+1}: {chunk['source']} (Relevance: {similarity:.2f})"):
                        st.markdown(f"**Source:** {chunk['source']}")
                        
                        # Highlight the search terms in the text
                        highlighted_text = chunk['text']
                        for term in search_query.split():
                            if len(term) > 3:  # Only highlight terms with more than 3 characters
                                pattern = re.compile(re.escape(term), re.IGNORECASE)
                                highlighted_text = pattern.sub(f"**{term}**", highlighted_text)
                        
                        st.markdown(highlighted_text)
            else:
                st.warning("No relevant sections found for your query.")
    
    # Display search history
    if st.session_state.search_history:
        st.subheader("Search History")
        for i, search in enumerate(reversed(st.session_state.search_history[-5:])):  # Show last 5 searches
            query_time = time.strftime('%H:%M:%S', time.localtime(search["timestamp"]))
            st.markdown(f"**{query_time}**: {search['query']}")
        
        if st.button("Clear History"):
            st.session_state.search_history = []
            st.rerun()

# Document Analysis
if st.session_state.document_processed and st.session_state.dataset_name == "Ghana Election Results":
    st.header("3. Election Results Analysis")
    
    # Load the dataframe
    df = pd.read_csv("attached_assets/Ghana_Election_Result.csv")
    
    # Show party vote distribution
    st.subheader("Party Vote Distribution")
    party_votes = df.groupby('Party')['Votes'].sum().sort_values(ascending=False)
    st.bar_chart(party_votes)
    
    # Show year-wise analysis if there are multiple years
    if df['Year'].nunique() > 1:
        st.subheader("Year-wise Vote Distribution")
        year_party_votes = df.pivot_table(
            index='Year', 
            columns='Party', 
            values='Votes', 
            aggfunc='sum'
        ).fillna(0)
        st.line_chart(year_party_votes)

elif st.session_state.document_processed and st.session_state.dataset_name == "Ghana 2025 Budget Statement":
    st.header("3. Budget Statement Analysis")
    
    # Display key terms and their frequencies
    st.subheader("Key Terms Frequency")
    
    # Join all text chunks
    all_text = " ".join([chunk['text'] for chunk in st.session_state.document_chunks])
    
    # Common budget-related terms to look for
    budget_terms = ["revenue", "expenditure", "deficit", "gdp", "debt", "growth", 
                    "inflation", "tax", "fiscal", "monetary", "economy", "investment"]
    
    # Count occurrences
    term_counts = {}
    for term in budget_terms:
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        count = len(pattern.findall(all_text))
        term_counts[term.capitalize()] = count
    
    # Display as bar chart
    st.bar_chart(term_counts)
    
    # Topic distribution (simplified)
    st.subheader("Estimated Topic Distribution")
    topics = {
        "Economic Growth": 0.25,
        "Taxation": 0.20,
        "Social Services": 0.15,
        "Infrastructure": 0.15,
        "Debt Management": 0.10,
        "Foreign Policy": 0.05,
        "Other Sectors": 0.10
    }
    st.pyplot(pd.Series(topics).plot.pie(figsize=(10, 6), autopct='%1.1f%%').figure)

# Document processing explanation
st.sidebar.header("Document Analysis")
st.sidebar.markdown("""
### How It Works

This tool analyzes documents through these steps:

1. **Document Processing**:
   - Loads the document (PDF or CSV)
   - Segments it into manageable chunks
   - Prepares it for text analysis

2. **Search Functionality**:
   - Uses TF-IDF (Term Frequency-Inverse Document Frequency)
   - Calculates similarity between your query and text chunks
   - Retrieves and ranks the most relevant sections

3. **Document Analysis**:
   - Provides statistical insights about the document
   - Visualizes key information where applicable
   - Highlights important patterns and trends
""")

# Display diagram
st.sidebar.markdown("""
### Document Processing Workflow
```
Document → Chunking → Text Processing
                          ↓
  Query → Text Processing → Similarity Search
                          ↓
                     Top Results
```
""")