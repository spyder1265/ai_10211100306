from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
import os
import torch

def process_document(documents):
    """
    Process documents for the RAG system.
    
    Parameters:
    -----------
    documents : list
        List of documents to process
    
    Returns:
    --------
    retriever : langchain.retrievers.Retriever
        The document retriever
    chunks : list
        List of processed document chunks
    """
    # Initialize a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    # Split the documents into chunks
    chunks = text_splitter.split_documents(documents)
    
    # Initialize the embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create a vector store from the chunks
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    return retriever, chunks

def get_llm_response(llm, retriever, query):
    """
    Get a response from the LLM using the RAG approach.
    
    Parameters:
    -----------
    llm : langchain.llms.LLM
        The LLM to use
    retriever : langchain.retrievers.Retriever
        The document retriever
    query : str
        The query to answer
    
    Returns:
    --------
    response : dict
        The LLM response
    docs : list
        The retrieved documents
    """
    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(query)
    
    # Create a template for the response
    from langchain.prompts import PromptTemplate
    template = """
    You are an AI assistant trained to answer questions based on the provided context.
    
    Context: {context}
    
    Question: {question}
    
    Please provide a detailed and accurate answer based solely on the information in the context.
    If the answer is not contained within the context, say "I don't have enough information to answer this question."
    
    Answer:
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    # Get the response
    response = qa_chain({"query": query})
    
    return response, docs
