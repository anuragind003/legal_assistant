import streamlit as st
import os
import time
from typing import Dict, List
import tempfile
from src.graphs.workflow import LegalWorkflow
from src.utils.document_loader import DocumentLoader
from src.data.vector_store import VectorStore
from google.api_core import exceptions as google_exceptions

# Initialize components
workflow = LegalWorkflow()
document_loader = DocumentLoader()
vector_store = VectorStore()

# Set page configuration
st.set_page_config(
    page_title="Legal RAG System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for document uploading and settings
with st.sidebar:
    st.title("⚖️ Legal RAG System")
    st.markdown("---")
    
    # Document upload section
    st.header("Document Management")
    
    upload_type = st.radio("Upload method:", ["Upload Files", "Specify Directory"])
    
    if upload_type == "Upload Files":
        uploaded_files = st.file_uploader("Upload legal documents", 
                                        accept_multiple_files=True,
                                        type=["pdf", "txt", "docx", "csv"])
        
        if uploaded_files and st.button("Process Uploaded Documents"):
            with st.spinner("Processing documents..."):
                # Create a temporary directory to save the uploaded files
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save uploaded files to temp directory
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        file_paths.append(file_path)
                    
                    # Process each file
                    all_docs = []
                    for file_path in file_paths:
                        docs = document_loader.load_file(file_path)
                        all_docs.extend(docs)
                    
                    # Add to vector store
                    if all_docs:
                        vector_store.add_documents(all_docs)
                        st.success(f"Successfully added {len(all_docs)} document chunks to the vector store!")
                    else:
                        st.error("No documents were processed. Please check the file formats.")
    
    else:  # Specify Directory
        directory_path = st.text_input("Enter directory path containing legal documents:")
        
        if directory_path and st.button("Process Directory"):
            if not os.path.exists(directory_path):
                st.error(f"Directory {directory_path} does not exist!")
            else:
                with st.spinner("Processing documents from directory..."):
                    docs = document_loader.load_directory(directory_path)
                    
                    if docs:
                        vector_store.add_documents(docs)
                        st.success(f"Successfully added {len(docs)} document chunks to the vector store!")
                    else:
                        st.error("No documents were processed. Please check the directory content.")
    
    # Vector store stats
    st.markdown("---")
    st.header("Vector Store Stats")
    
    if st.button("Refresh Stats"):
        try:
            stats = vector_store.get_collection_stats()
            st.write(f"Collection: {stats['name']}")
            st.write(f"Document count: {stats['count']}")
        except Exception as e:
            st.error(f"Error getting vector store stats: {str(e)}")
    
    # Clear vector store option
    if st.button("Clear Vector Store", type="primary"):
        try:
            vector_store.delete_collection()
            st.success("Vector store collection deleted!")
        except Exception as e:
            st.error(f"Error deleting collection: {str(e)}")

    st.markdown("---")
    st.header("Display Settings")
    st.session_state.show_metadata = st.toggle("Show Response Metadata", value=False)

# Main content area
st.title("Legal Research Assistant")
st.markdown("""
Ask any legal question, and the system will research it using both your uploaded documents 
and relevant web sources if needed.
""")

# Initialize chat history and metadata visibility state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_metadata" not in st.session_state:
    st.session_state.show_metadata = False

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and isinstance(message["content"], dict):
            st.markdown(message["content"]["answer"])
            if st.session_state.show_metadata and "metadata" in message["content"]:
                st.markdown(message["content"]["metadata"], unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a legal question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            # Format chat history for the model
            chat_history = [
                {"role": msg["role"], "content": msg["content"]} 
                for msg in st.session_state.messages[:-1]  # Exclude the current query
            ]
            
            # Process the query through the workflow
            response_container = st.empty()
            response_container.markdown("Researching your legal question...")
            
            # Process the query
            try:
                result = workflow.process_query(prompt)
            except google_exceptions.NotFound as e:
                print(f"Error processing query with Gemini model: {e}")
                print("Please check your Google API configuration and model availability")
                raise
            
            # Format the response
            answer = result["answer"]
            
            # Add references if available
            if result["references"]:
                answer += "\n\n**References:**\n"
                for ref in result["references"]:
                    answer += f"- {ref}\n"
            
            # Format metadata
            confidence = result.get("confidence", 0.0)
            confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
            search_performed = result.get("search_performed", False)
            docs_retrieved = bool(result.get("references", []))
            
            metadata = f"""
            <div style="font-size: 0.8em; color: gray; margin-top: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; display: {'block' if st.session_state.show_metadata else 'none'};">
                <h4 style="margin: 0 0 10px 0;">Response Metadata</h4>
                <p>Confidence: <span style="color: {confidence_color};">{confidence:.2f}</span></p>
                <p>Search performed: {"Yes" if search_performed else "No"}</p>
                <p>Documents retrieved: {"Yes" if docs_retrieved else "No"}</p>
                <p>Number of references: {len(result.get("references", []))}</p>
                <p>Response length: {len(answer)} characters</p>
                <p>Processing time: {result.get("processing_time", "N/A")} seconds</p>
            </div>
            """
            
            # Store answer and metadata separately
            message_content = {
                "answer": answer,
                "metadata": metadata
            }
            
            # Display the answer first
            response_container.markdown(answer)
            
            # Only show metadata if toggle is enabled
            if st.session_state.show_metadata:
                response_container.markdown(metadata, unsafe_allow_html=True)
            
            # Store in chat history with structured content
            st.session_state.messages.append({"role": "assistant", "content": message_content})

# Add footer
st.markdown("""
---
<div style="text-align: center; color: gray; font-size: 0.8em;">
    Legal RAG System powered by LangGraph, LangChain, Gemini, and Chroma DB
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    # This will only execute when the script is run directly
    pass