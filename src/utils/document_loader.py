from langchain_community.document_loaders import (
    DirectoryLoader, 
    PyPDFLoader, 
    TextLoader,
    UnstructuredFileLoader,
    CSVLoader,
    Docx2txtLoader  # Changed from DocxLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self):
        """Initialize document loader with text splitter."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
    def load_directory(self, directory_path):
        """Load all supported documents from a directory."""
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory {directory_path} does not exist")
        
        loaders = []
        
        # PDF loader
        pdf_loader = DirectoryLoader(
            directory_path, 
            glob="**/*.pdf", 
            loader_cls=PyPDFLoader
        )
        loaders.append(pdf_loader)
        
        # Text loader
        text_loader = DirectoryLoader(
            directory_path, 
            glob="**/*.txt", 
            loader_cls=TextLoader
        )
        loaders.append(text_loader)
        
        # DOCX loader - Updated to use Docx2txtLoader
        docx_loader = DirectoryLoader(
            directory_path, 
            glob="**/*.docx", 
            loader_cls=Docx2txtLoader
        )
        loaders.append(docx_loader)
        
        # CSV loader
        csv_loader = DirectoryLoader(
            directory_path, 
            glob="**/*.csv", 
            loader_cls=CSVLoader
        )
        loaders.append(csv_loader)
        
        # Load documents
        all_documents = []
        for loader in loaders:
            try:
                documents = loader.load()
                logger.info(f"Loaded {len(documents)} documents using {loader.__class__.__name__}")
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error loading documents with {loader.__class__.__name__}: {e}")
        
        # Split documents
        if all_documents:
            return self.text_splitter.split_documents(all_documents)
        else:
            logger.warning(f"No documents were loaded from {directory_path}")
            return []
    
    def load_file(self, file_path):
        """Load a single file based on its extension."""
        if not os.path.exists(file_path):
            raise ValueError(f"File {file_path} does not exist")
        
        _, ext = os.path.splitext(file_path.lower())
        
        try:
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext == '.txt':
                loader = TextLoader(file_path)
            elif ext == '.docx':
                loader = Docx2txtLoader(file_path)  # Changed from DocxLoader
            elif ext == '.csv':
                loader = CSVLoader(file_path)
            else:
                # Try with unstructured for other file types
                loader = UnstructuredFileLoader(file_path)
            
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return self.text_splitter.split_documents(documents)
        
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []