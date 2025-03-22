import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base project paths
ROOT_DIR = Path(__file__).parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
CACHE_DIR = ROOT_DIR / "cache"
CHROMA_PERSIST_DIRECTORY = str(DATA_DIR / "chroma_db")

# Create necessary directories
for directory in [DATA_DIR, LOGS_DIR, CACHE_DIR, Path(CHROMA_PERSIST_DIRECTORY)]:
    directory.mkdir(parents=True, exist_ok=True)

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Model Configuration 
MODEL_NAME = "gemini-1.5-flash-latest"
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 2048

# Document Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Search Configuration
MAX_SEARCH_RESULTS = 5
SEARCH_TIMEOUT = 10
MAX_DOCUMENTS_TO_RETRIEVE = 5
SEARCH_CONFIDENCE_THRESHOLD = 0.7

# Define what to export
__all__ = [
    'ROOT_DIR',
    'SRC_DIR',
    'DATA_DIR',
    'LOGS_DIR',
    'CACHE_DIR',
    'CHROMA_PERSIST_DIRECTORY',
    'GOOGLE_API_KEY',
    'TAVILY_API_KEY',
    'MODEL_NAME',
    'TEMPERATURE',
    'MAX_OUTPUT_TOKENS',
    'CHUNK_SIZE',
    'CHUNK_OVERLAP',
    'MAX_SEARCH_RESULTS',
    'SEARCH_TIMEOUT',
    'MAX_DOCUMENTS_TO_RETRIEVE',
    'SEARCH_CONFIDENCE_THRESHOLD'
]
