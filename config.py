import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LLM Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-1.5-flash-latest"

# Search API Configuration
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Vector DB Configuration
CHROMA_PERSIST_DIRECTORY = "./chroma_db"

# Document Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Agent Configuration
MAX_SEARCH_ITERATIONS = 3
MAX_DOCUMENTS_TO_RETRIEVE = 5
SEARCH_CONFIDENCE_THRESHOLD = 0.7