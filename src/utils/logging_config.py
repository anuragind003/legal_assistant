import logging
import os
from pathlib import Path

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / "legal_rag.log"),
        logging.StreamHandler()
    ]
)

# Create logger
logger = logging.getLogger("legal_rag")

__all__ = ["logger"]
