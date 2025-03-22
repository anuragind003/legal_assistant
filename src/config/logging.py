import logging
from pathlib import Path
from .config import LOGS_DIR

def setup_logging(name: str = "legal_rag") -> logging.Logger:
    """Set up logging with both file and console handlers."""
    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s [%(name)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Ensure logs directory exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create file handler
    file_handler = logging.FileHandler(
        LOGS_DIR / f"{name}.log",
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create default logger
logger = setup_logging()
