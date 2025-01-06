import logging
import os
import datetime
from typing import Optional

def setup_logging(level: Optional[str] = None):
    """Set up logging configuration with configurable level.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, defaults to INFO.

    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Convert string level to logging constant
    if level is None:
        level = "INFO"
    
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure logging
    logging.basicConfig(
        filename=f'logs/gen-{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True  # This ensures the configuration is applied even if logging was previously configured
    )
    
    # Create a logger instance
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized with level: {level}")
