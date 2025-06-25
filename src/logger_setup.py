"""
Logger Setup Module

Configures logging for the T-Rex reconciliation tool with appropriate formatting,
handlers, and log levels for both console and file output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration for T-Rex application.
    
    Configures both console and optional file logging with appropriate formatting.
    The logger captures execution steps, warnings, errors, and debug information.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional path to log file. If None, only console logging is used
        
    Returns:
        None
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter for log messages
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger and clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(numeric_level)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Setup file handler if log file is specified
    if log_file:
        try:
            # Ensure log directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create file handler
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            logging.info(f"File logging enabled: {log_file}")
            
        except Exception as e:
            logging.warning(f"Could not setup file logging: {e}")
    
    # Log initial setup information
    logging.info(f"Logging initialized at {log_level} level")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    This is a convenience function to get properly configured logger instances
    for different modules within the T-Rex application.
    
    Args:
        name: Name for the logger (typically __name__ of the calling module)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """
    Mixin class to provide logging capabilities to other classes.
    
    This mixin provides a convenient way to add logging to any class
    by inheriting from this mixin. The logger name will be based on
    the class's module and name.
    """
    
    @property
    def logger(self) -> logging.Logger:
        """
        Get logger instance for this class.
        
        Returns:
            logging.Logger: Logger instance named after the class
        """
        if not hasattr(self, '_logger'):
            class_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self._logger = logging.getLogger(class_name)
        return self._logger
