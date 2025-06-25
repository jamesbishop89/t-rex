"""
Unit tests for logger setup module.
"""

import pytest
import logging
import tempfile
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.logger_setup import setup_logger, get_logger, LoggerMixin


class TestLoggerSetup:
    """Test cases for logger setup functions."""
    
    def test_setup_logger_default(self):
        """Test setting up logger with default parameters."""
        setup_logger()
        
        # Check that root logger is configured
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) > 0
    
    def test_setup_logger_debug_level(self):
        """Test setting up logger with DEBUG level."""
        setup_logger('DEBUG')
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
    
    def test_setup_logger_with_file(self, temp_dir):
        """Test setting up logger with file output."""
        log_file = temp_dir / "test.log"
        setup_logger('INFO', str(log_file))
        
        # Log a test message
        logger = logging.getLogger('test')
        logger.info("Test message")
        
        # Check that log file was created and contains message
        assert log_file.exists()
        log_content = log_file.read_text()
        assert "Test message" in log_content
    
    def test_setup_logger_invalid_level(self):
        """Test setting up logger with invalid level."""
        setup_logger('INVALID')
        
        # Should default to INFO level
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
    
    def test_get_logger(self):
        """Test getting logger instance."""
        logger = get_logger('test_module')
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'test_module'
    
    def test_logger_mixin(self):
        """Test LoggerMixin class."""
        class TestClass(LoggerMixin):
            def test_method(self):
                return self.logger
        
        test_obj = TestClass()
        logger = test_obj.test_method()
        
        assert isinstance(logger, logging.Logger)
        assert 'TestClass' in logger.name
    
    def test_logger_mixin_caching(self):
        """Test that LoggerMixin caches logger instance."""
        class TestClass(LoggerMixin):
            pass
        
        test_obj = TestClass()
        logger1 = test_obj.logger
        logger2 = test_obj.logger
        
        # Should be the same instance
        assert logger1 is logger2
    
    def test_setup_logger_file_creation_error(self):
        """Test logger setup when file creation fails."""
        # Try to create log file in non-existent directory without permission
        invalid_path = "/invalid/path/test.log"
        
        # Should not raise exception, just log warning
        setup_logger('INFO', invalid_path)
        
        # Logger should still be configured for console output
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0
    
    def test_multiple_setup_calls(self):
        """Test calling setup_logger multiple times."""
        # First setup
        setup_logger('INFO')
        handler_count_1 = len(logging.getLogger().handlers)
        
        # Second setup should clear previous handlers
        setup_logger('DEBUG')
        handler_count_2 = len(logging.getLogger().handlers)
        
        # Should have same number of handlers (old ones cleared)
        assert handler_count_1 == handler_count_2
        
        # But level should be updated
        assert logging.getLogger().level == logging.DEBUG
