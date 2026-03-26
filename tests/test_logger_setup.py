"""
Unit tests for logger setup module.
"""

import pytest
import logging

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.logger_setup import setup_logger, get_logger, LoggerMixin


def _trex_handlers():
    """Return handlers managed by T-Rex logging setup."""
    return [h for h in logging.getLogger().handlers if getattr(h, '_t_rex_managed', False)]


@pytest.fixture(autouse=True)
def cleanup_trex_handlers():
    """Keep tests isolated without disturbing unrelated root handlers."""
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    yield

    for handler in list(root_logger.handlers):
        if handler not in original_handlers:
            root_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
    root_logger.setLevel(original_level)


class TestLoggerSetup:
    """Test cases for logger setup functions."""

    def test_setup_logger_default(self):
        """Test setting up logger with default parameters."""
        setup_logger()

        # Check that root logger is configured
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(_trex_handlers()) == 1

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
        assert any(isinstance(handler, logging.FileHandler) for handler in _trex_handlers())

        root_logger = logging.getLogger()
        for handler in list(_trex_handlers()):
            if isinstance(handler, logging.FileHandler):
                root_logger.removeHandler(handler)
                handler.close()

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
        def raise_file_error(*args, **kwargs):
            raise OSError("boom")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(logging, 'FileHandler', raise_file_error)

        # Should not raise exception, just log warning
        setup_logger('INFO', 'ignored.log')

        # Logger should still be configured for console output
        assert len(_trex_handlers()) == 1
        monkeypatch.undo()

    def test_multiple_setup_calls(self):
        """Test calling setup_logger multiple times."""
        # First setup
        setup_logger('INFO')
        handlers_1 = list(_trex_handlers())

        # Second setup should reuse managed handlers rather than duplicating them
        setup_logger('DEBUG')
        handlers_2 = list(_trex_handlers())

        assert len(handlers_1) == len(handlers_2) == 1
        assert handlers_1[0] is handlers_2[0]
        assert logging.getLogger().level == logging.DEBUG

    def test_setup_logger_preserves_existing_handlers(self):
        """setup_logger should not clear handlers it does not own."""
        root_logger = logging.getLogger()
        foreign_handler = logging.NullHandler()
        root_logger.addHandler(foreign_handler)

        setup_logger('INFO')

        assert foreign_handler in root_logger.handlers
