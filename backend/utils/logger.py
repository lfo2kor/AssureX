"""
Logger utility with sensitive data masking.

Sets up logging for test automation with file and console handlers,
automatically masking passwords, API keys, and other sensitive data.
"""

import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Optional


def mask_sensitive_data(message: str) -> str:
    """
    Mask sensitive data in log messages.

    Masks patterns like:
    - Passwords: password='xxx' or password: xxx
    - API keys: api_key='xxx' or api_key: xxx
    - Credentials: credentials='xxx'
    - Bearer tokens: Bearer xxx

    Args:
        message: Log message that may contain sensitive data

    Returns:
        Message with sensitive data replaced by '***MASKED***'
    """
    patterns = [
        # password in various formats
        (r'password["\']?\s*[:=]\s*["\']?([^"\'\s,}]+)', r'password=***MASKED***'),
        # api_key in various formats
        (r'api_key["\']?\s*[:=]\s*["\']?([^"\'\s,}]+)', r'api_key=***MASKED***'),
        # credentials
        (r'credentials?["\']?\s*[:=]\s*["\']?([^"\'\s,}]+)', r'credentials=***MASKED***'),
        # Bearer tokens
        (r'Bearer\s+([^\s,}]+)', r'Bearer ***MASKED***'),
        # Generic secret patterns
        (r'secret["\']?\s*[:=]\s*["\']?([^"\'\s,}]+)', r'secret=***MASKED***'),
        (r'token["\']?\s*[:=]\s*["\']?([^"\'\s,}]+)', r'token=***MASKED***'),
    ]

    masked_message = message
    for pattern, replacement in patterns:
        masked_message = re.sub(pattern, replacement, masked_message, flags=re.IGNORECASE)

    return masked_message


class SensitiveDataFilter(logging.Filter):
    """Logging filter that masks sensitive data in all log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log record by masking sensitive data.

        Args:
            record: Log record to filter

        Returns:
            True (always, as we only modify the record, not filter it out)
        """
        record.msg = mask_sensitive_data(str(record.msg))
        # Also mask arguments if they exist
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: mask_sensitive_data(str(v)) if isinstance(v, str) else v
                             for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(mask_sensitive_data(str(arg)) if isinstance(arg, str) else arg
                                  for arg in record.args)
        return True


def setup_logger(
    log_folder: str,
    ticket_id: str,
    logger_name: str = "TA_AI_Project",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """
    Set up logger with file and console handlers.

    Creates a logger that:
    - Logs to console at INFO level
    - Logs to file at DEBUG level
    - Automatically masks sensitive data (passwords, API keys)
    - Uses consistent timestamp formatting

    Args:
        log_folder: Directory where log file will be created
        ticket_id: Jira ticket ID for log filename
        logger_name: Name for the logger (default: "TA_AI_Project")
        console_level: Logging level for console output (default: INFO)
        file_level: Logging level for file output (default: DEBUG)

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("C:/logs", "RBPLCD-8835")
        >>> logger.info("Starting test execution")
    """
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter

    # Remove existing handlers if any (prevent duplicate logs)
    logger.handlers.clear()

    # Create log directory if it doesn't exist
    Path(log_folder).mkdir(parents=True, exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{ticket_id}_{timestamp}.log"
    log_path = Path(log_folder) / log_filename

    # Create formatters
    detailed_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_formatter = logging.Formatter(
        '[%(levelname)s] %(message)s'
    )

    # Create file handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(detailed_formatter)
    file_handler.addFilter(SensitiveDataFilter())

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(SensitiveDataFilter())

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log initialization
    logger.info(f"Logger initialized for ticket {ticket_id}")
    logger.debug(f"Log file: {log_path}")

    return logger


def get_logger(logger_name: str = "TA_AI_Project") -> logging.Logger:
    """
    Get existing logger instance.

    Args:
        logger_name: Name of the logger to retrieve

    Returns:
        Logger instance
    """
    return logging.getLogger(logger_name)
