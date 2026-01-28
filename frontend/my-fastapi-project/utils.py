"""
Utility functions for the API
"""
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
import os


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Setup application logging
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Logger instance
    """
    # Create logs directory
    Path("Logs").mkdir(exist_ok=True)
    
    # Configure logging
    log_file = Path("Logs") / f"api_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("TestAutomationAPI")


def ensure_directories_exist():
    """
    Ensure all required directories exist
    """
    directories = [
        "Jira_Tickets",
        "Reports",
        "Videos",
        "Generated_Scripts",
        "Logs",
        "Reports/screenshots"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in MB
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    if not os.path.exists(file_path):
        return 0.0
    
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return round(size_mb, 2)


def validate_file_extension(filename: str, allowed_extensions: str) -> bool:
    """
    Validate file extension
    
    Args:
        filename: File name to validate
        allowed_extensions: Comma-separated allowed extensions (e.g., ".txt,.csv")
        
    Returns:
        True if valid, False otherwise
    """
    file_ext = Path(filename).suffix.lower()
    allowed = [ext.strip() for ext in allowed_extensions.split(",")]
    return file_ext in allowed


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.0f}s"
    
    hours = int(minutes // 60)
    remaining_minutes = minutes % 60
    
    return f"{hours}h {remaining_minutes}m"


def cleanup_old_files(folder: str, days_old: int = 7):
    """
    Cleanup files older than specified days
    
    Args:
        folder: Folder path to clean
        days_old: Delete files older than this many days
    """
    if not Path(folder).exists():
        return
    
    current_time = datetime.now()
    deleted_count = 0
    
    for file_path in Path(folder).rglob("*"):
        if file_path.is_file():
            file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
            
            if file_age.days > days_old:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
    
    return deleted_count