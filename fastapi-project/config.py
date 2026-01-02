"""
Application configuration using Pydantic Settings
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    """
    external_project_path: str = r"C:\Users\ENZ1KOR\ASSUREX"
    # Database
    database_url: str = "postgresql://postgres:postgres123@localhost:5432/test_automation"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    api_key: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    
    # File Upload
    max_file_size_mb: int = 10
    allowed_file_extensions: str = ".txt"
    
    # Test Execution
    max_concurrent_tests: int = 3
    
    # Paths
    jira_tickets_folder: str = "Jira_Tickets"
    reports_folder: str = "Reports"
    videos_folder: str = "Videos"
    scripts_folder: str = "Generated_Scripts"
    logs_folder: str = "Logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()