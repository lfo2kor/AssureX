"""
Application configuration using Pydantic Settings
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    """
    # external_project_path: str = r"C:\Idea Projects\AI_Test_Assist"
    external_project_path: str = r"C:\AssureX\backend"
    # Database
    database_url: str = "postgresql://postgres:postgres123@localhost:5432/test_automation"
    chromadb_path: str = "chromadb_data"   # <-- Add this line

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

# Add these fields for Azure OpenAI and Jira
    azure_openai_api_key: str = "98dkVOUDLG4wm9OCmF8pxnR48BoUCPYKzfI9p4zYGP5uVh7TiLLwJQQJ99BDAC5RqLJXJ3w3AAABACOGWUeW"
    azure_openai_endpoint: str = "https://ai2ets.openai.azure.com/"
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2024-02-15-preview"
    jira_base_url: str = "https://rb-tracker.bosch.com/tracker03"
    jira_email: str = "YOUR_JIRA_EMAIL"
    jira_api_token: str = "YOUR_JIRA_API_TOKEN"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"   # optional but recommended


# Global settings instance
settings = Settings()
