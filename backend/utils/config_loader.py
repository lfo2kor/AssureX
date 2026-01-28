"""
Configuration loader utility.

Loads and validates the plcdtest_config.yaml file, resolving relative paths
to absolute paths and validating required fields.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "plcdtest_config.yaml") -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file.

    Args:
        config_path: Path to configuration YAML file (default: plcdtest_config.yaml)

    Returns:
        Dictionary containing validated configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML syntax is invalid
        KeyError: If required configuration fields are missing
        ValueError: If configuration values are invalid
    """
    # Check if file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please create plcdtest_config.yaml in the project root."
        )

    # Load YAML file
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML syntax in {config_path}: {e}")

    # Validate required top-level fields
    required_fields = [
        'base_folder',
        'web_url',
        'browser',
        'login',
        'wait_times',
        'folders',
        'azure_openai',
        'execution'
    ]

    for field in required_fields:
        if field not in config:
            raise KeyError(
                f"Missing required configuration field: '{field}'\n"
                f"Please check your plcdtest_config.yaml file."
            )

    # Validate login credentials
    if 'username' not in config['login'] or 'password' not in config['login']:
        raise KeyError("Missing 'username' or 'password' in login configuration")

    # Validate Azure OpenAI config
    required_openai_fields = ['api_key', 'endpoint', 'api_version', 'deployment_gpt4o']
    for field in required_openai_fields:
        if field not in config['azure_openai']:
            raise KeyError(f"Missing '{field}' in azure_openai configuration")

    # Check if API key is placeholder
    if config['azure_openai']['api_key'] in ['YOUR_API_KEY_HERE', 'your_api_key_here', '']:
        raise ValueError(
            "Azure OpenAI API key is not configured.\n"
            "Please update 'azure_openai.api_key' in plcdtest_config.yaml with your actual API key."
        )

    # Resolve base_folder to absolute path
    base_folder = Path(config['base_folder']).resolve()
    if not base_folder.exists():
        raise FileNotFoundError(f"Base folder does not exist: {base_folder}")

    config['base_folder'] = str(base_folder)

    # Resolve relative folder paths to absolute paths
    folders = config['folders']
    for folder_key, folder_name in folders.items():
        # Convert relative path to absolute
        absolute_path = base_folder / folder_name
        config['folders'][folder_key] = str(absolute_path)

        # Create folder if it doesn't exist
        absolute_path.mkdir(parents=True, exist_ok=True)

    # Validate browser choice
    supported_browsers = ['edge', 'chromium', 'firefox', 'webkit']
    if config['browser'].lower() not in supported_browsers:
        raise ValueError(
            f"Unsupported browser: {config['browser']}\n"
            f"Supported browsers: {', '.join(supported_browsers)}"
        )

    # Validate wait_times are positive integers
    for wait_key, wait_value in config['wait_times'].items():
        if not isinstance(wait_value, int) or wait_value < 0:
            raise ValueError(
                f"Invalid wait time '{wait_key}': {wait_value}\n"
                f"Wait times must be positive integers (milliseconds)"
            )

    # Validate execution settings
    if config['execution']['max_retries'] < 1:
        raise ValueError("max_retries must be at least 1")

    return config


def get_folder_path(config: Dict[str, Any], folder_key: str) -> str:
    """
    Get absolute path for a configured folder.

    Args:
        config: Configuration dictionary
        folder_key: Key from config['folders'] (e.g., 'jira', 'reports', 'videos')

    Returns:
        Absolute path to the folder

    Raises:
        KeyError: If folder_key doesn't exist in configuration
    """
    if folder_key not in config['folders']:
        raise KeyError(f"Folder key '{folder_key}' not found in configuration")

    return config['folders'][folder_key]
