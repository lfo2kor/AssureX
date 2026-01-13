"""
PLCD Testing Assistant - Configuration Loader
Loads and validates YAML configuration file
"""

import yaml
import os
import re
from pathlib import Path
from typing import Dict, Any
from openai import AzureOpenAI
import chromadb
from chromadb.config import Settings


def _load_env_file(env_path: str = ".env"):
    """Load environment variables from .env file"""
    env_file = Path(env_path)
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()


def load_config(config_path: str = "plcdtestassistant.yaml") -> Dict[str, Any]:
    """
    Load and validate YAML configuration

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValueError: If required sections are missing
    """
    # Load .env file if it exists
    _load_env_file()

    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_file.absolute()}\n"
            f"Please ensure '{config_path}' exists in the project root."
        )

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")

    # Expand environment variables in config
    config = _expand_env_vars(config)

    # Validate configuration
    if not validate_config(config):
        raise ValueError("Configuration validation failed")

    return config


def _expand_env_vars(config: Any) -> Any:
    """
    Recursively expand environment variables in config.
    Supports ${VAR_NAME} syntax.
    """
    if isinstance(config, dict):
        return {k: _expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_expand_env_vars(item) for item in config]
    elif isinstance(config, str):
        # Match ${VAR_NAME} pattern
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, config)
        for var_name in matches:
            env_value = os.environ.get(var_name, '')
            if not env_value:
                raise ValueError(f"Environment variable '{var_name}' is not set")
            config = config.replace(f'${{{var_name}}}', env_value)
        return config
    else:
        return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate that all required sections exist in configuration

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, raises ValueError otherwise

    Raises:
        ValueError: If required sections are missing
    """
    required_sections = [
        'azure_openai',
        'vector_database',
        'selectors',
        'modules',
        'agent1_selector_discovery',
        'folders'
    ]

    missing_sections = []
    for section in required_sections:
        if section not in config:
            missing_sections.append(section)

    if missing_sections:
        raise ValueError(
            f"Missing required configuration sections: {', '.join(missing_sections)}"
        )

    # Validate Azure OpenAI section
    required_azure_fields = ['api_key', 'endpoint', 'api_version', 'models']
    for field in required_azure_fields:
        if field not in config['azure_openai']:
            raise ValueError(f"Missing required field in azure_openai: {field}")

    # Validate models exist
    required_models = ['chat', 'embedding', 'vision']
    for model in required_models:
        if model not in config['azure_openai']['models']:
            raise ValueError(f"Missing required model in azure_openai.models: {model}")

    # Validate selectors configuration
    if 'source_file' not in config['selectors']:
        raise ValueError("Missing 'source_file' in selectors configuration")

    # Validate vector database configuration
    if 'persist_directory' not in config['vector_database']:
        raise ValueError("Missing 'persist_directory' in vector_database configuration")

    return True


def get_azure_client(config: Dict[str, Any]) -> AzureOpenAI:
    """
    Initialize Azure OpenAI client from configuration

    Args:
        config: Configuration dictionary

    Returns:
        Initialized AzureOpenAI client

    Raises:
        ValueError: If Azure OpenAI configuration is invalid
    """
    try:
        azure_config = config['azure_openai']
        import httpx

        # Create httpx client WITH proxy support (needed for internal Bosch Azure endpoints)
        # Proxy settings will be picked up from environment variables automatically
        http_client = httpx.Client(
            verify=False,  # Disable SSL verification for corporate proxies
            timeout=30.0
            # proxies parameter not needed - httpx uses environment variables by default
        )

        client = AzureOpenAI(
            api_key=azure_config['api_key'],
            api_version=azure_config['api_version'],
            azure_endpoint=azure_config['endpoint'],
            http_client=http_client
        )

        return client

    except KeyError as e:
        raise ValueError(f"Missing Azure OpenAI configuration field: {e}")
    except Exception as e:
        raise ValueError(f"Error initializing Azure OpenAI client: {e}")


def get_selector_file_path(config: Dict[str, Any]) -> Path:
    """
    Get absolute path to selectors JSON file

    Args:
        config: Configuration dictionary

    Returns:
        Path to selectors file

    Raises:
        FileNotFoundError: If selectors file doesn't exist
    """
    selector_file = Path(config['selectors']['source_file'])

    if not selector_file.exists():
        raise FileNotFoundError(
            f"Selectors file not found: {selector_file.absolute()}\n"
            f"Please ensure the file exists at: {config['selectors']['source_file']}"
        )

    return selector_file


def get_chromadb_path(config: Dict[str, Any]) -> Path:
    """
    Get path to ChromaDB persistence directory

    Args:
        config: Configuration dictionary

    Returns:
        Path to ChromaDB directory
    """
    return Path(config['vector_database']['persist_directory'])


def get_embedding_model(config: Dict[str, Any]) -> str:
    """
    Get embedding model deployment name

    Args:
        config: Configuration dictionary

    Returns:
        Embedding model name
    """
    return config['azure_openai']['models']['embedding']


def get_batch_size(config: Dict[str, Any]) -> int:
    """
    Get embedding batch size

    Args:
        config: Configuration dictionary

    Returns:
        Batch size for embeddings
    """
    return config['azure_openai']['embedding_config']['batch_size']


def get_chroma_client(config: Dict[str, Any]) -> chromadb.Client:
    """
    Initialize ChromaDB client from configuration
    Uses same settings as Agent1 to avoid conflicts

    Args:
        config: Configuration dictionary

    Returns:
        Initialized ChromaDB client
    """
    persist_directory = get_chromadb_path(config)

    client = chromadb.PersistentClient(
        path=str(persist_directory),
        settings=Settings(anonymized_telemetry=False)
    )

    return client


# Test the configuration loader
if __name__ == "__main__":
    print("=" * 80)
    print("Testing Configuration Loader")
    print("=" * 80)

    try:
        # Load configuration
        print("\n[1/4] Loading configuration...")
        config = load_config()
        print(f"[OK] Configuration loaded successfully")

        # Test Azure client
        print("\n[2/4] Initializing Azure OpenAI client...")
        client = get_azure_client(config)
        print(f"[OK] Azure OpenAI client initialized")
        print(f"    Endpoint: {config['azure_openai']['endpoint']}")
        print(f"    API Version: {config['azure_openai']['api_version']}")

        # Test selectors file path
        print("\n[3/4] Checking selectors file...")
        selector_path = get_selector_file_path(config)
        print(f"[OK] Selectors file found: {selector_path}")

        # Test ChromaDB path
        print("\n[4/4] Checking ChromaDB path...")
        chroma_path = get_chromadb_path(config)
        print(f"[OK] ChromaDB path: {chroma_path}")

        print("\n" + "=" * 80)
        print("Configuration Summary:")
        print("=" * 80)
        print(f"Embedding Model: {get_embedding_model(config)}")
        print(f"Batch Size: {get_batch_size(config)}")
        print(f"Total Selectors: {config['selectors']['total_count']}")
        print(f"Known Modules: {len(config['modules']['known_modules'])}")
        print(f"Common Modules: {', '.join(config['modules']['common_modules'])}")

        print("\n[OK] All configuration tests passed!")

    except Exception as e:
        print(f"\n[ERROR] Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
