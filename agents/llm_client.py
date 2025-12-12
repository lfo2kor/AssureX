"""
LLM Client Utility - Creates Azure OpenAI client from plcdtestassistant.yaml config.
"""

import sys
from pathlib import Path

# Add parent directory to path to import config_loader
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_loader import load_config, get_azure_client


def get_azure_openai_client():
    """
    Get Azure OpenAI client using credentials from plcdtestassistant.yaml.

    Returns:
        Tuple of (AzureOpenAI client instance, model name)
    """
    # Load configuration using the centralized config_loader
    config = load_config()

    # Get Azure client (with proxy support)
    client = get_azure_client(config)

    # Get chat model name
    model_name = config['azure_openai']['models']['chat']

    return client, model_name
