from dotenv import load_dotenv
load_dotenv()

from config_loader import get_azure_client
from config_loader import load_config


config = load_config()

# Add your debug prints here:
print("DEBUG: EMBEDDING MODEL:", config['azure_openai']['models']['embedding'])
print("DEBUG: API KEY:", config['azure_openai']['api_key'])
print("DEBUG: ENDPOINT:", config['azure_openai']['endpoint'])

azure_client = get_azure_client(config)

def generate_embedding(text: str):
    embedding_model = config['azure_openai']['models']['embedding']
    response = azure_client.embeddings.create(
        input=text,
        model=embedding_model
    )
    return response.data[0].embedding