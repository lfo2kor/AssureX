import chromadb
import os
from dotenv import load_dotenv

load_dotenv()
chromadb_path = os.getenv("CHROMADB_PATH", "chromadb_data")  # Update if you use a different path

client = chromadb.PersistentClient(path=chromadb_path)

collection_name = "selectors_base_collection"
try:
    client.create_collection(name=collection_name)
    print(f"Collection '{collection_name}' created successfully.")
except Exception as e:
    print(f"Error creating collection: {e}")