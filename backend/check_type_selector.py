"""Check Type selector in ChromaDB"""
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="./data/chromadb_llm",
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_collection(name="selectors_base_collection")

# Get all and filter
results = collection.get(limit=1340)

print("Searching for Type selectors...\n")
print("=" * 100)

for i, meta in enumerate(results['metadatas']):
    selector = meta.get('full_selector', '')
    if 'Type' in selector and 'data-attribute' in selector:
        print(f"\nSelector: {selector}")
        print(f"Embedded as: {results['documents'][i]}")
        print(f"Element: {meta.get('tagName', 'N/A')} / {meta.get('elementType', 'N/A')}")
        print(f"Module: {meta.get('module', 'N/A')}")
        print("-" * 100)
