"""Check if dynamic selector for autocomplete is in ChromaDB"""
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="./data/chromadb_llm",
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_collection(name="selectors_base_collection")

# Search for autocomplete selectors
results = collection.get(limit=1340)

print("Searching for data-autoCompleteItem selectors...\n")
print("=" * 100)

found = False
for i, meta in enumerate(results['metadatas']):
    selector = meta.get('full_selector', '')
    if 'autocompleteitem' in selector.lower() or 'data-autocompleteitem' in selector.lower():
        print(f"\nSelector: {selector}")
        print(f"Module: {meta.get('module', 'N/A')}")
        print(f"isDynamic: {meta.get('isDynamic', 'NOT FOUND')}")
        print(f"value: {meta.get('value', 'N/A')}")
        print(f"Document: {results['documents'][i][:150]}")
        print("-" * 100)
        found = True

if not found:
    print("No autocompleteitem selectors found!")
