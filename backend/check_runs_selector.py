"""Check the Runs navigation selector metadata"""
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="./data/chromadb_llm",
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_collection(name="selectors_base_collection")

# Get all selectors and filter in Python
results = collection.get(limit=1340)

print("Searching for sidebar-nav-item-nav_item_teststeps...\n")

for i, full_selector in enumerate(results['metadatas']):
    if 'sidebar-nav-item-nav_item_teststeps' in full_selector.get('full_selector', ''):
        print(f"FOUND!")
        print(f"Full selector: {full_selector['full_selector']}")
        print(f"Module: {full_selector.get('module', 'N/A')}")
        print(f"Page context: {full_selector.get('page_context', 'N/A')}")
        print(f"Embedded text: {results['documents'][i]}")
        print(f"\nAll metadata fields:")
        for key, value in full_selector.items():
            print(f"  {key}: {value}")
        break
else:
    print("NOT FOUND in ChromaDB!")
