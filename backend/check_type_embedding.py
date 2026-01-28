"""Check what embedding was generated for Type dropdown selector"""
import chromadb
from chromadb.config import Settings
from pathlib import Path

# Load config to get correct path
from config_loader import load_config, get_chromadb_path

config = load_config()
chroma_path = get_chromadb_path(config)

print(f"Using ChromaDB at: {chroma_path}")

client = chromadb.PersistentClient(
    path=str(chroma_path),
    settings=Settings(anonymized_telemetry=False)
)

# Get collection
collection = client.get_collection(name="selectors_base_collection")

print(f"Collection count: {collection.count()}")
print()

# Get all results with attr containing 'Type'
print("Searching for selectors with 'Type' in value field...")
all_results = collection.get(
    include=["documents", "metadatas"]
)

# Filter for Type in value
results = {
    'ids': [],
    'documents': [],
    'metadatas': []
}

for doc_id, doc, meta in zip(all_results['ids'], all_results['documents'], all_results['metadatas']):
    if meta.get('value', '') == 'Type':
        results['ids'].append(doc_id)
        results['documents'].append(doc)
        results['metadatas'].append(meta)

print("=" * 100)
print(f"Found {len(results['ids'])} selector(s) with [data-attributegroupattribute='Type']")
print("=" * 100)

for i, (doc_id, doc, meta) in enumerate(zip(results['ids'], results['documents'], results['metadatas'])):
    print(f"\n{i+1}. ID: {doc_id}")
    print(f"   Natural Language: {doc}")
    print(f"   Full Selector: {meta.get('full_selector', 'N/A')}")
    print(f"   Module: {meta.get('module', 'N/A')}")
    print(f"   Context: {meta.get('context', 'N/A')}")
    print(f"   Element Type: {meta.get('elementType', 'N/A')}")
    print(f"   Label: {meta.get('label', 'N/A')}")

print("\n" + "=" * 100)
