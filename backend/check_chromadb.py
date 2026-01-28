"""
Quick script to check ChromaDB collections
"""
import chromadb
from pathlib import Path

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./data/chromadb")

# List all collections
collections = chroma_client.list_collections()

print("=" * 80)
print("ChromaDB Status")
print("=" * 80)
print(f"\nTotal Collections: {len(collections)}\n")

for col in collections:
    print(f"Collection: {col.name}")
    print(f"  ID: {col.id}")
    print(f"  Count: {col.count()}")

    # Get sample metadata
    if col.count() > 0:
        sample = col.peek(limit=3)
        print(f"  Sample metadata keys: {list(sample['metadatas'][0].keys()) if sample['metadatas'] else 'None'}")
    print()

print("=" * 80)
