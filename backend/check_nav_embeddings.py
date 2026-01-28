"""
Quick script to check navigation selector embeddings in ChromaDB
"""
import chromadb
from chromadb.config import Settings

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(
    path="./data/chromadb_llm",
    settings=Settings(anonymized_telemetry=False)
)

collection = chroma_client.get_collection(name="selectors_base_collection")

# Query for navigation-related selectors
results = collection.get(
    where={"attr": "data-test"},
    limit=1340
)

# Filter for sidebar navigation items
nav_selectors = []
for i, selector_id in enumerate(results['ids']):
    full_selector = results['metadatas'][i].get('full_selector', '')
    document = results['documents'][i]

    if 'sidebar-nav-item' in full_selector:
        nav_selectors.append({
            'id': selector_id,
            'selector': full_selector,
            'embedding_text': document,
            'module': results['metadatas'][i].get('module', '')
        })

print(f"Found {len(nav_selectors)} sidebar navigation selectors:\n")
print("=" * 100)

for nav in sorted(nav_selectors, key=lambda x: x['selector']):
    print(f"\nSelector: {nav['selector']}")
    print(f"Module: {nav['module']}")
    print(f"Embedded as: {nav['embedding_text']}")
    print("-" * 100)
